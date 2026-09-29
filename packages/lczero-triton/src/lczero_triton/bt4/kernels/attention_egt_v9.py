"""EGT2 attention, v9: per-warp head pipelines over a half-position, delivered as a CUBIN (`egt_v9.cu`).

The H3 shape's attention (32 heads x 32, the H3 form: output gate, int8 `attn_out` codes or an FP16 output, the FP16 state copy).
The kernel and its measurements are described in `egt_v9.cu`; the design and the gates in
`Kovax/briefs_2026-09-28/PLAN_backend_h3b_head_group_mma_attention_port_goal_steps_gates_0928.md` and the v9 report beside it.

It reads what the served kernel reads -- packed ``[q | k | v | g]``, K1's packed E and S, the block's plans in `BLOCK_TABLES`
order, the `attn_out` prescale -- with two differences: the state is the FP16 copy CELL-MAJOR
(`egt_state_tiles.cast_state_cell_major`), and the query / key tables, the read / door / gate weights and the constant bias are
read as FP16 packs that `pack_egt_v9_tables` writes from the FP32 plans into the block's own persistent buffers at the start of
every batch (reading the FP32 plans inside the kernel measured 83.5 against 73.3 us at b64: the paired FP32 loads at 128
registers). No carrier change: a promotion stays a new `.pb.gz` / carrier. It does not read the edge list.

Compile-time per rung: the batch (the work list), the build device's SM count (the persistent grid), the block's gate scale.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
import triton
import triton.language as tl

from lc0ex import Buffer, KernelArtifact, ProgramBuilder
from lc0ex.cubin_module_compiler import artifact_from_cubin, compile_cuda
from lc0ex.proto import lc0ex_pb2
from lc0ex.triton_module_compiler import artifact_from_triton

from lczero_triton.bt4.kernels._cache import KernelCache
from lczero_triton.bt4.kernels.attention_egt import BLOCK_TABLES

_POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER
_NULL_POINTER = lc0ex_pb2.PARAMETER_TYPE_NULL_POINTER
SOURCE = Path(__file__).with_name("egt_v9.cu")
ENTRY_POINT = "egt_v9_attention"
HEADS_IN_FLIGHT = 4
THREADS = 128 * HEADS_IN_FLIGHT
ROWS = 32
LIST_CAPACITY = 512
# Gate controls (a build that MUST fail the fidelity gate): "no_sigma" drops E's scaled-coefficient term.
CONTROLS = ("", "no_sigma")


def shared_memory_bytes(head_dim: int) -> int:
    """`kSmem` of `egt_v9.cu` (checked there by `SMEM_BYTES`)."""
    state = 2 * ROWS * (64 * 16 + 16)
    tables = (ROWS + 64) * 40 * 2 + 40 * 4
    exchange = 4 * (32 + 16 * (head_dim // 2)) * 4
    return state + HEADS_IN_FLIGHT * max(tables, exchange) + 128 + 4 * LIST_CAPACITY


def supports(heads: int, head_dim: int, *, output_gate: bool) -> bool:
    """The shapes v9 serves: 4 heads in flight, head_dim 16 or 32 (at 64 it spills and loses to the Triton kernel), the H3 form."""
    return output_gate and heads % HEADS_IN_FLIGHT == 0 and head_dim in (16, 32)


@dataclass(frozen=True, slots=True)
class AttentionEgtV9Specialization:
    """One v9 attention specialization. `sms` is the build device's SM count (the persistent grid's size)."""

    batch_size: int
    heads: int
    head_dim: int
    architecture: int
    sms: int
    cap: bool
    output_gate: bool = True
    gate_scale: float = 2.0
    export_h: bool = False
    # True: the out-projection's int8 codes through `quant_prescale` (Q1); False: FP16 output (the FP16 twin, or a conversion pass).
    quant_output: bool = True
    # B9's gated arithmetic class: FP16 accumulate in the MMAs; 2 sigmoid(x) = 1 + tanh.approx(x / 2).
    accumulate_f16: bool = True
    gate_tanh: bool = True
    # The odd head slots start this many cycles late so that the pipes overlap (measured best on the 4090).
    skew: int = 6000
    control: str = ""
    # The FP16 packs of `pack_egt_v9_tables` (the served form); False reads the FP32 plans in the kernel (a test switch).
    packed: bool = True

    def __post_init__(self) -> None:
        """Refuse what the kernel does not implement."""
        if not supports(self.heads, self.head_dim, output_gate=self.output_gate):
            message = (f"attention v9 serves heads % {HEADS_IN_FLIGHT} == 0, head_dim 16 / 32, with the output gate; got "
                       f"{self.heads} x {self.head_dim}, output_gate={self.output_gate}")
            raise ValueError(message)
        if self.batch_size < 1 or self.sms < 1:
            message = f"attention v9 needs batch_size >= 1 and sms >= 1; got {self.batch_size}, {self.sms}"
            raise ValueError(message)
        if self.control not in CONTROLS:
            message = f"attention v9 control={self.control!r}; expected one of {CONTROLS}"
            raise ValueError(message)
        if shared_memory_bytes(self.head_dim) > 101376:
            message = "attention v9's shared memory is over the sm_89 / sm_120 opt-in ceiling"
            raise ValueError(message)

    @property
    def grid(self) -> int:
        """One CTA per SM, at most one per (half-position, head group) unit."""
        return min(2 * self.batch_size * (self.heads // HEADS_IN_FLIGHT), self.sms)


def compile_arguments(specialization: AttentionEgtV9Specialization) -> tuple[str, ...]:
    """The -D flags of one specialization."""
    flags = [
        f"-DHEADS={specialization.heads}",
        f"-DHD={specialization.head_dim}",
        f"-DHALVES={2 * specialization.batch_size}",
        f"-DGATE_SCALE={float(specialization.gate_scale)!r}f",
        f"-DOGATE={int(specialization.output_gate)}",
        f"-DCAP={int(specialization.cap)}",
        f"-DEXPORT_H={int(specialization.export_h)}",
        f"-DOUT_F16={int(not specialization.quant_output)}",
        f"-DSKEW={specialization.skew}",
        f"-DSMEM_BYTES={shared_memory_bytes(specialization.head_dim)}",
    ]
    if specialization.accumulate_f16:
        flags.append("-DACC16")
    if specialization.gate_tanh:
        flags.append("-DGATE_TANH")
    if specialization.control == "no_sigma":
        flags.append("-DCONTROL_NO_SIGMA")
    if specialization.packed:
        flags.append("-DPACKED")
    return tuple(flags)


def compile_attention_egt_v9(specialization: AttentionEgtV9Specialization) -> KernelArtifact:
    """Compile one v9 specialization into a linker artifact (no autotuning: the configuration is fixed)."""
    cubin = compile_cuda(SOURCE.read_text(encoding="utf-8"), architecture=f"sm_{specialization.architecture}",
                         extra_arguments=(*compile_arguments(specialization), "-w"))
    # BLOCK_TABLES order; packed, the read slot carries all three weight vectors and the door / gate slots are unused.
    tables = tuple(_NULL_POINTER if specialization.packed and name in ("door/w", "gate/w") else _POINTER
                   for name in BLOCK_TABLES)
    parameters = (
        _POINTER,                                                     # output
        _POINTER if specialization.export_h else _NULL_POINTER,        # H
        *(_POINTER,) * 4,                                              # qkv, edges, S, state (cell-major)
        *tables,
        _POINTER if specialization.quant_output else _NULL_POINTER,    # the attn_out prescale
    )
    return artifact_from_cubin(cubin, function=ENTRY_POINT, parameters=parameters, grid=(specialization.grid, 1, 1),
                               block=(THREADS, 1, 1), dynamic_shared_memory_bytes=shared_memory_bytes(specialization.head_dim))


# ---------------------------------------------------------------- the FP16 packs, written from the FP32 plans every batch
PACKED_TABLES = ("query", "key", "weights", "constant_bias")


@triton.autotune(configs=[triton.Config({}, num_warps=warps) for warps in (4, 8)], key=["heads", "depth"], cache_results=True)
@triton.jit
def _pack_tables_kernel(  # noqa: PLR0913
    query16, key16, weights16, cbias16, query, key, read, door, gate, cbias,
    heads: tl.constexpr,  # noqa: ARG001  # Autotune key and launch grid.
    depth: tl.constexpr,
) -> None:
    """One head: query / key [34, depth] FP32 -> [40, depth] FP16 (zero rows 34-39); read | door | gate [16] each -> 48 FP16;
    constant bias [4096] FP32 -> FP16."""
    head = tl.program_id(0)
    channel = tl.arange(0, 64)[:, None]
    dim = tl.arange(0, depth)[None, :]
    source = (head * 34 + channel) * depth + dim
    target = (head * 40 + channel) * depth + dim
    real, stored = channel < 34, channel < 40
    tl.store(query16 + target, tl.load(query + source, mask=real, other=0.0).to(tl.float16), mask=stored)
    tl.store(key16 + target, tl.load(key + source, mask=real, other=0.0).to(tl.float16), mask=stored)
    state = tl.arange(0, 16)
    tl.store(weights16 + head * 48 + state, tl.load(read + head * 16 + state).to(tl.float16))
    tl.store(weights16 + head * 48 + 16 + state, tl.load(door + head * 16 + state).to(tl.float16))
    tl.store(weights16 + head * 48 + 32 + state, tl.load(gate + head * 16 + state).to(tl.float16))
    cells = tl.arange(0, 4096)
    tl.store(cbias16 + head * 4096 + cells, tl.load(cbias + head * 4096 + cells).to(tl.float16))


@dataclass(frozen=True, slots=True)
class PackEgtV9Specialization:
    """The FP16 packs of one block shape."""

    heads: int
    head_dim: int
    architecture: int


def packed_bytes(heads: int, head_dim: int) -> dict[str, int]:
    """Bytes of each FP16 pack (`PACKED_TABLES`)."""
    return {"query": 2 * heads * 40 * head_dim, "key": 2 * heads * 40 * head_dim, "weights": 2 * heads * 48,
            "constant_bias": 2 * heads * 4096}


def launch_pack_egt_v9_tables(packs: dict[str, torch.Tensor], tables: dict[str, torch.Tensor], heads: int, head_dim: int) -> object:
    """Launch on torch tensors (tests and `compile_pack_egt_v9_tables`)."""
    return _pack_tables_kernel[(heads,)](
        packs["query"], packs["key"], packs["weights"], packs["constant_bias"], tables["query"], tables["key"],
        tables["edge_read/w"], tables["door/w"], tables["gate/w"], tables["constant_bias"], heads, head_dim)


def compile_pack_egt_v9_tables(specialization: PackEgtV9Specialization) -> KernelArtifact:
    """Autotune and compile the pack kernel of one block shape."""
    heads, depth = specialization.heads, specialization.head_dim
    packs = {name: torch.empty(size // 2, dtype=torch.float16, device="cuda")
             for name, size in packed_bytes(heads, depth).items()}
    tables = {"query": torch.zeros(heads, 34, depth, device="cuda"), "key": torch.zeros(heads, 34, depth, device="cuda"),
              "edge_read/w": torch.zeros(heads, 16, device="cuda"), "door/w": torch.zeros(heads, 16, device="cuda"),
              "gate/w": torch.zeros(heads, 16, device="cuda"), "constant_bias": torch.zeros(heads, 64, 64, device="cuda")}
    compiled = launch_pack_egt_v9_tables(packs, tables, heads, depth)
    return artifact_from_triton(compiled, grid=(heads, 1, 1), parameters=(_POINTER,) * 10, autotuner=_pack_tables_kernel)


def pack_egt_v9_tables(
    builder: ProgramBuilder,
    kernels: KernelCache,
    packs: dict[str, Buffer],
    tables: dict[str, Buffer],
    specialization: PackEgtV9Specialization,
) -> None:
    """Append the pack call of one block: `packs` (PACKED_TABLES) from the block's FP32 plans (BLOCK_TABLES' short names)."""
    builder.set_target(lc0ex_pb2.Target.VENDOR_NVIDIA, f"sm_{specialization.architecture}")
    kernel = kernels.get(compile_pack_egt_v9_tables, specialization)
    reads = tuple(tables[name] for name in ("query", "key", "edge_read/w", "door/w", "gate/w", "constant_bias"))
    builder.call(kernel, *(packs[name] for name in PACKED_TABLES), *reads, readonly=reads)


def served_tables(tables: dict[str, Buffer], packs: dict[str, Buffer]) -> dict[str, Buffer]:
    """The pointers a packed v9 call takes, keyed by BLOCK_TABLES' short names (door / gate are unused slots)."""
    return tables | {"query": packs["query"], "key": packs["key"], "edge_read/w": packs["weights"],
                     "constant_bias": packs["constant_bias"]}


def attention_egt_v9(  # noqa: PLR0913
    builder: ProgramBuilder,
    kernels: KernelCache,
    output: Buffer,
    qkv: Buffer,
    edges: Buffer,
    edge_norm: Buffer,
    state_cell_major: Buffer,
    tables: dict[str, Buffer],
    specialization: AttentionEgtV9Specialization,
    logits: Buffer | None = None,
    quant_prescale: Buffer | None = None,
) -> None:
    """Append one v9 attention block. `tables` maps BLOCK_TABLES' short names to the block's buffers: the FP32 plans, or with
    `specialization.packed` the `served_tables` view (the packs in the query / key / read / constant-bias slots)."""
    if specialization.export_h != (logits is not None):
        message = "export_h and a logits buffer must come together"
        raise ValueError(message)
    if specialization.quant_output != (quant_prescale is not None):
        message = "quant_output and a quant_prescale buffer must come together"
        raise ValueError(message)
    builder.set_target(lc0ex_pb2.Target.VENDOR_NVIDIA, f"sm_{specialization.architecture}")
    kernel = kernels.get(compile_attention_egt_v9, specialization)
    used = [name for name in BLOCK_TABLES if not (specialization.packed and name in ("door/w", "gate/w"))]
    reads = (qkv, edges, edge_norm, state_cell_major, *(tables[name] for name in used))
    if quant_prescale is not None:
        reads = (*reads, quant_prescale)
    written = (output, logits) if logits is not None else (output,)
    builder.call(kernel, *written, *reads, readonly=reads)
