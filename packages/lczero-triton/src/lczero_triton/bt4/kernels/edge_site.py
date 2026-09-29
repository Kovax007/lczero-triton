"""EGT2 edge update site (item E, K3): readback, rms, 16 -> 64 -> 16 FFN, rms.

The lab's EGT2 512x15 nets update their 16-channel edge state after encoder blocks 3, 7 and 11 (0-indexed). In the
gcap export the site after block 3 is nodes 10897-10923, after block 7 11487-11513, after block 11 12077-12103.
Per sample, with ``i`` the query square, ``j`` the key square, ``c`` an edge channel and ``h`` a head::

    e_hat = e + sum_h O[c, h] H[h, i, j]                    readback   /encoder{b}/edge_site/readback/w    [16, 32]
            (K4 extension point: the triplet adds W_o (va_in | va_out), built from rms(e_hat), to e_hat here)
    e_til = e_hat + W2 relu(W1 rms(e_hat) + b1)             edge FFN   /encoder{b}/edge_site/ffn/dense1/w  [64, 16]
                                                                       /encoder{b}/edge_site/ffn/dense1/b  [64]
                                                                       /encoder{b}/edge_site/ffn/dense2/w  [16, 64]
    e'    = rms(e_til),   rms(x) = x / sqrt(mean_c x^2 + 1e-6)

``reverse`` (the lab's round-32 ``rev_edge``, in the BT6-test sponsor net): the FFN's hidden layer also reads the
REVERSE edge through a second in-projection, ``relu(W1 rms(e_hat)[i, j] + b1 + Wr rms(e_hat)[j, i])``
(``/encoder{b}/edge_site/ffn/dense1_rev/w`` [64, 16]). The rms reduces over channels only, so the reversed read is the
rms of the cell mirrored across the board diagonal; a program loads that mirrored ``[16, pixels]`` tile as a gather.
Every program reads ``state`` and writes ``output``, which are distinct buffers, so the mirrored read never races.
A reverse site may run its programs over ``tile_rows x (pixels / tile_rows)`` TILES of the grid instead of runs of
consecutive cells: a run's mirror is a column (one cell per row of the grid, 64 floats apart), a square tile's mirror
is a square tile, so both reads stay short runs. The autotuner picks; without ``reverse`` only the run form competes.

The tables are R0's plans, FP32 and already ``[out, in]``. Every step is FP32 (the map's precision probe).

Layouts (fixed interfaces)
--------------------------
* ``logits`` (K2's H): FP32 ``[B * heads, 64, 64]``, index ``(sample * heads + head) * 4096 + 64 * i + j``, the
  post-door, pre-softmax logits of the block the site follows. FP16 with ``logits_f16`` (attention v9's FP16 export); the
  readback dot still runs in FP32 on the widened values.
* ``state`` (e) and ``output`` (e'): FP32 ``[B, 16, 64, 64]``, index ``sample * 16 * 4096 + c * 4096 + 64 * i + j``
  (K1's e0 layout). ``output`` is a separate buffer.

Form
----
One Triton program per (sample, run of ``pixels`` consecutive cells), planar. The run's ``[16, pixels]`` state tile and
``[32, pixels]`` logits tile are loaded once, and the three channel maps run as ``tl.dot`` over the pixel columns, so
the only buffer written is e'. The FFN hidden tile ``[64, pixels]`` lives in the program. Chosen over readback + FFN as
GEMMs over ``B * 4096`` interleaved rows by measurement (K3 report): the row form must also pack H and e into rows and
unpack e', and writes the 67 MB (batch 64) hidden buffer.

``tl.dot`` defaults to TF32 on FP32 inputs in this Triton (``default_dot_input_precision = "tf32"``), which rounds
the products; every dot here pins ``input_precision="ieee"``.

Stages: the K4 extension point
------------------------------
``stage="site"`` runs the whole site in one call (gcap). A triplet export instead calls ``stage="readback"`` (writes
e_hat to ``output``), then K4's triplet kernel (adds its branch into e_hat in place), then ``stage="ffn"`` (reads e_hat
from ``state``, writes e'). The triplet cannot live inside these programs: its softmaxes run over whole rows and
columns of the 64 x 64 grid, while a site program sees only its own cells.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

import torch
import triton
import triton.language as tl
from lc0ex import Buffer, KernelArtifact, ProgramBuilder
from lc0ex.proto import lc0ex_pb2
from lc0ex.triton_module_compiler import artifact_from_triton

from lczero_triton.bt4.kernels._cache import KernelCache

_POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER
_NULL_POINTER = lc0ex_pb2.PARAMETER_TYPE_NULL_POINTER
CELLS = 64 * 64
HEADS = 32
STATES = 16
HIDDEN = 64
# `const_1331` / `const_1338` (site after block 3) and their twins at the other sites.
EPSILON = 1e-6
SITE_BLOCKS = (3, 7, 11)
# The plan names each call reads, in kernel argument order, below `/encoder{b}/edge_site/` (`lab._names._plan_egt_site`).
SITE_TABLES = ("readback/w", "ffn/dense1/w", "ffn/dense1/b", "ffn/dense2/w")
# The reverse-edge in-projection (`rev_edge`), the kernel's last pointer; absent from a net without it.
REVERSE_TABLE = "ffn/dense1_rev/w"

Stage = Literal["site", "readback", "ffn"]
_STAGES: dict[str, int] = {"site": 0, "readback": 1, "ffn": 2}
_STAGE_READBACK = tl.constexpr(1)
_STAGE_FFN = tl.constexpr(2)
# Which of the kernel's first seven pointers each stage reads; a NULL slot takes no builder argument. The eighth
# pointer (the reverse-edge table) is read by the stages that run the FFN, and only on a `reverse` specialization.
_STAGE_POINTERS: dict[str, tuple[bool, ...]] = {
    "site": (True,) * 7,
    "readback": (True, True, True, True, False, False, False),
    "ffn": (True, True, False, False, True, True, True),
}
# (pixels per program, num_warps): the microbenchmark's finalists on sm_120 (K3 report). Batch 64: 32 cells x 2 warps
# 84.4 us, 64 x 2 92.0, 32 x 4 92.2; batch 8: 32 x 4 14.4, 64 x 4 14.5. Runs of 256 cells are slower (134.7 us at
# batch 64), 16 or 8 cells slower again, and 1024 cells took 432 s to compile.
_CONFIGURATIONS = ((32, 2), (32, 4), (64, 2), (64, 4))
# (pixels, tile rows, num_warps) of the tile form, which competes only on a `reverse` site: 8 x 8, 4 x 8, 4 x 4.
_TILE_CONFIGURATIONS = ((64, 8, 2), (64, 8, 4), (32, 4, 2), (32, 4, 4), (16, 4, 2), (16, 4, 4))


def site_table_names(after_block: int, *, reverse: bool = False) -> tuple[str, ...]:
    """Return the plan names one site reads, in kernel argument order: four, or five with the reverse-edge table."""
    tables = (*SITE_TABLES, REVERSE_TABLE) if reverse else SITE_TABLES
    return tuple(f"/encoder{after_block}/edge_site/{table}" for table in tables)


@triton.jit
def _rms_columns(values, states: tl.constexpr, epsilon: tl.constexpr):
    """``x / sqrt(mean_c x^2 + eps)`` over axis 0 of a ``[states, cells]`` tile, as the export's Mul/ReduceMean/Div."""
    mean_square = tl.sum(values * values, axis=0) * (1.0 / states)
    return values * (1.0 / tl.sqrt(mean_square + epsilon))[None, :]


@triton.jit
def _edge_site_body(  # noqa: PLR0913
    output,
    state,
    logits,
    readback,
    dense1_weight,
    dense1_bias,
    dense2_weight,
    dense1_reverse,
    copy_out,
    batch_count: tl.constexpr,  # noqa: ARG001  # Autotune key and launch grid.
    heads: tl.constexpr,
    states: tl.constexpr,
    hidden: tl.constexpr,
    stage: tl.constexpr,
    epsilon: tl.constexpr,
    reverse: tl.constexpr,
    pixels: tl.constexpr,
    tile_rows: tl.constexpr,
    dot_fp16: tl.constexpr = False,
    heads_pad: tl.constexpr = 0,
    copy_cell_major: tl.constexpr = False,
) -> None:
    """Update the edge state of one run (or, with ``tile_rows``, one tile) of ``pixels`` cells of one sample.

    ``dot_fp16`` (round 26 C2): the FFN's three maps run on the tensor cores -- weights and the rms'd state rounded
    once to FP16, FP32 accumulation, the class of the triplet's served `dot="fp16"` and of every FP16 GEMM. The rms,
    the bias, the relu and the residual stay FP32; the readback dot stays IEEE (it is memory-bound, and H's logits are
    large enough that FP16 would round them visibly).
    """
    program = tl.program_id(0)
    runs: tl.constexpr = 4096 // pixels
    sample = program // runs
    if tile_rows == 0:
        cells = (program % runs) * pixels + tl.arange(0, pixels)
    else:
        # A `tile_rows x tile_columns` tile of the 64 x 64 grid, tiles in row-major order; its mirror is a tile too.
        tile_columns: tl.constexpr = pixels // tile_rows
        tiles_per_row: tl.constexpr = 64 // tile_columns
        tile = program % runs
        local = tl.arange(0, pixels)
        cells = ((tile // tiles_per_row) * tile_rows + local // tile_columns) * 64 + (
            (tile % tiles_per_row) * tile_columns + local % tile_columns)
    channels = tl.arange(0, states)
    state_offsets = (sample * states + channels[:, None]) * 4096 + cells[None, :]
    values = tl.load(state + state_offsets)  # [states, pixels]
    if reverse:
        # The same cells mirrored across the diagonal, (i, j) -> (j, i): the reverse edge of every cell in the run.
        mirrored = (cells % 64) * 64 + cells // 64
        mirrored_values = tl.load(state + (sample * states + channels[:, None]) * 4096 + mirrored[None, :])
    if stage != _STAGE_FFN:
        if heads_pad == 0:
            head_index = tl.arange(0, heads)
            logit_values = tl.load(logits + (sample * heads + head_index[:, None]) * 4096 + cells[None, :]).to(tl.float32)
            readback_values = tl.load(readback + channels[:, None] * heads + head_index[None, :])
        else:
            # A head count that is not a power of two: the dot's K axis padded, the missing heads read as zero.
            head_index = tl.arange(0, heads_pad)
            live = head_index < heads
            logit_values = tl.load(logits + (sample * heads + head_index[:, None]) * 4096 + cells[None, :],
                                   mask=live[:, None], other=0.0).to(tl.float32)
            readback_values = tl.load(readback + channels[:, None] * heads + head_index[None, :],
                                      mask=live[None, :], other=0.0)
        values = tl.dot(readback_values, logit_values, values, input_precision="ieee")  # e_hat
        if reverse:
            # One-call site: `state` is still e, so the mirrored e_hat takes the mirrored readback too.
            if heads_pad == 0:
                mirrored_logits = tl.load(
                    logits + (sample * heads + head_index[:, None]) * 4096 + mirrored[None, :]).to(tl.float32)
            else:
                mirrored_logits = tl.load(logits + (sample * heads + head_index[:, None]) * 4096 + mirrored[None, :],
                                          mask=live[:, None], other=0.0).to(tl.float32)
            mirrored_values = tl.dot(readback_values, mirrored_logits, mirrored_values, input_precision="ieee")
    if stage == _STAGE_READBACK:
        # K4 extension point: the triplet kernel adds its branch to this e_hat, then stage "ffn" runs.
        tl.store(output + state_offsets, values)
        if copy_cell_major:
            # `triplet_mma`'s input: e_hat FP16 cell-major, [B, 64, 64, states] (a cell's channels contiguous)
            tl.store(copy_out + (sample * 4096 + cells[None, :]) * states + channels[:, None], values.to(tl.float16))
    else:
        units = tl.arange(0, hidden)
        weight1 = tl.load(dense1_weight + units[:, None] * states + channels[None, :])
        bias1 = tl.load(dense1_bias + units)
        weight2 = tl.load(dense2_weight + channels[:, None] * hidden + units[None, :])
        if dot_fp16:
            activation = tl.dot(weight1.to(tl.float16), _rms_columns(values, states, epsilon).to(tl.float16))
        else:
            activation = tl.dot(weight1, _rms_columns(values, states, epsilon), input_precision="ieee")
        activation = activation + bias1[:, None]
        if reverse:
            weight_reverse = tl.load(dense1_reverse + units[:, None] * states + channels[None, :])
            if dot_fp16:
                activation = tl.dot(weight_reverse.to(tl.float16),
                                    _rms_columns(mirrored_values, states, epsilon).to(tl.float16), activation)
            else:
                activation = tl.dot(weight_reverse, _rms_columns(mirrored_values, states, epsilon), activation,
                                    input_precision="ieee")
        activation = tl.maximum(activation, 0.0)
        if dot_fp16:
            values = tl.dot(weight2.to(tl.float16), activation.to(tl.float16), values)  # e_til
        else:
            values = tl.dot(weight2, activation, values, input_precision="ieee")  # e_til
        e_new = _rms_columns(values, states, epsilon)
        tl.store(output + state_offsets, e_new)
        if copy_cell_major:
            # v9's FP16 cell-major state copy of e', written here instead of by a separate cast after the site
            tl.store(copy_out + (sample * 4096 + cells[None, :]) * states + channels[:, None], e_new.to(tl.float16))


def _prune_site_configs(configs: list[triton.Config], named_args: dict[str, object], **kwargs: object) -> list[triton.Config]:
    """The tile form competes only where it can win: on a site that reads the reverse edge."""
    if named_args.get("reverse", kwargs.get("reverse")):
        return configs
    return [config for config in configs if config.kwargs["tile_rows"] == 0]


_edge_site_kernel = triton.autotune(
    configs=[triton.Config({"pixels": pixels, "tile_rows": 0}, num_warps=warps) for pixels, warps in _CONFIGURATIONS]
    + [triton.Config({"pixels": pixels, "tile_rows": rows}, num_warps=warps)
       for pixels, rows, warps in _TILE_CONFIGURATIONS],
    key=["batch_count", "heads", "states", "hidden", "stage", "reverse", "dot_fp16", "copy_cell_major"],
    prune_configs_by={"early_config_prune": _prune_site_configs},
    cache_results=True,
)(_edge_site_body)


@dataclass(frozen=True, slots=True)
class EdgeSiteSpecialization:
    """Immutable edge update site specialization (the gcap family's widths by default)."""

    batch_count: int
    architecture: int
    stage: Stage = "site"
    heads: int = HEADS
    states: int = STATES
    hidden: int = HIDDEN
    # The lab's `rev_edge`: the FFN also reads rms(e_hat)[j, i] through `ffn/dense1_rev/w`.
    reverse: bool = False
    # Round 26 C2: "ieee" (K3's FP32 class, the default) or "fp16" (the FFN's maps on the tensor cores).
    dot: str = "ieee"
    # H read as FP16 (attention v9's `h_f16` export); stages "site" and "readback" only.
    logits_f16: bool = False
    # The output also written as an FP16 cell-major copy, the kernel's ninth pointer: stage "readback" writes e_hat's
    # (`triplet_mma`'s input); stages "ffn" / "site" write e''s (v9's state copy, replacing `cast_state_cell_major`).
    copy_cell_major: bool = False

    def __post_init__(self) -> None:
        """Reject widths a ``tl.arange`` tile cannot take."""
        if not 0 < self.heads <= 64:  # noqa: PLR2004
            message = f"EdgeSiteSpecialization.heads={self.heads} must be in 1..64"
            raise ValueError(message)
        for name in ("states", "hidden"):
            width = getattr(self, name)
            if width <= 0 or width & (width - 1):
                message = f"EdgeSiteSpecialization.{name}={width} must be a power of two"
                raise ValueError(message)
        if self.stage not in _STAGES:
            message = f"unknown stage {self.stage!r}; expected one of {sorted(_STAGES)}"
            raise ValueError(message)
        if self.reverse and self.stage == "readback":
            message = "the reverse edge is read by the FFN; stage 'readback' does not run it"
            raise ValueError(message)
        if self.dot not in ("ieee", "fp16"):
            message = f"EdgeSiteSpecialization.dot={self.dot!r}; expected ieee or fp16"
            raise ValueError(message)
        if self.logits_f16 and self.stage == "ffn":
            message = "stage 'ffn' reads no logits; logits_f16 belongs to stages 'site' and 'readback'"
            raise ValueError(message)


def buffer_bytes(specialization: EdgeSiteSpecialization) -> dict[str, int]:
    """Return the buffers one call touches, in bytes: it writes only ``output``; no hidden buffer exists."""
    cells = specialization.batch_count * CELLS
    return {
        "output": 4 * specialization.states * cells,
        "state": 4 * specialization.states * cells,
        "logits": (2 if specialization.logits_f16 else 4) * specialization.heads * cells,
        "hidden": 0,
    }


def _autotune_grid(configuration: Mapping[str, object]) -> tuple[int]:
    """Return one program per (sample, run of cells)."""
    return (cast("int", configuration["batch_count"]) * CELLS // cast("int", configuration["pixels"]),)


def _heads_pad(heads: int) -> int:
    """0 for a power-of-two head count (the unpadded kernel), else the next power of two (at least 16, tl.dot)."""
    if heads & (heads - 1) == 0:
        return 0
    return max(16, 1 << heads.bit_length())


def compile_edge_site(specialization: EdgeSiteSpecialization) -> KernelArtifact:
    """Autotune and compile one edge update site specialization."""
    batch, states, heads, hidden = (
        specialization.batch_count, specialization.states, specialization.heads, specialization.hidden,
    )
    output = torch.empty((batch, states, 64, 64), dtype=torch.float32, device="cuda")
    state = torch.zeros((batch, states, 64, 64), dtype=torch.float32, device="cuda")
    logits_dtype = torch.float16 if specialization.logits_f16 else torch.float32
    logits = torch.zeros((batch * heads, 64, 64), dtype=logits_dtype, device="cuda")
    readback = torch.zeros((states, heads), dtype=torch.float32, device="cuda")
    dense1_weight = torch.zeros((hidden, states), dtype=torch.float32, device="cuda")
    dense1_bias = torch.zeros(hidden, dtype=torch.float32, device="cuda")
    dense2_weight = torch.zeros((states, hidden), dtype=torch.float32, device="cuda")
    dense1_reverse = torch.zeros((hidden, states), dtype=torch.float32, device="cuda")
    copy_out = torch.zeros((batch, 64, 64, states), dtype=torch.float16, device="cuda")
    compiled = _edge_site_kernel[_autotune_grid](
        output, state, logits, readback, dense1_weight, dense1_bias, dense2_weight, dense1_reverse, copy_out,
        batch, heads, states, hidden, _STAGES[specialization.stage], EPSILON, specialization.reverse,
        dot_fp16=specialization.dot == "fp16", heads_pad=_heads_pad(heads), copy_cell_major=specialization.copy_cell_major,
    )
    pixels = cast("int", _edge_site_kernel.best_config.kwargs["pixels"])
    parameters = tuple(
        _POINTER if used else _NULL_POINTER
        for used in (*_STAGE_POINTERS[specialization.stage], specialization.reverse, specialization.copy_cell_major)
    )
    return artifact_from_triton(
        compiled, grid=(batch * CELLS // pixels, 1, 1), parameters=parameters, autotuner=_edge_site_kernel,
    )


def edge_site(  # noqa: PLR0913
    builder: ProgramBuilder,
    kernels: KernelCache,
    output: Buffer,
    state: Buffer,
    logits: Buffer | None,
    tables: Mapping[str, Buffer],
    specialization: EdgeSiteSpecialization,
    *,
    after_block: int,
    copy: Buffer | None = None,
) -> None:
    """Append one site call: ``output = e'`` (stage "site" or "ffn") or ``output = e_hat`` (stage "readback").

    `tables` maps the plan names of `site_table_names(after_block, reverse=...)` to their persistent FP32 buffers.
    `logits` is the block's H in K2's layout; stage "ffn" takes None, because it reads e_hat from `state`.
    """
    if output is state:
        message = "the edge site writes e' to a separate buffer; output and state must differ"
        raise ValueError(message)
    used = (*_STAGE_POINTERS[specialization.stage], specialization.reverse, specialization.copy_cell_major)
    if (copy is not None) != specialization.copy_cell_major:
        message = "the FP16 cell-major copy is passed exactly when the specialization has copy_cell_major"
        raise ValueError(message)
    if (logits is not None) != used[2]:
        message = f"stage {specialization.stage!r} {'needs' if used[2] else 'takes no'} logits"
        raise ValueError(message)
    builder.set_target(lc0ex_pb2.Target.VENDOR_NVIDIA, f"sm_{specialization.architecture}")
    kernel = kernels.get(compile_edge_site, specialization)
    names = site_table_names(after_block, reverse=True)
    pointers = (output, state, logits, *(tables.get(name) for name in names), copy)
    arguments = tuple(buffer for buffer, needed in zip(pointers, used, strict=True) if needed)
    if any(buffer is None for buffer in arguments):
        message = f"stage {specialization.stage!r} (reverse={specialization.reverse}) is missing one of its tables"
        raise ValueError(message)
    builder.call(kernel, *arguments, readonly=[buffer for buffer in arguments[1:] if buffer is not copy])
