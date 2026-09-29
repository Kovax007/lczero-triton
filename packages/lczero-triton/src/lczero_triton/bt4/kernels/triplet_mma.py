"""EGT2 triplet operator on the tensor cores, delivered as a CUBIN (`triplet_mma.cu`): `triplet_site`'s form "fused" stage, faster.

One CTA per (sample, direction, triplet head) reads the sample's e_hat once from an FP16 CELL-MAJOR copy (a cell's 16 channels
contiguous; `edge_site`'s readback writes it with `copy_cell_major`), builds the head's six maps with one MMA per 16 cells, the
softmax x sigmoid in FP32, and va = A V on the tensor cores. It writes `va` in form "fused"'s layout, FP32 or (`va_f16`) FP16, which
`triplet_site.triplet_out` reads. Arithmetic class: form "fused" with `state_f16` and `dot="fp16"`.

Measured (4090, b64, standalone, against the served Triton fused stage writing FP32 va): 41.7 vs 54.6 us with FP32 va, 35.7 with
FP16 va (`tools/test_triplet_mma_standalone.py`).
"""

from dataclasses import dataclass
from pathlib import Path

from lc0ex import Buffer, KernelArtifact, ProgramBuilder
from lc0ex.cubin_module_compiler import artifact_from_cubin, compile_cuda
from lc0ex.proto import lc0ex_pb2

from lczero_triton.bt4.kernels._cache import KernelCache
from lczero_triton.bt4.kernels.triplet_site import _CONTRACTIONS, Contraction, triplet_table_names

_POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER
SOURCE = Path(__file__).with_name("triplet_mma.cu")
ENTRY_POINT = "triplet_mma"
THREADS = 512
# (sample, direction, triplet head) per CTA
UNITS_PER_SAMPLE = 2 * 4
# Gate controls (a build that MUST fail the fidelity gate): "no_swap" never transposes V (the contraction order ignored);
# "no_door_bias" drops the door's bias (measured too mild to fail: KL 2e-5).
CONTROLS = ("", "no_swap", "no_door_bias")


def shared_memory_bytes() -> int:
    """`kSmem` of `triplet_mma.cu` (checked there by `SMEM_BYTES`): logit + sigmoid(door) FP32, 4 value maps + A FP16."""
    return 2 * 64 * 65 * 4 + 4 * 64 * 72 * 2 + 64 * 72 * 2


@dataclass(frozen=True, slots=True)
class TripletMmaSpecialization:
    """One triplet-on-MMA specialization."""

    batch_size: int
    architecture: int
    contraction: Contraction = "path"
    # va stored FP16 (half the stage's write and `out`'s read); `out` must then be compiled with `va_f16`.
    va_f16: bool = True
    control: str = ""

    def __post_init__(self) -> None:
        """Refuse what the kernel does not implement."""
        if self.batch_size < 1:
            message = f"triplet_mma needs batch_size >= 1; got {self.batch_size}"
            raise ValueError(message)
        if self.contraction not in _CONTRACTIONS:
            message = f"triplet_mma contraction={self.contraction!r}; expected one of {sorted(_CONTRACTIONS)}"
            raise ValueError(message)
        if self.control not in CONTROLS:
            message = f"triplet_mma control={self.control!r}; expected one of {CONTROLS}"
            raise ValueError(message)


def compile_arguments(specialization: TripletMmaSpecialization) -> tuple[str, ...]:
    """The -D flags of one specialization."""
    flags = [f"-DSAMPLES={specialization.batch_size}", f"-DCONTRACTION={_CONTRACTIONS[specialization.contraction]}",
             f"-DVA_F16={int(specialization.va_f16)}", f"-DSMEM_BYTES={shared_memory_bytes()}"]
    if specialization.control == "no_door_bias":
        flags.append("-DCONTROL_NO_DOOR_BIAS")
    if specialization.control == "no_swap":
        flags.append("-DCONTROL_NO_SWAP")
    return tuple(flags)


def compile_triplet_mma(specialization: TripletMmaSpecialization) -> KernelArtifact:
    """Compile one specialization into a linker artifact (no autotuning: the configuration is fixed)."""
    cubin = compile_cuda(SOURCE.read_text(encoding="utf-8"), architecture=f"sm_{specialization.architecture}",
                         extra_arguments=(*compile_arguments(specialization), "-w"))
    return artifact_from_cubin(cubin, function=ENTRY_POINT, parameters=(_POINTER,) * 5,
                               grid=(specialization.batch_size * UNITS_PER_SAMPLE, 1, 1), block=(THREADS, 1, 1),
                               dynamic_shared_memory_bytes=shared_memory_bytes())


def triplet_mma(  # noqa: PLR0913
    builder: ProgramBuilder,
    kernels: KernelCache,
    va: Buffer,
    copy: Buffer,
    tables: dict[str, Buffer],
    specialization: TripletMmaSpecialization,
    *,
    after_block: int,
) -> None:
    """Append the stage: ``va`` (FP32 or FP16 ``[B, 32, 64, 64]``) from ``copy`` (e_hat, FP16 ``[B, 64, 64, 16]``).

    `tables` maps the plan names of `triplet_table_names(after_block)`; the value, gate and gate-bias plans are read.
    """
    if va is copy:
        message = "triplet_mma writes va to its own buffer"
        raise ValueError(message)
    builder.set_target(lc0ex_pb2.Target.VENDOR_NVIDIA, f"sm_{specialization.architecture}")
    value_weight, gate_weight, gate_bias, _ = (tables[name] for name in triplet_table_names(after_block))
    builder.call(kernels.get(compile_triplet_mma, specialization), va, copy, value_weight, gate_weight, gate_bias,
                 readonly=[copy, value_weight, gate_weight, gate_bias])
