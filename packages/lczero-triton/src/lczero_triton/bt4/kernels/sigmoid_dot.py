"""`sigmoid(x . w + b)` per row, FP32: the final layer of a value head's ERROR output (backend 10-13).

The lab value heads of the flagship (`value_head.py`) end in two parallel 128-wide layers: `wdl` (128 -> 3, served by
`matmul`) and, on the heads with `has_error_output`, `error` (128 -> 1) followed by a sigmoid. The dense `matmul`
family has no sigmoid, and adding one to it would change the source of every dense kernel of the net (Triton keys its
cache on the kernel's source), so the error layer has this kernel of its own:

    output[row] = 1 / (1 + exp(-(sum_k activations[row, k] * weights[k] + bias)))        row < row_count

`activations` is the head's FP16 hidden vector (`dense1` + Mish), `weights` [width, 1] and `bias` [1] are FP16
initializers, the product is accumulated in FP32 and the output is FP32 in 0..1 (lc0's `Sigmoid` after
`/error/matmul` + `/error/add` in `converter.cc`). One program per position; `width` must be a power of two (the lab
heads: 128). Nothing is tuned: the kernel is one 128-term dot product per position.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from lc0ex import Buffer, KernelArtifact, ProgramBuilder
from lc0ex.proto import lc0ex_pb2
from lc0ex.triton_module_compiler import artifact_from_triton

from lczero_triton.bt4.kernels._cache import KernelCache

_POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER


@triton.autotune(
    configs=[triton.Config({}, num_warps=1)],
    key=["row_count", "width"],
    cache_results=True,
)
@triton.jit
def _sigmoid_dot_kernel(
    output,
    activations,
    weights,
    bias,
    row_count: tl.constexpr,
    width: tl.constexpr,
) -> None:
    """One position: the FP32 dot product of its hidden row with the weight column, plus bias, through a sigmoid."""
    row = tl.program_id(0)
    columns = tl.arange(0, width)
    values = tl.load(activations + row * width + columns).to(tl.float32)
    weight_values = tl.load(weights + columns).to(tl.float32)
    logit = tl.sum(values * weight_values, axis=0) + tl.load(bias).to(tl.float32)
    tl.store(output + row, 1.0 / (1.0 + tl.exp(-logit)))


@dataclass(frozen=True, slots=True)
class SigmoidDotSpecialization:
    """Immutable `sigmoid(x . w + b)` specialization: `row_count` positions of `width` FP16 values each."""

    row_count: int
    width: int
    architecture: int


def compile_sigmoid_dot(specialization: SigmoidDotSpecialization) -> KernelArtifact:
    """Compile one `sigmoid(x . w + b)` specialization (a single configuration: nothing is autotuned)."""
    rows, width = specialization.row_count, specialization.width
    if width < 1 or width & (width - 1):
        message = f"sigmoid_dot: width {width} is not a power of two"
        raise ValueError(message)
    output = torch.empty(rows, dtype=torch.float32, device="cuda")
    activations = torch.zeros((rows, width), dtype=torch.float16, device="cuda")
    weights = torch.zeros((width, 1), dtype=torch.float16, device="cuda")
    bias = torch.zeros(1, dtype=torch.float16, device="cuda")
    compiled = _sigmoid_dot_kernel[(rows,)](output, activations, weights, bias, rows, width)
    return artifact_from_triton(
        compiled,
        grid=(rows, 1, 1),
        parameters=(_POINTER, _POINTER, _POINTER, _POINTER),
        autotuner=_sigmoid_dot_kernel,
    )


def sigmoid_dot(  # noqa: PLR0913
    builder: ProgramBuilder,
    kernels: KernelCache,
    output: Buffer,
    activations: Buffer,
    weights: Buffer,
    bias: Buffer,
    specialization: SigmoidDotSpecialization,
) -> None:
    """Append `output = sigmoid(activations . weights + bias)`, one FP32 value per row."""
    builder.set_target(
        lc0ex_pb2.Target.VENDOR_NVIDIA,
        f"sm_{specialization.architecture}",
    )
    kernel = kernels.get(compile_sigmoid_dot, specialization)
    builder.call(kernel, output, activations, weights, bias, readonly=[activations, weights, bias])
