"""Fused FP16 activation, residual, and layer-normalization family."""

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

Activation = Literal["none", "mish", "swish"]

_ACTIVATION_NONE = 0
_ACTIVATION_MISH = tl.constexpr(1)
_ACTIVATION_SWISH = tl.constexpr(2)
_ACTIVATIONS: dict[Activation, int] = {
    "none": _ACTIVATION_NONE,
    "mish": 1,
    "swish": 2,
}
_MISH_BRANCH = tl.constexpr(-0.6)
_POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER
_WARP_COUNTS = (
    (1, 0),
    (2, 0),
    (4, 0),
    (8, 0),
    (16, 0),
    (1, 1),
    (2, 1),
    (4, 1),
    (8, 1),
    (16, 1),
)

def _layer_norm_configs() -> list[triton.Config]:
    """Return independent warp-count candidates for one normalized row."""
    return [triton.Config({"online_algorithm": online}, num_warps=num_warps)
            for num_warps, online in _WARP_COUNTS]


@triton.jit
def welford_combine(mean_a, m2_a, weight_a, mean_b, m2_b, weight_b, use_mask: tl.constexpr):
    delta = mean_b - mean_a
    total_weight = weight_a + weight_b
    if use_mask:
        weight_b_share = tl.where(total_weight == 0.0, 0.0, weight_b / total_weight)
    else:
        weight_b_share = weight_b / total_weight
    new_mean = mean_a + delta * weight_b_share
    new_m2 = m2_a + m2_b + (delta * weight_a) * (delta * weight_b_share)
    return new_mean, new_m2, total_weight

@triton.jit
def welford_combine_masked(mean_a, m2_a, weight_a, mean_b, m2_b, weight_b):
    return welford_combine(mean_a, m2_a, weight_a, mean_b, m2_b, weight_b, use_mask=True)

@triton.jit
def welford_combine_unmasked(mean_a, m2_a, weight_a, mean_b, m2_b, weight_b):
    return welford_combine(mean_a, m2_a, weight_a, mean_b, m2_b, weight_b, use_mask=False)

@triton.jit
def _layer_norm_row(  # noqa: PLR0915
    output,
    input_,
    bias,
    skip,
    gammas,
    betas,
    alpha,
    row_count: tl.constexpr,
    width: tl.constexpr,
    epsilon: tl.constexpr,
    activation: tl.constexpr,
    has_bias: tl.constexpr,
    has_skip: tl.constexpr,
    block_size: tl.constexpr,
    online_algorithm: tl.constexpr
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    pointers = row * width + offsets

    if width == block_size:
        values = tl.load(input_ + pointers).to(tl.float32)
        if has_bias:
            biases = tl.load(bias + offsets).to(tl.float32)
            values += biases

        if activation == _ACTIVATION_MISH:
            exponential = tl.exp(values)
            numerator = exponential * exponential + 2.0 * exponential
            division = values / (numerator + 2.0)
            values = tl.where(
                values <= _MISH_BRANCH,
                numerator * division,
                values - 2.0 * division,
            )
        elif activation == _ACTIVATION_SWISH:
            values /= 1.0 + tl.exp(-values)

        if has_skip:
            alpha_value = tl.load(alpha).to(tl.float32)
            skip_values = tl.load(skip + pointers).to(tl.float32)
            values = alpha_value * values + skip_values

        inverse_width = 1.0 / width

        if online_algorithm:
            mean = values
            m2 = tl.zeros_like(values)
            weight = tl.full(values.shape, 1.0, dtype=tl.float32)

            mean, m2, weight = tl.reduce((mean, m2, weight), axis=0, combine_fn=welford_combine_unmasked)

            variance = tl.maximum(m2 * inverse_width, 0.0)
            centered = values - mean
        else:
            mean = tl.sum(values, axis=0) * inverse_width
            centered = values - mean
            variance = tl.sum(centered * centered, axis=0) * inverse_width

        inverse_stddev = 1.0 / tl.sqrt(variance + epsilon)
        normalized = centered * inverse_stddev

        gamma_values = tl.load(gammas + offsets).to(tl.float32)
        beta_values = tl.load(betas + offsets).to(tl.float32)
        result = normalized * gamma_values + beta_values
        tl.store(output + pointers, result.to(tl.float16))
    else:
        valid = (row < row_count) & (offsets < width)
        values = tl.load(input_ + pointers, mask=valid, other=0.0).to(tl.float32)
        if has_bias:
            biases = tl.load(bias + offsets, mask=valid, other=0.0).to(tl.float32)
            values += biases

        if activation == _ACTIVATION_MISH:
            exponential = tl.exp(values)
            numerator = exponential * exponential + 2.0 * exponential
            division = values / (numerator + 2.0)
            values = tl.where(
                values <= _MISH_BRANCH,
                numerator * division,
                values - 2.0 * division,
            )
        elif activation == _ACTIVATION_SWISH:
            values /= 1.0 + tl.exp(-values)

        if has_skip:
            alpha_value = tl.load(alpha).to(tl.float32)
            skip_values = tl.load(skip + pointers, mask=valid, other=0.0).to(tl.float32)
            values = alpha_value * values + skip_values

        inverse_width = 1.0 / width

        if online_algorithm:
            mean = values
            m2 = tl.zeros_like(values)
            weight = tl.where(valid, 1.0, 0.0)

            mean, m2, weight = tl.reduce((mean, m2, weight), axis=0, combine_fn=welford_combine_masked)

            variance = tl.maximum(m2 * inverse_width, 0.0)
            centered = values - mean
        else:
            values = tl.where(valid, values, 0.0)
            mean = tl.sum(values, axis=0) * inverse_width
            centered = tl.where(valid, values - mean, 0.0)
            variance = tl.sum(centered * centered, axis=0) * inverse_width

        inverse_stddev = 1.0 / tl.sqrt(variance + epsilon)
        normalized = centered * inverse_stddev

        gamma_values = tl.load(gammas + offsets, mask=valid, other=0.0).to(tl.float32)
        beta_values = tl.load(betas + offsets, mask=valid, other=0.0).to(tl.float32)
        result = normalized * gamma_values + beta_values
        tl.store(output + pointers, result.to(tl.float16), mask=valid)


@triton.autotune(
    configs=_layer_norm_configs(),
    key=["row_count", "width", "epsilon", "activation", "has_bias"],
    cache_results=True,
)
@triton.jit
def _layer_norm_kernel(
    output,
    input_,
    bias,
    gammas,
    betas,
    row_count: tl.constexpr,
    width: tl.constexpr,
    epsilon: tl.constexpr,
    activation: tl.constexpr,
    has_bias: tl.constexpr,
    block_size: tl.constexpr,
    online_algorithm: tl.constexpr,
) -> None:
    """Normalize rows without a residual connection or runtime alpha."""
    _layer_norm_row(
        output,
        input_,
        bias,
        input_,
        gammas,
        betas,
        gammas,
        row_count,
        width,
        epsilon,
        activation,
        has_bias,
        0,
        block_size,
        online_algorithm,
    )


@triton.autotune(
    configs=_layer_norm_configs(),
    key=["row_count", "width", "epsilon", "activation", "has_bias"],
    cache_results=True,
)
@triton.jit
def _layer_norm_skip_kernel(
    output,
    input_,
    bias,
    skip,
    gammas,
    betas,
    alpha,
    row_count: tl.constexpr,
    width: tl.constexpr,
    epsilon: tl.constexpr,
    activation: tl.constexpr,
    has_bias: tl.constexpr,
    block_size: tl.constexpr,
    online_algorithm: tl.constexpr,
) -> None:
    """Normalize rows after alpha scaling and a residual connection."""
    _layer_norm_row(
        output,
        input_,
        bias,
        skip,
        gammas,
        betas,
        alpha,
        row_count,
        width,
        epsilon,
        activation,
        has_bias,
        1,
        block_size,
        online_algorithm,
    )


@dataclass(frozen=True, slots=True)
class LayerNormSpecialization:
    """Immutable contiguous FP16 layer-normalization specialization."""

    row_count: int
    width: int
    activation: Activation
    has_skip: bool
    architecture: int
    has_bias: bool = True
    epsilon: float = 1e-3


def _autotune_grid(configuration: Mapping[str, object]) -> tuple[int]:
    """Return one Triton program per normalized row."""
    return (cast("int", configuration["row_count"]),)


def _artifact_grid(row_count: int) -> tuple[int, int, int]:
    """Return the serialized one-program-per-row launch grid."""
    return (row_count, 1, 1)


def compile_layer_norm(
    specialization: LayerNormSpecialization,
) -> KernelArtifact:
    """Autotune and compile one fused FP16 layer-normalization variant."""
    shape = (specialization.row_count, specialization.width)
    output = torch.empty(shape, dtype=torch.float16, device="cuda")
    input_ = torch.zeros(shape, dtype=torch.float16, device="cuda")
    bias = torch.zeros(specialization.width, dtype=torch.float16, device="cuda")
    gammas = torch.ones(specialization.width, dtype=torch.float16, device="cuda")
    betas = torch.zeros(specialization.width, dtype=torch.float16, device="cuda")
    block_size = triton.next_power_of_2(specialization.width)
    activation = _ACTIVATIONS[specialization.activation]
    parameters: tuple[int, ...]

    if specialization.has_skip:
        skip = torch.zeros(shape, dtype=torch.float16, device="cuda")
        alpha = torch.ones(1, dtype=torch.float16, device="cuda")
        compiled = _layer_norm_skip_kernel[_autotune_grid](
            output,
            input_,
            bias,
            skip,
            gammas,
            betas,
            alpha,
            specialization.row_count,
            specialization.width,
            specialization.epsilon,
            activation,
            specialization.has_bias,
            block_size,
        )
        autotuner = _layer_norm_skip_kernel
        parameters = (_POINTER,) * 7
    else:
        compiled = _layer_norm_kernel[_autotune_grid](
            output,
            input_,
            bias,
            gammas,
            betas,
            specialization.row_count,
            specialization.width,
            specialization.epsilon,
            activation,
            specialization.has_bias,
            block_size,
        )
        autotuner = _layer_norm_kernel
        parameters = (_POINTER,) * 5

    return artifact_from_triton(
        compiled,
        grid=_artifact_grid(specialization.row_count),
        parameters=parameters,
        autotuner=autotuner,
    )


def layer_norm(
    builder: ProgramBuilder,
    kernels: KernelCache,
    output: Buffer,
    input_: Buffer,
    bias: Buffer | None,
    gammas: Buffer,
    betas: Buffer,
    specialization: LayerNormSpecialization,
    *,
    skip: Buffer | None = None,
    alpha: Buffer | None = None,
) -> None:
    """Append fused activation, residual addition, and layer normalization."""
    builder.set_target(
        lc0ex_pb2.Target.VENDOR_NVIDIA,
        f"sm_{specialization.architecture}",
    )
    kernel = kernels.get(compile_layer_norm, specialization)
    bias_buffer = bias if (bias is not None and specialization.has_bias) else output
    arguments = [output, input_, bias_buffer]
    if specialization.has_skip:
        if skip is None or alpha is None:
            message = (
                "Skip and alpha buffers are required when "
                "specialization.has_skip is True"
            )
            raise ValueError(message)
        arguments.append(skip)
        arguments.extend((gammas, betas))
        arguments.append(alpha)
    else:
        arguments.extend((gammas, betas))
    readonly = [source for source in arguments[1:] if source is not output]
    builder.call(kernel, *arguments, readonly=readonly)
