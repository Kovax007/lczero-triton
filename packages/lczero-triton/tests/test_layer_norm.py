"""Tests for fused FP16 activation, residual, and layer normalization."""

from collections.abc import Sequence

import pytest
import torch
import triton
from lc0ex import Buffer, ExecutableBuilder, ProgramBuilder
from lc0ex.proto import lc0ex_pb2
from lczero_triton.bt4.kernels._cache import KernelCache
from lczero_triton.bt4.kernels.layer_norm import (
    _ACTIVATIONS,
    _WARP_COUNTS,
    Activation,
    LayerNormSpecialization,
    _artifact_grid,
    _autotune_grid,
    _layer_norm_kernel,
    _layer_norm_skip_kernel,
    compile_layer_norm,
    layer_norm,
)

_CUDA_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)
_FP16_ATOL = 8e-3
_FP16_RTOL = 8e-3
_MISH_BRANCH = -0.6
_NULL_POINTERS = (lc0ex_pb2.PARAMETER_TYPE_NULL_POINTER,) * 2


def _architecture() -> int:
    """Return the active CUDA device's `sm_*` integer suffix."""
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    return major * 10 + minor


def _activate(values: torch.Tensor, activation: Activation) -> torch.Tensor:
    """Apply the LC0 CUDA activation approximation in FP32."""
    if activation == "mish":
        exponential = torch.exp(values)
        numerator = exponential * exponential + 2.0 * exponential
        division = values / (numerator + 2.0)
        return torch.where(
            values <= _MISH_BRANCH,
            numerator * division,
            values - 2.0 * division,
        )
    if activation == "swish":
        return values / (1.0 + torch.exp(-values))
    return values


def _reference(  # noqa: PLR0913, PLR0917
    input_: torch.Tensor,
    bias: torch.Tensor,
    gammas: torch.Tensor,
    betas: torch.Tensor,
    activation: Activation,
    epsilon: float,
    *,
    skip: torch.Tensor | None = None,
    alpha: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate the CUDA layer-normalization operation order in FP32."""
    values = _activate(input_.float() + bias.float(), activation)
    if skip is not None and alpha is not None:
        values = alpha.float() * values + skip.float()
    mean = values.mean(dim=1, keepdim=True)
    variance = ((values - mean) ** 2).mean(dim=1, keepdim=True)
    normalized = (values - mean) / torch.sqrt(variance + epsilon)
    return (normalized * gammas.float() + betas.float()).half()


def _launch(  # noqa: PLR0913, PLR0917
    output: torch.Tensor,
    input_: torch.Tensor,
    bias: torch.Tensor,
    gammas: torch.Tensor,
    betas: torch.Tensor,
    activation: Activation,
    epsilon: float,
    *,
    skip: torch.Tensor | None = None,
    alpha: torch.Tensor | None = None,
    has_bias: bool = True,
) -> None:
    """Launch the physical kernel variant selected by optional operands."""
    row_count, width = input_.shape
    block_size = triton.next_power_of_2(width)
    if skip is None or alpha is None:
        _layer_norm_kernel[_autotune_grid](
            output,
            input_,
            bias,
            gammas,
            betas,
            row_count,
            width,
            epsilon,
            _ACTIVATIONS[activation],
            has_bias,
            block_size,
        )
    else:
        _layer_norm_skip_kernel[_autotune_grid](
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
            _ACTIVATIONS[activation],
            has_bias,
            block_size,
        )


@pytest.mark.gpu
@_CUDA_REQUIRED
@pytest.mark.parametrize(
    ("width", "activation", "variant"),
    [
        (256, "swish", "plain"),
        (1024, "mish", "plain"),
        (1024, "none", "skip"),
        (8192, "swish", "plain"),
    ],
)
def test_required_variants_match_fp32_reference(
    width: int,
    activation: Activation,
    variant: str,
) -> None:
    """Every required width and operation variant matches the fused ordering."""
    torch.manual_seed(width)
    row_count = 3
    has_skip = variant == "skip"
    input_ = (torch.randn((row_count, width), dtype=torch.float32) * 1.7).half()
    bias = torch.linspace(-0.8, 0.9, width, dtype=torch.float32).half()
    gammas = (torch.rand(width, dtype=torch.float32) * 1.5 + 0.25).half()
    betas = torch.linspace(-0.5, 0.4, width, dtype=torch.float32).half()
    skip = (
        (torch.randn((row_count, width), dtype=torch.float32) * 0.6).half()
        if has_skip
        else None
    )
    alpha = torch.tensor([0.37], dtype=torch.float16) if has_skip else None
    expected = _reference(
        input_,
        bias,
        gammas,
        betas,
        activation,
        1e-3,
        skip=skip,
        alpha=alpha,
    )
    result = torch.empty_like(input_, device="cuda")

    _launch(
        result,
        input_.cuda(),
        bias.cuda(),
        gammas.cuda(),
        betas.cuda(),
        activation,
        1e-3,
        skip=None if skip is None else skip.cuda(),
        alpha=None if alpha is None else alpha.cuda(),
    )

    torch.testing.assert_close(
        result.cpu(),
        expected,
        rtol=_FP16_RTOL,
        atol=_FP16_ATOL,
    )


@pytest.mark.gpu
@_CUDA_REQUIRED
def test_bias_activation_alpha_and_skip_ordering() -> None:
    """Bias precedes activation, which precedes alpha scaling and skip addition."""
    width = 256
    base = torch.linspace(-2.0, 2.0, width, dtype=torch.float32)
    input_ = torch.stack((base, base.flip(0))).half()
    bias = torch.linspace(0.7, -0.4, width, dtype=torch.float32).half()
    skip = torch.stack((base.square() * 0.1, -base * 0.3)).half()
    gammas = torch.linspace(0.3, 1.7, width, dtype=torch.float32).half()
    betas = torch.linspace(-0.2, 0.5, width, dtype=torch.float32).half()
    alpha = torch.tensor([0.41], dtype=torch.float16)
    expected = _reference(
        input_,
        bias,
        gammas,
        betas,
        "mish",
        1e-3,
        skip=skip,
        alpha=alpha,
    )
    result = torch.empty_like(input_, device="cuda")

    _launch(
        result,
        input_.cuda(),
        bias.cuda(),
        gammas.cuda(),
        betas.cuda(),
        "mish",
        1e-3,
        skip=skip.cuda(),
        alpha=alpha.cuda(),
    )

    torch.testing.assert_close(result.cpu(), expected, rtol=8e-3, atol=8e-3)


@pytest.mark.gpu
@_CUDA_REQUIRED
def test_population_variance_uses_fp32_at_width_8192() -> None:
    """Large-offset rows neither use sample variance nor overflow FP16 sums."""
    width = 8192
    torch.manual_seed(8192)
    input_ = (8.0 + torch.randn((2, width), dtype=torch.float32) * 0.4).half()
    bias = torch.zeros(width, dtype=torch.float16)
    gammas = torch.ones(width, dtype=torch.float16)
    betas = torch.zeros(width, dtype=torch.float16)
    expected = _reference(input_, bias, gammas, betas, "none", 1e-3)
    result = torch.empty_like(input_, device="cuda")

    _launch(
        result,
        input_.cuda(),
        bias.cuda(),
        gammas.cuda(),
        betas.cuda(),
        "none",
        1e-3,
    )

    assert torch.isfinite(result).all()
    torch.testing.assert_close(result.cpu(), expected, rtol=5e-3, atol=5e-3)


@pytest.mark.gpu
@_CUDA_REQUIRED
@pytest.mark.parametrize("alias", ["input", "skip"])
def test_deepnorm_allows_cuda_safe_in_place_output(alias: str) -> None:
    """A row may overwrite either full-row source after all loads complete."""
    torch.manual_seed(7)
    input_ = torch.randn((2, 256), dtype=torch.float16)
    skip = torch.randn_like(input_)
    bias = torch.randn(256, dtype=torch.float16) * 0.1
    gammas = torch.rand(256, dtype=torch.float16) + 0.5
    betas = torch.randn(256, dtype=torch.float16) * 0.1
    alpha = torch.tensor([0.73], dtype=torch.float16)
    expected = _reference(
        input_,
        bias,
        gammas,
        betas,
        "none",
        1e-3,
        skip=skip,
        alpha=alpha,
    )
    input_cuda = input_.cuda()
    skip_cuda = skip.cuda()

    # Select a tuning result using a separate output before mutating either input.
    _launch(
        torch.empty_like(input_cuda),
        input_cuda,
        bias.cuda(),
        gammas.cuda(),
        betas.cuda(),
        "none",
        1e-3,
        skip=skip_cuda,
        alpha=alpha.cuda(),
    )
    output = input_cuda if alias == "input" else skip_cuda
    _launch(
        output,
        input_cuda,
        bias.cuda(),
        gammas.cuda(),
        betas.cuda(),
        "none",
        1e-3,
        skip=skip_cuda,
        alpha=alpha.cuda(),
    )

    torch.testing.assert_close(output.cpu(), expected, rtol=5e-3, atol=5e-3)


def test_autotune_contract_covers_semantics_and_warp_candidates() -> None:
    """Persistent tuning keys every workload decision but not launch choices."""
    expected_keys = ["row_count", "width", "epsilon", "activation", "has_bias"]
    assert _layer_norm_kernel.keys == expected_keys
    assert _layer_norm_skip_kernel.keys == expected_keys
    assert _layer_norm_kernel.cache_results
    assert _layer_norm_skip_kernel.cache_results
    for kernel in (_layer_norm_kernel, _layer_norm_skip_kernel):
        assert tuple((config.num_warps, config.kwargs['online_algorithm']) for config in kernel.configs) == _WARP_COUNTS
        assert all(len(config.kwargs) and "online_algorithm" in config.kwargs for config in kernel.configs)


@pytest.mark.gpu
@_CUDA_REQUIRED
@pytest.mark.parametrize("variant", ["plain", "skip"])
def test_compilation_captures_variant_abi_and_launch(variant: str) -> None:
    """Each optional-operand variant serializes only its physical pointers."""
    has_skip = variant == "skip"
    specialization = LayerNormSpecialization(
        2,
        256,
        "none",
        has_skip=has_skip,
        architecture=_architecture(),
    )

    artifact = compile_layer_norm(specialization)
    kernel = _layer_norm_skip_kernel if has_skip else _layer_norm_kernel

    assert artifact.binary_format == lc0ex_pb2.Binary.FORMAT_CUBIN
    assert artifact.binary_data
    assert artifact.function
    assert artifact.parameters == (
        *((lc0ex_pb2.PARAMETER_TYPE_POINTER,) * (7 if has_skip else 5)),
        *_NULL_POINTERS,
    )
    assert artifact.grid == _artifact_grid(2)
    assert artifact.block == (kernel.best_config.num_warps * 32, 1, 1)


def _external_buffer(
    program: ProgramBuilder,
    name: str,
    shape: Sequence[int],
    *,
    writable: bool = False,
    persistent: bool = False,
) -> Buffer:
    """Declare one FP16 test buffer with the requested lifetime."""
    if persistent:
        return program.persistent_buffer(
            name=name,
            shape=shape,
            dtype=lc0ex_pb2.Buffer.DATA_TYPE_F16,
            writable=writable,
        )
    return program.buffer(
        name=name,
        shape=shape,
        dtype=lc0ex_pb2.Buffer.DATA_TYPE_F16,
        writable=writable,
    )


@pytest.mark.gpu
@_CUDA_REQUIRED
def test_graph_call_preserves_deepnorm_argument_order() -> None:
    """DeepNorm graph arguments follow the fused CUDA operation's pointer ABI."""
    builder = ExecutableBuilder()
    program = builder.program(name="main")
    output = _external_buffer(program, "output", (2, 256), writable=True)
    input_ = _external_buffer(program, "input", (2, 256))
    skip = _external_buffer(program, "skip", (2, 256))
    bias = _external_buffer(program, "bias", (256,), persistent=True)
    gammas = _external_buffer(program, "gammas", (256,), persistent=True)
    betas = _external_buffer(program, "betas", (256,), persistent=True)
    alpha = _external_buffer(program, "alpha", (1,), persistent=True)

    layer_norm(
        program,
        KernelCache(builder),
        output,
        input_,
        bias,
        gammas,
        betas,
        LayerNormSpecialization(
            2,
            256,
            "none",
            has_skip=True,
            architecture=_architecture(),
        ),
        skip=skip,
        alpha=alpha,
    )

    executable = builder.build()
    node = executable.programs[0].nodes[0]
    locations = {
        buffer.name: (buffer.offset,)
        for buffer in (
            *executable.buffers,
            *executable.programs[0].buffers,
        )
    }
    arguments = [(argument.allocation.offset,) for argument in node.arguments]

    assert executable.target.architecture == f"sm_{_architecture()}"
    assert arguments == [
        locations[name]
        for name in ("output", "input", "bias", "skip", "gammas", "betas", "alpha")
    ]


@pytest.mark.gpu
@_CUDA_REQUIRED
def test_layer_norm_no_bias_graph_call() -> None:
    """When has_bias=False, dummy bias pointer is passed and not loaded."""
    builder = ExecutableBuilder()
    program = builder.program(name="main")
    output = _external_buffer(program, "output", (2, 256), writable=True)
    input_ = _external_buffer(program, "input", (2, 256))
    gammas = _external_buffer(program, "gammas", (256,), persistent=True)
    betas = _external_buffer(program, "betas", (256,), persistent=True)

    layer_norm(
        program,
        KernelCache(builder),
        output,
        input_,
        None,
        gammas,
        betas,
        LayerNormSpecialization(
            2,
            256,
            "none",
            has_skip=False,
            has_bias=False,
            architecture=_architecture(),
        ),
    )

    executable = builder.build()
    node = executable.programs[0].nodes[0]
    locations = {
        buffer.name: (buffer.offset,)
        for buffer in (
            *executable.buffers,
            *executable.programs[0].buffers,
        )
    }
    arguments = [(argument.allocation.offset,) for argument in node.arguments]

    assert executable.target.architecture == f"sm_{_architecture()}"
    assert arguments == [
        locations["output"],
        locations["input"],
        locations["output"],
        locations["gammas"],
        locations["betas"],
    ]
