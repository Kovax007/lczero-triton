"""Child-Q distribution head kernels for the BT4 executable.

The head (`childq_dist_head.py` in the lab tree, served by `export512.py --childq-head`) reads the trunk output beside
the policy head: t = mish(tokens(x)) [64, E], q = Q(t) and k = K(t) [64, d]. Then, per sample, exactly as the exported
ONNX graph computes it:
    logit[i, j, b] = sum_d q[i, d] W[d, b] k[j, d] / sqrt(d)                over the 64 x 64 from/to pairs
    off[p, f, b]   = sum_d k[56 + f, d] P[d, p, b];  off[c] += off[3]       the policy head's promotion rule, per bin
    promotion      = logit[48 + s, 56 + f, b] + off[c, f, b]                 [8 source files, 8 files, 3 pieces]
    rows           = gather(concat(logit.reshape(4096, K), promotion.reshape(192, K)), policy map)   [1858, K]
    rows          += mean_i(t[i]) @ U + u                                    the per-sample position bias
    p = softmax_b(rows);  mean = sum_b p a;  var = max(sum_b (p a) a - mean^2, 0)   on the atom grid a
Kernels:
  * the three GEMMs: either the generic `matmul` kernel (FP16 accumulator, as BT4's own heads) or `childq_linear` here,
    the same tiled GEMM with an FP32 accumulator (FP16 x FP16 products are exact in float32, so only the running sum's
    rounding goes), mish on float32, output float16 or float32. The head's 33-bin softmax amplifies accumulator rounding;
  * `childq_sample_terms`, one program per sample: the position bias [K] and the 24 promotion offset rows, float32;
  * `childq_moments`, one program per (sample, block of policy rows): the bin logits of exactly the gathered pairs, the
    softmax and both moments, float32, written straight into `/output/childq_mean` and `/output/childq_var`.
All arithmetic after the GEMMs is IEEE float32. `tl.dot` on float32 operands defaults to TF32 on NVIDIA targets, the
precision that moved onnx-cuda's child-Q by ~1e-2 (`REPORT_lc0_childq_consumer_impl_0911.md` §4 G3), so it is set
explicitly. Temporaries stay at or below [32, 64] (or [64, 64] single loads) to keep within per-SM shared memory.

Per-sample term layout (float32, 25 K per sample): [0, K) position bias; K + (f * 3 + c) * K + b the promotion offset of
destination file f, piece channel c, bin b.
"""

import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import torch
import triton
import triton.language as tl
from lc0ex import Buffer, KernelArtifact, ProgramBuilder, SymbolHandle
from lc0ex.proto import lc0ex_pb2
from lc0ex.triton_module_compiler import artifact_from_triton

from lczero_triton.bt4.kernels._autotune import cold_do_bench
from lczero_triton.bt4.kernels._cache import KernelCache
from lczero_triton.bt4.kernels.mapping_table import values as mapping_values

_POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER
POLICY_OUTPUTS = 1858
TERM_ROWS = 25
_SQUARE_COUNT = 64

_SAMPLE_TERMS_WARPS = (1, 2, 4, 8)
_MOMENTS_CONFIGURATIONS = ((16, 1), (16, 2), (32, 2), (32, 4))
# (block_m, block_n, block_k, num_warps, num_stages): matmul's tiles whose smem, counted as matmul counts it plus the
# float32 accumulator (2 (m k + k n) + 4 m n bytes per stage), stays under matmul's 160 KB cap.
_LINEAR_TILES = (
    (64, 64, 32, 4, 3),
    (64, 64, 64, 4, 3),
    (128, 64, 32, 4, 3),
    (64, 128, 32, 4, 3),
    (32, 64, 64, 4, 3),
    (32, 32, 64, 2, 3),
)
_ACTIVATION_CODES = {"none": 0, "mish": 1}


def depth_scale(model_width: int) -> float:
    """Return sqrt(d) rounded to float32, the constant the exported graph divides by."""
    return struct.unpack("<f", struct.pack("<f", math.sqrt(model_width)))[0]


def _prune_linear_configs(
    configs: list[triton.Config],
    named_args: Mapping[str, object],
    **kwargs: object,
) -> list[triton.Config]:
    """Drop output tiles wider than a narrow GEMM's output."""
    n = named_args.get("n") or kwargs.get("n")
    if n is None:
        return configs
    kept = [c for c in configs if not (cast("int", n) <= 128 and c.kwargs["block_n"] > 128)]
    return kept or configs[:1]


@triton.autotune(
    configs=[
        triton.Config(
            {"block_m": block_m, "block_n": block_n, "block_k": block_k},
            num_warps=warps,
            num_stages=stages,
        )
        for block_m, block_n, block_k, warps, stages in _LINEAR_TILES
    ],
    key=["m", "n", "k", "activation", "output_f32"],
    prune_configs_by={"early_config_prune": _prune_linear_configs},
    do_bench=cold_do_bench,
    cache_results=True,
)
@triton.jit
def _linear_kernel(
    output,
    activations,
    weights,
    bias,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    activation: tl.constexpr,
    output_f32: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
) -> None:
    """One tile of an FP16-operand GEMM with an FP32 accumulator, bias and optional mish."""
    program_id = tl.program_id(0)
    program_count_n: tl.constexpr = tl.cdiv(n, block_n)
    program_m = program_id // program_count_n
    program_n = program_id % program_count_n
    offsets_m = program_m * block_m + tl.arange(0, block_m)
    offsets_n = program_n * block_n + tl.arange(0, block_n)
    offsets_k = tl.arange(0, block_k)
    valid_m = offsets_m < m
    valid_n = offsets_n < n
    activation_pointers = activations + offsets_m[:, None] * k + offsets_k[None, :]
    weight_pointers = weights + offsets_k[:, None] * n + offsets_n[None, :]
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k_block in range(tl.cdiv(k, block_k)):
        remaining_k = k - k_block * block_k
        activation_values = tl.load(
            activation_pointers,
            mask=valid_m[:, None] & (offsets_k[None, :] < remaining_k),
            other=0.0,
        )
        weight_values = tl.load(
            weight_pointers,
            mask=(offsets_k[:, None] < remaining_k) & valid_n[None, :],
            other=0.0,
        )
        accumulator = tl.dot(activation_values, weight_values, accumulator, out_dtype=tl.float32)
        activation_pointers += block_k
        weight_pointers += block_k * n
    values = accumulator + tl.load(bias + offsets_n, mask=valid_n, other=0.0).to(tl.float32)[None, :]
    if activation == 1:
        # LC0 mish x * tanh(softplus(x)), the matmul kernel's float32 form.
        exponential = tl.exp(values)
        numerator = exponential * exponential + 2.0 * exponential
        division = values / (numerator + 2.0)
        values = tl.where(values <= -0.6, numerator * division, values - 2.0 * division)
    output_pointers = output + offsets_m[:, None] * n + offsets_n[None, :]
    output_mask = valid_m[:, None] & valid_n[None, :]
    if output_f32:
        tl.store(output_pointers, values, mask=output_mask)
    else:
        tl.store(output_pointers, values.to(tl.float16), mask=output_mask)


@triton.autotune(
    configs=[triton.Config({}, num_warps=warps) for warps in _SAMPLE_TERMS_WARPS],
    key=["batch_size", "embedding_width", "model_width", "bin_count"],
    cache_results=True,
)
@triton.jit
def _sample_terms_kernel(
    terms,
    tokens,
    keys,
    position_weights,
    position_bias,
    promotion_weights,
    batch_size: tl.constexpr,
    embedding_width: tl.constexpr,
    model_width: tl.constexpr,
    bin_count: tl.constexpr,
    bin_block: tl.constexpr,
) -> None:
    """Per sample: the position bias and the 24 promotion offset rows."""
    sample = tl.program_id(0)
    base = sample * 25 * bin_count
    bins = tl.arange(0, bin_block)
    bins_valid = bins < bin_count

    # Position bias: mean over the 64 token rows, then @ U + u, one 64-column tile at a time.
    columns_lanes = tl.arange(0, 64)
    position = tl.zeros((bin_block,), dtype=tl.float32)
    for tile in range(tl.cdiv(embedding_width, 64)):
        columns = tile * 64 + columns_lanes
        columns_valid = columns < embedding_width
        pooled = tl.zeros((64,), dtype=tl.float32)
        for square in range(64):
            pooled += tl.load(
                tokens + (sample * 64 + square) * embedding_width + columns,
                mask=columns_valid,
                other=0.0,
            ).to(tl.float32)
        pooled = pooled / 64.0
        weights = tl.load(
            position_weights + columns[:, None] * bin_count + bins[None, :],
            mask=columns_valid[:, None] & bins_valid[None, :],
            other=0.0,
        )
        position += tl.sum(pooled[:, None] * weights, axis=0)
    position += tl.load(position_bias + bins, mask=bins_valid, other=0.0)
    tl.store(terms + base + bins, position, mask=bins_valid)

    # Promotion offsets per destination file: off[p] = k[56 + f] @ P[:, p], then off[c] + off[3] (the knight row).
    depth_lanes = tl.arange(0, 32)
    for file in range(8):
        piece0 = tl.zeros((bin_block,), dtype=tl.float32)
        piece1 = tl.zeros((bin_block,), dtype=tl.float32)
        piece2 = tl.zeros((bin_block,), dtype=tl.float32)
        piece3 = tl.zeros((bin_block,), dtype=tl.float32)
        for tile in range(tl.cdiv(model_width, 32)):
            depth = tile * 32 + depth_lanes
            depth_valid = depth < model_width
            key_values = tl.load(
                keys + (sample * 64 + 56 + file) * model_width + depth,
                mask=depth_valid,
                other=0.0,
            ).to(tl.float32)
            row_base = promotion_weights + depth[:, None] * (4 * bin_count) + bins[None, :]
            weight_mask = depth_valid[:, None] & bins_valid[None, :]
            piece0 += tl.sum(
                key_values[:, None] * tl.load(row_base, mask=weight_mask, other=0.0),
                axis=0,
            )
            piece1 += tl.sum(
                key_values[:, None]
                * tl.load(row_base + bin_count, mask=weight_mask, other=0.0),
                axis=0,
            )
            piece2 += tl.sum(
                key_values[:, None]
                * tl.load(row_base + 2 * bin_count, mask=weight_mask, other=0.0),
                axis=0,
            )
            piece3 += tl.sum(
                key_values[:, None]
                * tl.load(row_base + 3 * bin_count, mask=weight_mask, other=0.0),
                axis=0,
            )
        file_base = terms + base + bin_count + file * 3 * bin_count + bins
        tl.store(file_base, piece0 + piece3, mask=bins_valid)
        tl.store(file_base + bin_count, piece1 + piece3, mask=bins_valid)
        tl.store(file_base + 2 * bin_count, piece2 + piece3, mask=bins_valid)


@triton.autotune(
    configs=[
        triton.Config({"rows_per_program": rows}, num_warps=warps)
        for rows, warps in _MOMENTS_CONFIGURATIONS
    ],
    key=["batch_size", "model_width", "bin_count"],
    cache_results=True,
)
@triton.jit
def _moments_kernel(
    means,
    variances,
    queries,
    keys,
    bin_weights,
    terms,
    atoms,
    mapping,
    batch_size: tl.constexpr,
    model_width: tl.constexpr,
    bin_count: tl.constexpr,
    bin_block: tl.constexpr,
    scale: tl.constexpr,
    rows_per_program: tl.constexpr,
) -> None:
    """Per (sample, policy row): bin logits of the gathered pair, softmax, mean and variance."""
    sample = tl.program_id(0)
    rows = tl.program_id(1) * rows_per_program + tl.arange(0, rows_per_program)
    valid = rows < 1858
    source = tl.load(mapping + rows, mask=valid, other=0)
    is_promotion = source >= 4096
    promotion = tl.where(is_promotion, source - 4096, 0)
    destination_file = (promotion // 3) % 8
    from_square = tl.where(is_promotion, 48 + promotion // 24, source // 64)
    to_square = tl.where(is_promotion, 56 + destination_file, source % 64)
    bins = tl.arange(0, bin_block)
    bins_valid = bins < bin_count

    accumulator = tl.zeros((rows_per_program, bin_block), dtype=tl.float32)
    depth_lanes = tl.arange(0, 32)
    for tile in range(tl.cdiv(model_width, 32)):
        depth = tile * 32 + depth_lanes
        depth_valid = depth < model_width
        row_mask = valid[:, None] & depth_valid[None, :]
        query_values = tl.load(
            queries + (sample * 64 + from_square)[:, None] * model_width + depth[None, :],
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)
        key_values = tl.load(
            keys + (sample * 64 + to_square)[:, None] * model_width + depth[None, :],
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)
        weights = tl.load(
            bin_weights + depth[:, None] * bin_count + bins[None, :],
            mask=depth_valid[:, None] & bins_valid[None, :],
            other=0.0,
        )
        accumulator = tl.dot(
            query_values * key_values, weights, accumulator, input_precision="ieee"
        )
    logits = accumulator / scale

    base = sample * 25 * bin_count
    offsets = tl.load(
        terms
        + base
        + bin_count
        + ((destination_file * 3 + promotion % 3) * bin_count)[:, None]
        + bins[None, :],
        mask=(valid & is_promotion)[:, None] & bins_valid[None, :],
        other=0.0,
    )
    position = tl.load(terms + base + bins, mask=bins_valid, other=0.0)
    logits = (logits + offsets) + position[None, :]
    logits = tl.where(bins_valid[None, :], logits, -float("inf"))
    top = tl.max(logits, axis=1)
    exponents = tl.exp(logits - top[:, None])
    probabilities = exponents / tl.sum(exponents, axis=1)[:, None]
    grid = tl.load(atoms + bins, mask=bins_valid, other=0.0)
    weighted = probabilities * grid[None, :]
    mean = tl.sum(weighted, axis=1)
    variance = tl.maximum(tl.sum(weighted * grid[None, :], axis=1) - mean * mean, 0.0)
    outputs = sample * 1858 + rows
    tl.store(means + outputs, mean, mask=valid)
    tl.store(variances + outputs, variance, mask=valid)


@dataclass(frozen=True, slots=True)
class ChildQLinearSpecialization:
    """Immutable FP32-accumulator GEMM specialization."""

    m: int
    n: int
    k: int
    activation: str
    output_f32: bool
    architecture: int


@dataclass(frozen=True, slots=True)
class ChildQSampleTermsSpecialization:
    """Immutable per-sample child-Q terms specialization."""

    batch_size: int
    embedding_width: int
    model_width: int
    bin_count: int
    architecture: int
    projection_f32: bool = False


@dataclass(frozen=True, slots=True)
class ChildQMomentsSpecialization:
    """Immutable child-Q moments specialization."""

    batch_size: int
    model_width: int
    bin_count: int
    architecture: int
    projection_f32: bool = False
    # The two outputs as float16: half the device-to-host traffic. `tl.store` casts, so the kernel is the same.
    output_f16: bool = False


def _linear_grid(configuration: Mapping[str, object]) -> tuple[int]:
    """Return the one-dimensional tile grid for a tuning candidate."""
    m = cast("int", configuration["m"])
    n = cast("int", configuration["n"])
    block_m = cast("int", configuration["block_m"])
    block_n = cast("int", configuration["block_n"])
    return (((m + block_m - 1) // block_m) * ((n + block_n - 1) // block_n),)


def _moments_grid(configuration: Mapping[str, object]) -> tuple[int, int, int]:
    """Return (samples, blocks of policy rows, 1) for a tuning candidate."""
    rows_per_program = cast("int", configuration["rows_per_program"])
    return (
        cast("int", configuration["batch_size"]),
        (POLICY_OUTPUTS + rows_per_program - 1) // rows_per_program,
        1,
    )


def _projection_dtype(projection_f32: bool) -> torch.dtype:  # noqa: FBT001
    return torch.float32 if projection_f32 else torch.float16


def compile_childq_linear(specialization: ChildQLinearSpecialization) -> KernelArtifact:
    """Autotune and compile one FP32-accumulator child-Q GEMM."""
    s = specialization
    generator = torch.Generator(device="cuda").manual_seed(20260913)
    output = torch.empty(
        (s.m, s.n), dtype=torch.float32 if s.output_f32 else torch.float16, device="cuda"
    )
    activations = (torch.randn((s.m, s.k), generator=generator, device="cuda") * 0.5).half()
    weights = (torch.randn((s.k, s.n), generator=generator, device="cuda") * 0.05).half()
    bias = (torch.randn((s.n,), generator=generator, device="cuda") * 0.05).half()
    compiled = _linear_kernel[_linear_grid](
        output,
        activations,
        weights,
        bias,
        s.m,
        s.n,
        s.k,
        _ACTIVATION_CODES[s.activation],
        s.output_f32,
    )
    selected = _linear_kernel.best_config
    grid = _linear_grid({"m": s.m, "n": s.n, **selected.kwargs})
    return artifact_from_triton(
        compiled,
        grid=(grid[0], 1, 1),
        parameters=(_POINTER,) * 4,
        autotuner=_linear_kernel,
    )


def compile_childq_sample_terms(
    specialization: ChildQSampleTermsSpecialization,
) -> KernelArtifact:
    """Autotune and compile the per-sample child-Q terms kernel."""
    s = specialization
    generator = torch.Generator(device="cuda").manual_seed(20260911)

    def normal(count: int, scale: float, dtype: torch.dtype) -> torch.Tensor:
        values = torch.randn(count, generator=generator, device="cuda") * scale
        return values.to(dtype)

    terms = torch.empty(s.batch_size * TERM_ROWS * s.bin_count, dtype=torch.float32, device="cuda")
    tokens = normal(s.batch_size * _SQUARE_COUNT * s.embedding_width, 0.5, torch.float16)
    keys = normal(s.batch_size * _SQUARE_COUNT * s.model_width, 0.5, _projection_dtype(s.projection_f32))
    position_weights = normal(s.embedding_width * s.bin_count, 0.05, torch.float32)
    position_bias = normal(s.bin_count, 0.02, torch.float32)
    promotion_weights = normal(s.model_width * 4 * s.bin_count, 0.03, torch.float32)
    compiled = _sample_terms_kernel[(s.batch_size, 1, 1)](
        terms,
        tokens,
        keys,
        position_weights,
        position_bias,
        promotion_weights,
        s.batch_size,
        s.embedding_width,
        s.model_width,
        s.bin_count,
        triton.next_power_of_2(s.bin_count),
    )
    return artifact_from_triton(
        compiled,
        grid=(s.batch_size, 1, 1),
        parameters=(_POINTER,) * 6,
        autotuner=_sample_terms_kernel,
    )


def compile_childq_moments(specialization: ChildQMomentsSpecialization) -> KernelArtifact:
    """Autotune and compile the child-Q logits, softmax and moments kernel."""
    s = specialization
    generator = torch.Generator(device="cuda").manual_seed(20260912)

    def normal(count: int, scale: float, dtype: torch.dtype) -> torch.Tensor:
        values = torch.randn(count, generator=generator, device="cuda") * scale
        return values.to(dtype)

    output_dtype = torch.float16 if s.output_f16 else torch.float32
    means = torch.empty(s.batch_size * POLICY_OUTPUTS, dtype=output_dtype, device="cuda")
    variances = torch.empty_like(means)
    projection_dtype = _projection_dtype(s.projection_f32)
    queries = normal(s.batch_size * _SQUARE_COUNT * s.model_width, 0.7, projection_dtype)
    keys = normal(s.batch_size * _SQUARE_COUNT * s.model_width, 0.7, projection_dtype)
    bin_weights = normal(s.model_width * s.bin_count, 0.1, torch.float32)
    terms = normal(s.batch_size * TERM_ROWS * s.bin_count, 0.1, torch.float32)
    atoms = torch.linspace(-1.0, 1.0, s.bin_count, dtype=torch.float32, device="cuda")
    mapping = torch.tensor(mapping_values(), dtype=torch.int32, device="cuda")
    compiled = _moments_kernel[_moments_grid](
        means,
        variances,
        queries,
        keys,
        bin_weights,
        terms,
        atoms,
        mapping,
        s.batch_size,
        s.model_width,
        s.bin_count,
        triton.next_power_of_2(s.bin_count),
        depth_scale(s.model_width),
    )
    selected = _moments_kernel.best_config
    return artifact_from_triton(
        compiled,
        grid=_moments_grid({"batch_size": s.batch_size, **selected.kwargs}),
        parameters=(_POINTER,) * 8,
        autotuner=_moments_kernel,
    )


def childq_linear(  # noqa: PLR0913
    builder: ProgramBuilder,
    kernels: KernelCache,
    output: Buffer,
    activations: Buffer,
    weights: Buffer,
    bias: Buffer,
    specialization: ChildQLinearSpecialization,
) -> None:
    """Append one FP32-accumulator child-Q GEMM with bias."""
    builder.set_target(
        lc0ex_pb2.Target.VENDOR_NVIDIA,
        f"sm_{specialization.architecture}",
    )
    kernel = kernels.get(compile_childq_linear, specialization)
    builder.call(kernel, output, activations, weights, bias, readonly=(activations, weights, bias))


def childq_sample_terms(  # noqa: PLR0913
    builder: ProgramBuilder,
    kernels: KernelCache,
    terms: Buffer,
    tokens: Buffer,
    keys: Buffer,
    position_weights: Buffer,
    position_bias: Buffer,
    promotion_weights: Buffer,
    specialization: ChildQSampleTermsSpecialization,
) -> None:
    """Append the per-sample position bias and promotion offsets."""
    builder.set_target(
        lc0ex_pb2.Target.VENDOR_NVIDIA,
        f"sm_{specialization.architecture}",
    )
    kernel = kernels.get(compile_childq_sample_terms, specialization)
    builder.call(
        kernel,
        terms,
        tokens,
        keys,
        position_weights,
        position_bias,
        promotion_weights,
        readonly=(tokens, keys, position_weights, position_bias, promotion_weights),
    )


def childq_moments(  # noqa: PLR0913
    builder: ProgramBuilder,
    kernels: KernelCache,
    means: Buffer,
    variances: Buffer,
    queries: Buffer,
    keys: Buffer,
    bin_weights: Buffer,
    terms: Buffer,
    atoms: Buffer,
    mapping: SymbolHandle,
    specialization: ChildQMomentsSpecialization,
) -> None:
    """Append the child-Q bin logits, softmax and moments into the two outputs."""
    builder.set_target(
        lc0ex_pb2.Target.VENDOR_NVIDIA,
        f"sm_{specialization.architecture}",
    )
    kernel = kernels.get(compile_childq_moments, specialization)
    builder.call(
        kernel,
        means,
        variances,
        queries,
        keys,
        bin_weights,
        terms,
        atoms,
        mapping,
        readonly=(queries, keys, bin_weights, terms, atoms),
    )
