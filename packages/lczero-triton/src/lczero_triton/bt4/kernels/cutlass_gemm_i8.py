"""Q1: the integer GEMM of SPEC v2 -- int8 x int8 -> int32 on the tensor cores, delivered as a CUTLASS CUBIN.

Every quantised site computes

    y[m, n] = acc[m, n] * scale[n] + bias[n]      (+ skip[m, n] at the two residual sites)
    acc     = sum_j q[m, j] * Wq[n, j]            int32, `IMMA.16832.S8.S8.S32` on sm_89

`q` are the int8 codes the producer wrote through the analyser's per-channel vector (`r_j = 1 / (s_j D)`, in the
norm for `attn_in` / `ffn_in`, in this GEMM's GLU epilogue for `ffn_mid`, in a conversion pass for `attn_out`);
`Wq` are the int8 weights of `W'[:, j] = W[:, j] * s_j`, quantised per output column with step `w[n]`, stored
`[n, k]` (K-contiguous: the int8 tensor-op path reads B column-major, and `ldmatrix` cannot transpose 8-bit data).
`scale[n] = D * w[n]` -- times the block's residual alpha on the residual sites, whose bias is folded the same
way -- is built at carrier time (`lab/_quant_gemm.py`); the kernel knows nothing of D, s or alpha. Both vectors are
FP32: `D * w[n]` sits near 1e-5..1e-3, where FP16 is already subnormal.

The accumulator is converted to FP32 once, in registers, inside the epilogue: nothing writes int32 anywhere
(design 09-23 §6). The epilogue is CUTLASS's `EpilogueWithVisitor` -- the accumulators are staged through shared
memory by CUTLASS exactly as for an ordinary epilogue, and the visitor sees each thread's 8 consecutive columns
with their coordinates, which is what a per-column scale and bias need.

Three epilogues:
* `bias`     -- QKV: `acc * scale + bias` -> FP16.
* `residual` -- attention out-projection, FFN2: `acc * scale + bias + skip` -> FP16 (alpha folded in the vectors).
* `glu`      -- FFN1: the weight's columns are interleaved gate/up PAIRS (`[g0 u0 g1 u1 ...]`, the carrier's
  `interleave_pairs`), so each visited 8-column fragment holds four complete pairs; the sigmoid GLU with the
  lab's `ffn_softcap` is applied in the epilogue and the hidden is written as int8 through the `ffn_mid` vector
  (`output="i8"`) -- the "for free" epilogue of the 09-22 report -- or as FP16 (`output="f16"`, a test route).
  Round 25's G2 (interleaved columns) is therefore this family, not a separate kernel.
* `mish`     -- BT4's FFN1 (B3, 09-28): `mish(acc * scale + bias)`, LC0's Mish transcribed exactly as the FP16 CUTLASS
  epilogue has it (`cutlass_matmul`'s `lc0act::mish`: the `__expf` / `__fdividef` pair), written as FP16 (a layer whose
  `ffn_mid` stays FP16) or as int8 codes through the `ffn_mid` vector (`output="i8"`), eight codes in one 64-bit store.

Rounding of every int8 this module writes: `floor(x * r + 0.5)` clamped to +-127 -- the same rule the norm's
conversion uses (`layer_norm`, `quantise_operand`), so one artifact has one rounding rule. It differs from
round-half-to-even only on exact ties.
"""

import logging
import math
import os
import subprocess
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from lc0ex import Buffer, KernelArtifact, ProgramBuilder
from lc0ex.cubin_module_compiler import artifact_from_cubin, compile_cuda
from lc0ex.proto import lc0ex_pb2

from lczero_triton.bt4.kernels._activation import SOFTCAP_UNIT_SERIES_MIN
from lczero_triton.bt4.kernels._cache import KernelCache
from lczero_triton.bt4.kernels.cutlass_matmul import (
    _NVCC,
    CUTLASS_INCLUDE,
    _store_sweep_result,
    device_key,
    dynamic_shared_limit,
)

_LOGGER = logging.getLogger(__name__)
_POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER
_WARP_SIZE = 32
EPILOGUES = ("bias", "residual", "glu", "mish")
OUTPUTS = ("f16", "i8")
# int8 tensor-op GEMMs need 16-element (128-bit) rows on both operands.
I8_ALIGNMENT = 16

Tile = tuple[tuple[int, int, int], tuple[int, int, int], int]

# Candidates on an unswept device: the tiles that won at least one (shape, M) of round 26's two RTX 4090 sweeps
# (`briefs_2026-09-23/REPORT_backend_round26_q1_int8_gemm_served_0923.md` §3). int8 needs a K-step of 64 or more (the
# 8-bit crosswise shared-memory layout: every K-32 tile failed to compile); shared memory per stage is
# (tile_m + tile_n) * tile_k bytes, and at 48 KB two CTAs share an SM -- which is what the winners have in common.
_SWEEP_CANDIDATES: tuple[Tile, ...] = (
    ((64, 64, 64), (32, 32, 64), 4),
    ((64, 64, 64), (32, 32, 64), 6),
    ((64, 128, 64), (32, 64, 64), 3),
    ((128, 64, 64), (64, 32, 64), 3),
    ((128, 128, 64), (64, 64, 64), 3),
    ((128, 128, 64), (64, 64, 64), 4),
    ((256, 128, 64), (64, 64, 64), 2),
    ((256, 128, 64), (64, 64, 64), 3),
    ((64, 64, 128), (32, 32, 128), 3),
)


@dataclass(frozen=True, slots=True)
class CutlassGemmI8Specialization:
    """One int8 GEMM site: `m` rows, `n` OUTPUT columns, `k` reduction; the GLU computes `2n` accumulator columns."""

    m: int
    n: int
    k: int
    architecture: int
    epilogue: str = "bias"
    output: str = "f16"
    # BT6-test's `ffn_softcap` on the GLU's two branches (0 = off), as `cutlass_matmul`'s `glu_softcap`.
    glu_softcap: float = 0.0

    def __post_init__(self) -> None:
        """Refuse a combination the rendered kernel does not implement."""
        if self.epilogue not in EPILOGUES:
            message = f"CutlassGemmI8Specialization.epilogue={self.epilogue!r}; expected one of {EPILOGUES}"
            raise ValueError(message)
        if self.output not in OUTPUTS:
            message = f"CutlassGemmI8Specialization.output={self.output!r}; expected one of {OUTPUTS}"
            raise ValueError(message)
        if self.output == "i8" and self.epilogue not in ("glu", "mish"):
            message = "an int8 output is served by the glu and mish epilogues only (the FFN hidden feeding ffn_mid)"
            raise ValueError(message)
        if self.glu_softcap and self.epilogue != "glu":
            message = "glu_softcap needs epilogue='glu'"
            raise ValueError(message)
        if self.glu_softcap < 0.0 or self.glu_softcap != self.glu_softcap:
            message = f"glu_softcap must be >= 0 (0 = off); got {self.glu_softcap}"
            raise ValueError(message)
        if not i8_supports(self.gemm_n, self.k):
            message = (f"int8 GEMM widths n={self.gemm_n}, k={self.k} must be multiples of {I8_ALIGNMENT} "
                       "(128-bit rows on both int8 operands)")
            raise ValueError(message)

    @property
    def gemm_n(self) -> int:
        """Accumulator columns: the GLU computes gate and up side by side."""
        return 2 * self.n if self.epilogue == "glu" else self.n

    @property
    def family(self) -> str:
        """The tile-cache family: the epilogue changes the register budget, so it is part of the key."""
        return f"i8_{self.epilogue}_{self.output}"


def i8_supports(n: int, k: int) -> bool:
    """Whether both widths meet the int8 tensor-op alignment."""
    return n % I8_ALIGNMENT == 0 and k % I8_ALIGNMENT == 0


def entry_point_name(specialization: CutlassGemmI8Specialization) -> str:
    """The `extern "C"` symbol of one specialization."""
    suffix = f"_{specialization.epilogue}"
    if specialization.glu_softcap > 0.0:
        suffix += "_cap" + f"{specialization.glu_softcap:g}".replace(".", "p").replace("-", "m").replace("+", "")
    if specialization.output != "f16":
        suffix += f"_{specialization.output}out"
    return f"lc0ex_cutlass_gemm_i8_m{specialization.m}_n{specialization.n}_k{specialization.k}{suffix}"


def _thread_count(threadblock: tuple[int, int, int], warp: tuple[int, int, int]) -> int:
    warps = (threadblock[0] // warp[0]) * (threadblock[1] // warp[1]) * max(1, threadblock[2] // warp[2])
    return warps * _WARP_SIZE


_TYPES_TEMPLATE = """\
#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/threadblock/default_mma.h>
#include <cutlass/epilogue/threadblock/default_epilogue_tensor_op.h>
#include <cutlass/epilogue/threadblock/epilogue_with_visitor.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cstdint>

#if defined(__CUDA_ARCH__)
#define LC0_EXP(x) __expf(x)
#define LC0_DIV(a, b) __fdividef((a), (b))
#else
#define LC0_EXP(x) expf(x)
#define LC0_DIV(a, b) ((a) / (b))
#endif

namespace {{

// LC0's Mish, as `cutlass_matmul`'s `lc0act::mish` (the FP16 FFN1 epilogue this int8 site replaces).
CUTLASS_HOST_DEVICE float lc0_mish(float x) {{
  const float e = LC0_EXP(x);
  const float n = e * e + 2.0f * e;
  const float d = LC0_DIV(x, n + 2.0f);
  return x <= -0.6f ? n * d : x - 2.0f * d;
}}

using ElementA = int8_t;
using ElementB = int8_t;
using ElementAccumulator = int32_t;
// The epilogue's tile iterator is instantiated for an FP16 [M, gemm_n] output: its thread map is what hands each
// thread 8 consecutive accumulator columns (one 128-bit FP16 access). The visitor does its own stores.
using ElementTile = cutlass::half_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

constexpr int kM = {m};
constexpr int kN = {gemm_n};
constexpr int kK = {k};
constexpr int kOutputColumns = {n};

using ThreadblockShape = cutlass::gemm::GemmShape<{tile_m}, {tile_n}, {tile_k}>;
using WarpShape = cutlass::gemm::GemmShape<{warp_m}, {warp_n}, {warp_k}>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

using Mma = typename cutlass::gemm::threadblock::DefaultMma<
    ElementA, LayoutA, 16, ElementB, LayoutB, 16,
    ElementAccumulator, LayoutC,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    ThreadblockShape, WarpShape, InstructionShape, {stages},
    cutlass::arch::OpMultiplyAddSaturate>::ThreadblockMma;

// Only its types are used: the visitor replaces the output functor.
using PlaceholderOp = cutlass::epilogue::thread::LinearCombination<
    ElementTile, 128 / cutlass::sizeof_bits<ElementTile>::value, ElementAccumulator, float>;
using DefaultEpilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
    ThreadblockShape, typename Mma::Operator, ThreadblockShape::kK / WarpShape::kK,
    PlaceholderOp, PlaceholderOp::kCount>::Epilogue;
using OutputTileIterator = typename DefaultEpilogue::OutputTileIterator;

{glu_softcap_decl}
struct Lc0I8Visitor {{
  static int const kIterations = OutputTileIterator::kIterations;
  static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;
  static_assert(kElementsPerAccess == 8, "the visitor is written for 8 accumulator columns per access");
  using ElementAccumulator = int32_t;
  using AccumulatorFragment = cutlass::Array<ElementAccumulator, kElementsPerAccess>;
  struct SharedStorage {{}};

  // A thread's columns do not change from step to step (only its rows advance), so each column group's scale and
  // bias are loaded once, in begin_epilogue, instead of once per visited fragment.
  static int const kColumnIterations = OutputTileIterator::ThreadMap::Iterations::kColumn;

  OutputTileIterator coordinates_;
  {output_type}* output_;
  const float* scale_;
  const float* bias_;
  const cutlass::half_t* skip_;
  const float* prescale_;
  float scale_cache_[kColumnIterations][8];
  float bias_cache_[kColumnIterations][8];

  CUTLASS_DEVICE Lc0I8Visitor(int thread_index, cutlass::MatrixCoord const& threadblock_offset,
                              {output_type}* output, const float* scale, const float* bias,
                              const cutlass::half_t* skip, const float* prescale)
      : coordinates_(typename OutputTileIterator::Params(LayoutC(kN)),
                     reinterpret_cast<ElementTile*>(output), {{kM, kN}}, thread_index, threadblock_offset),
        output_(output), scale_(scale), bias_(bias), skip_(skip), prescale_(prescale) {{}}

  CUTLASS_DEVICE void set_k_partition(int, int) {{}}
  CUTLASS_DEVICE void set_batch_index(int) {{}}
  CUTLASS_DEVICE void begin_epilogue() {{
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < kColumnIterations; ++c) {{
      const int column = coordinates_.thread_start().column() + OutputTileIterator::ThreadMap::iteration_offset(c).column();
      if (column < kN) {{
        const float4 s0 = *reinterpret_cast<const float4*>(scale_ + column);
        const float4 s1 = *reinterpret_cast<const float4*>(scale_ + column + 4);
        const float4 b0 = *reinterpret_cast<const float4*>(bias_ + column);
        const float4 b1 = *reinterpret_cast<const float4*>(bias_ + column + 4);
        scale_cache_[c][0] = s0.x; scale_cache_[c][1] = s0.y; scale_cache_[c][2] = s0.z; scale_cache_[c][3] = s0.w;
        scale_cache_[c][4] = s1.x; scale_cache_[c][5] = s1.y; scale_cache_[c][6] = s1.z; scale_cache_[c][7] = s1.w;
        bias_cache_[c][0] = b0.x; bias_cache_[c][1] = b0.y; bias_cache_[c][2] = b0.z; bias_cache_[c][3] = b0.w;
        bias_cache_[c][4] = b1.x; bias_cache_[c][5] = b1.y; bias_cache_[c][6] = b1.z; bias_cache_[c][7] = b1.w;
      }}
    }}
  }}
  CUTLASS_DEVICE void begin_step(int) {{}}
  CUTLASS_DEVICE void begin_row(int) {{}}
  CUTLASS_DEVICE void end_row(int) {{}}
  CUTLASS_DEVICE void end_step(int) {{ ++coordinates_; }}
  CUTLASS_DEVICE void end_epilogue() {{}}

  CUTLASS_DEVICE void visit(int, int, int, int fragment_index, AccumulatorFragment const& accumulators) {{
    const cutlass::MatrixCoord coordinate =
        coordinates_.thread_start() + OutputTileIterator::ThreadMap::iteration_offset(fragment_index);
    const int row = coordinate.row();
    const int column = coordinate.column();
    if (row >= kM || column >= kN) return;
    const int group = fragment_index % kColumnIterations;
    float value[8];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 8; ++i) {{
      value[i] = fmaf(static_cast<float>(accumulators[i]), scale_cache_[group][i], bias_cache_[group][i]);
    }}
{visit_body}
  }}
}};

using Epilogue = typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<
    Lc0I8Visitor, DefaultEpilogue>::Epilogue;

union SharedStorage {{
  typename Mma::SharedStorage main_loop;
  typename Epilogue::SharedStorage epilogue;
}};

}}  // namespace
"""

# FP16 [M, n] out: one 128-bit store per fragment, the same access a plain CUTLASS epilogue makes.
_VISIT_BIAS = """    cutlass::Array<cutlass::half_t, 8> packed;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 8; ++i) packed[i] = cutlass::half_t(value[i]);
    *reinterpret_cast<uint4*>(output_ + static_cast<int64_t>(row) * kOutputColumns + column) =
        *reinterpret_cast<uint4 const*>(&packed);"""

# The skip is read at the destination's own coordinate: 8 FP16 in one 128-bit load. Order as the FP16 residual
# epilogue: the projection (with its alpha-folded scale and bias) first, the skip last, all in FP32.
_VISIT_RESIDUAL = """    const uint4 raw_skip =
        *reinterpret_cast<uint4 const*>(skip_ + static_cast<int64_t>(row) * kOutputColumns + column);
    cutlass::Array<cutlass::half_t, 8> const& skip = *reinterpret_cast<cutlass::Array<cutlass::half_t, 8> const*>(&raw_skip);
    cutlass::Array<cutlass::half_t, 8> packed;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 8; ++i) packed[i] = cutlass::half_t(value[i] + static_cast<float>(skip[i]));
    *reinterpret_cast<uint4*>(output_ + static_cast<int64_t>(row) * kOutputColumns + column) =
        *reinterpret_cast<uint4 const*>(&packed);"""

# Interleaved pairs: value[2p] is the gate of hidden column column/2 + p, value[2p + 1] its up.
_VISIT_GLU_HEAD = """    const int hidden = column / 2;
    float h[4];
    CUTLASS_PRAGMA_UNROLL
    for (int p = 0; p < 4; ++p) {
      const float gate = LC0_DIV(1.0f, 1.0f + LC0_EXP(-value[2 * p]));
      h[p] = {gate_expr} * {up_expr};
    }"""

_VISIT_GLU_F16 = """    cutlass::Array<cutlass::half_t, 4> packed;
    CUTLASS_PRAGMA_UNROLL
    for (int p = 0; p < 4; ++p) packed[p] = cutlass::half_t(h[p]);
    *reinterpret_cast<uint2*>(output_ + static_cast<int64_t>(row) * kOutputColumns + hidden) =
        *reinterpret_cast<uint2 const*>(&packed);"""

# The ffn_mid codes: floor(h * r + 0.5) clamped to +-127, the norm's rule; four codes in one 32-bit store.
_VISIT_GLU_I8 = """    const float4 r = *reinterpret_cast<const float4*>(prescale_ + hidden);
    const float rs[4] = {r.x, r.y, r.z, r.w};
    uint32_t word = 0;
    CUTLASS_PRAGMA_UNROLL
    for (int p = 0; p < 4; ++p) {
      const float code = fminf(fmaxf(floorf(fmaf(h[p], rs[p], 0.5f)), -127.0f), 127.0f);
      word |= (static_cast<uint32_t>(static_cast<int32_t>(code)) & 0xffu) << (8 * p);
    }
    *reinterpret_cast<uint32_t*>(output_ + static_cast<int64_t>(row) * kOutputColumns + hidden) = word;"""

# BT4's FFN1 (B3): Mish on the eight columns, then FP16 out (one 128-bit store) or the ffn_mid codes (one 64-bit store,
# the norm's rounding rule).
_VISIT_MISH_F16 = """    cutlass::Array<cutlass::half_t, 8> packed;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 8; ++i) packed[i] = cutlass::half_t(lc0_mish(value[i]));
    *reinterpret_cast<uint4*>(output_ + static_cast<int64_t>(row) * kOutputColumns + column) =
        *reinterpret_cast<uint4 const*>(&packed);"""

_VISIT_MISH_I8 = """    const float4 r0 = *reinterpret_cast<const float4*>(prescale_ + column);
    const float4 r1 = *reinterpret_cast<const float4*>(prescale_ + column + 4);
    const float rs[8] = {r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w};
    uint32_t words[2] = {0u, 0u};
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 8; ++i) {
      const float code = fminf(fmaxf(floorf(fmaf(lc0_mish(value[i]), rs[i], 0.5f)), -127.0f), 127.0f);
      words[i / 4] |= (static_cast<uint32_t>(static_cast<int32_t>(code)) & 0xffu) << (8 * (i % 4));
    }
    *reinterpret_cast<uint2*>(output_ + static_cast<int64_t>(row) * kOutputColumns + column) =
        make_uint2(words[0], words[1]);"""

# `cutlass_matmul`'s GLU softcap helpers, verbatim in form: c * tanh(x / c) two-sided, and the series for a
# sigmoid gate in (0, 1) at c >= 8 -- the same fidelity class as the FP16 route this replaces.
_GLU_SOFTCAP_DECL = """constexpr float kGluSoftcap = {cap!r}f;
CUTLASS_HOST_DEVICE float lc0_glu_softcap(float x) {{
  const float magnitude = LC0_DIV(x < 0.0f ? -x : x, kGluSoftcap);
  const float decay = LC0_EXP(-2.0f * magnitude);
  const float tangent = LC0_DIV(1.0f - decay, 1.0f + decay);
  return kGluSoftcap * (x < 0.0f ? -tangent : tangent);
}}
CUTLASS_HOST_DEVICE float lc0_glu_softcap_unit(float g) {{
{unit_body}
}}
"""
_UNIT_SERIES = """  const float ratio = g * (1.0f / kGluSoftcap);
  const float square = ratio * ratio;
  return g * (1.0f - square * ((1.0f / 3.0f) - square * (2.0f / 15.0f)));"""
_UNIT_GENERAL = "  return lc0_glu_softcap(g);"

_KERNEL_TEMPLATE = _TYPES_TEMPLATE + """
// Destination first, then the readonly operands: the lc0ex graph ABI.
extern "C" __global__ __launch_bounds__({threads}) void {entry}(
    {output_type}* d, const ElementA* a, const ElementB* b, const float* scale, const float* bias{extra_parameters}) {{
  extern __shared__ char lc0ex_shared_base[];
  SharedStorage& shared = *reinterpret_cast<SharedStorage*>(lc0ex_shared_base);

  const int tile_row = blockIdx.x;
  const int tile_column = blockIdx.y;
  if (tile_row * ThreadblockShape::kM >= kM) return;
  if (tile_column * ThreadblockShape::kN >= kN) return;

  const cutlass::MatrixCoord offset_a{{tile_row * ThreadblockShape::kM, 0}};
  const cutlass::MatrixCoord offset_b{{0, tile_column * ThreadblockShape::kN}};
  const int k_iterations = (kK + ThreadblockShape::kK - 1) / ThreadblockShape::kK;

  // A is the [M, K] codes (row-major); B is [N, K] row-major = [K, N] column-major, leading dimension K.
  typename Mma::IteratorA iterator_a(
      typename Mma::IteratorA::Params(LayoutA(kK)), const_cast<ElementA*>(a), {{kM, kK}}, threadIdx.x, offset_a);
  typename Mma::IteratorB iterator_b(
      typename Mma::IteratorB::Params(LayoutB(kK)), const_cast<ElementB*>(b), {{kK, kN}}, threadIdx.x, offset_b);

  const int warp_index = threadIdx.x / 32;
  const int lane_index = threadIdx.x % 32;

  typename Mma::FragmentC accumulators;
  accumulators.clear();
  Mma mma(shared.main_loop, threadIdx.x, warp_index, lane_index);
  mma(k_iterations, accumulators, iterator_a, iterator_b, accumulators);

  const cutlass::MatrixCoord offset_d{{tile_row * ThreadblockShape::kM, tile_column * ThreadblockShape::kN}};
  Lc0I8Visitor visitor(threadIdx.x, offset_d, d, scale, bias, {skip_argument}, {prescale_argument});
  Epilogue epilogue(shared.epilogue, threadIdx.x, warp_index, lane_index);
  epilogue(visitor, accumulators);
}}
"""

_PROBE_TEMPLATE = _TYPES_TEMPLATE + """
#include <cstdio>
int main() {{ std::printf("%zu\\n", sizeof(SharedStorage)); return 0; }}
"""

# Timing only (operands zeroed; a GEMM's runtime does not depend on its values).
_SWEEP_MAIN = r"""
#include <cuda_runtime.h>
#include <cstdio>

int main() {{
  void *d = nullptr, *a = nullptr, *b = nullptr, *scale = nullptr, *bias = nullptr, *skip = nullptr, *prescale = nullptr;
  if (cudaMalloc(&d, (size_t)kM * kN * 2) != cudaSuccess || cudaMalloc(&a, (size_t)kM * kK) != cudaSuccess ||
      cudaMalloc(&b, (size_t)kN * kK) != cudaSuccess || cudaMalloc(&scale, (size_t)kN * 4) != cudaSuccess ||
      cudaMalloc(&bias, (size_t)kN * 4) != cudaSuccess || cudaMalloc(&skip, (size_t)kM * kN * 2) != cudaSuccess ||
      cudaMalloc(&prescale, (size_t)kN * 4) != cudaSuccess) {{
    std::printf("-1\n");
    return 0;
  }}
  cudaMemset(a, 0, (size_t)kM * kK);
  cudaMemset(b, 0, (size_t)kN * kK);
  cudaMemset(scale, 0, (size_t)kN * 4);
  cudaMemset(bias, 0, (size_t)kN * 4);
  cudaMemset(skip, 0, (size_t)kM * kN * 2);
  cudaMemset(prescale, 0, (size_t)kN * 4);
  const int shared_bytes = (int)sizeof(SharedStorage);
  if (cudaFuncSetAttribute((const void*)&{entry}, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes) !=
      cudaSuccess) {{
    std::printf("-1\n");
    return 0;
  }}
  const dim3 grid((kM + ThreadblockShape::kM - 1) / ThreadblockShape::kM, (kN + ThreadblockShape::kN - 1) / ThreadblockShape::kN, 1);
  const dim3 block({threads}, 1, 1);
  for (int i = 0; i < 20; ++i) {entry}<<<grid, block, shared_bytes>>>({sweep_arguments});
  if (cudaDeviceSynchronize() != cudaSuccess) {{
    std::printf("-1\n");
    return 0;
  }}
  // Timed as lc0ex serves: 200 launches captured in ONE CUDA graph, the best of 5 replays. A plain-stream loop rounds
  // every kernel period up to a multiple of 2.048 us on some sm_120 drivers (RTX 5090, 590.48), which turned this
  // sweep's picks into coin flips between tied candidates (backend sm_120 report 09-28 §3).
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  for (int i = 0; i < 200; ++i) {entry}<<<grid, block, shared_bytes, stream>>>({sweep_arguments});
  cudaStreamEndCapture(stream, &graph);
  if (cudaGraphInstantiate(&exec, graph, 0) != cudaSuccess) {{
    std::printf("-1\n");
    return 0;
  }}
  cudaGraphLaunch(exec, stream);
  cudaStreamSynchronize(stream);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float best = 1e30f;
  for (int replay = 0; replay < 5; ++replay) {{
    cudaEventRecord(start, stream);
    cudaGraphLaunch(exec, stream);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float milliseconds = 0.f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    best = milliseconds < best ? milliseconds : best;
  }}
  if (cudaGetLastError() != cudaSuccess) {{
    std::printf("-1\n");
    return 0;
  }}
  std::printf("%f\n", best * 1000.f / 200.f);
  return 0;
}}
"""


def _glu_fields(specialization: CutlassGemmI8Specialization) -> dict[str, str]:
    cap = float(specialization.glu_softcap)
    if specialization.epilogue != "glu":
        return {"glu_softcap_decl": "", "gate_expr": "gate", "up_expr": "value[2 * p + 1]"}
    if cap == 0.0:
        return {"glu_softcap_decl": "", "gate_expr": "gate", "up_expr": "value[2 * p + 1]"}
    unit = _UNIT_SERIES if cap >= SOFTCAP_UNIT_SERIES_MIN else _UNIT_GENERAL
    return {
        "glu_softcap_decl": _GLU_SOFTCAP_DECL.format(cap=cap, unit_body=unit),
        "gate_expr": "lc0_glu_softcap_unit(gate)",
        "up_expr": "lc0_glu_softcap(value[2 * p + 1])",
    }


def _visit_body(specialization: CutlassGemmI8Specialization) -> str:
    if specialization.epilogue == "bias":
        return _VISIT_BIAS
    if specialization.epilogue == "residual":
        return _VISIT_RESIDUAL
    if specialization.epilogue == "mish":
        return _VISIT_MISH_I8 if specialization.output == "i8" else _VISIT_MISH_F16
    fields = _glu_fields(specialization)
    head = _VISIT_GLU_HEAD.replace("{gate_expr}", fields["gate_expr"]).replace("{up_expr}", fields["up_expr"])
    return head + "\n" + (_VISIT_GLU_I8 if specialization.output == "i8" else _VISIT_GLU_F16)


def _parameter_fields(specialization: CutlassGemmI8Specialization) -> dict[str, str]:
    extra, skip, prescale = "", "nullptr", "nullptr"
    sweep = ["static_cast<{t}*>(d)", "static_cast<const int8_t*>(a)", "static_cast<const int8_t*>(b)",
             "static_cast<const float*>(scale)", "static_cast<const float*>(bias)"]
    if specialization.epilogue == "residual":
        extra, skip = ", const cutlass::half_t* skip", "skip"
        sweep.append("static_cast<const cutlass::half_t*>(skip)")
    if specialization.output == "i8":
        extra, prescale = ", const float* prescale", "prescale"
        sweep.append("static_cast<const float*>(prescale)")
    output_type = "int8_t" if specialization.output == "i8" else "cutlass::half_t"
    return {
        "extra_parameters": extra,
        "skip_argument": skip,
        "prescale_argument": prescale,
        "output_type": output_type,
        "sweep_arguments": ", ".join(sweep).replace("{t}", output_type),
    }


def _render(template: str, specialization: CutlassGemmI8Specialization, tile: Tile | None = None) -> str:
    threadblock, warp, stages = tile or select_tile(specialization)
    fields = _parameter_fields(specialization)
    return template.format(
        m=specialization.m, n=specialization.n, gemm_n=specialization.gemm_n, k=specialization.k,
        tile_m=threadblock[0], tile_n=threadblock[1], tile_k=threadblock[2],
        warp_m=warp[0], warp_n=warp[1], warp_k=warp[2], stages=stages,
        threads=_thread_count(threadblock, warp), entry=entry_point_name(specialization),
        glu_softcap_decl=_glu_fields(specialization)["glu_softcap_decl"],
        visit_body=_visit_body(specialization),
        **fields,
    )


def render_source(specialization: CutlassGemmI8Specialization, tile: Tile | None = None) -> str:
    """The kernel translation unit of one specialization."""
    return _render(_KERNEL_TEMPLATE, specialization, tile)


def _tile_override() -> Tile | None:
    raw = os.environ.get("LC0EX_CUTLASS_I8_TILE", "")
    if not raw:
        return None
    values = [int(field) for field in raw.replace("x", ",").replace(" ", ",").split(",") if field]
    expected = 7
    if len(values) != expected:
        message = "LC0EX_CUTLASS_I8_TILE wants tile_m,tile_n,tile_k,warp_m,warp_n,warp_k,stages"
        raise ValueError(message)
    return (values[0], values[1], values[2]), (values[3], values[4], values[5]), values[6]


def _cache_key(specialization: CutlassGemmI8Specialization, architecture: int, multiprocessors: int) -> str:
    return "/".join(str(part) for part in (architecture, multiprocessors, specialization.m, specialization.gemm_n,
                                          specialization.k, specialization.family))


_MEMO: dict[tuple[object, ...], Tile] = {}


def select_tile(specialization: CutlassGemmI8Specialization) -> Tile:
    """The tile this process uses for one specialization: override, then cache, then a sweep on this device."""
    override = _tile_override()
    if override is not None:
        return override
    memo = _MEMO.get((specialization.m, specialization.gemm_n, specialization.k, specialization.family))
    if memo is not None:
        return memo
    architecture, multiprocessors = device_key()
    key = _cache_key(specialization, architecture, multiprocessors)
    from lczero_triton.bt4.kernels.cutlass_matmul import _load_sweep_cache  # noqa: PLC0415

    cached = _load_sweep_cache().get(key)
    if cached is not None:
        threadblock, warp, stages = cached
        tile: Tile = (tuple(threadblock), tuple(warp), stages)  # type: ignore[assignment]
    elif os.environ.get("LC0EX_CUTLASS_TILE_REUSE") == "1" and (nearest := _nearest_cached(specialization)) is not None:
        # B3: as the FP16 path does (`cutlass_matmul._nearest_cached_tile`), an unswept rung borrows the nearest swept
        # rung's tile (same device, gemm_n, k and epilogue family; nearest in log-M) and says so -- it is not a
        # measurement of this shape. A dense ladder sweeps its key rungs first, then reuses.
        tile, donor_m = nearest
        _LOGGER.info("reusing the int8 tile measured at m=%d for m=%d n=%d k=%d (%s): %s", donor_m,
                     specialization.m, specialization.gemm_n, specialization.k, specialization.family, tile)
    else:
        tile = _sweep(specialization, key)
    _MEMO[(specialization.m, specialization.gemm_n, specialization.k, specialization.family)] = tile
    return tile


def _nearest_cached(specialization: CutlassGemmI8Specialization) -> tuple[Tile, int] | None:
    """The nearest swept rung's int8 tile for the same device, `gemm_n`, `k` and family, and its M (log-M distance)."""
    from lczero_triton.bt4.kernels.cutlass_matmul import _load_sweep_cache  # noqa: PLC0415

    architecture, multiprocessors = device_key()
    prefix = f"{architecture}/{multiprocessors}/"
    suffix = f"/{specialization.gemm_n}/{specialization.k}/{specialization.family}"
    best: tuple[float, int, list[object]] | None = None
    for key, tile in _load_sweep_cache().items():
        if not key.startswith(prefix) or not key.endswith(suffix):
            continue
        try:
            donor_m = int(key[len(prefix):].split("/", 1)[0])
        except ValueError:
            continue
        if donor_m <= 0:
            continue
        distance = abs(math.log(donor_m / specialization.m))
        if best is None or distance < best[0]:
            best = (distance, donor_m, tile)
    if best is None:
        return None
    threadblock, warp, stages = best[2]  # type: ignore[misc]
    return (tuple(threadblock), tuple(warp), stages), best[1]  # type: ignore[return-value]


def _compile_candidate(specialization: CutlassGemmI8Specialization, tile: Tile, directory: Path,
                       include_directories: Sequence[Path | str]) -> Path | None:
    """Build one candidate's timing binary in `directory`; None if it does not compile."""
    source_path = directory / "sweep.cu"
    binary_path = directory / "sweep"
    source_path.write_text(_render(_KERNEL_TEMPLATE + _SWEEP_MAIN, specialization, tile), encoding="utf-8")
    command = [str(_NVCC), "-std=c++17", "-O3", "--expt-relaxed-constexpr", "-w",
               f"-arch=sm_{specialization.architecture}"]
    for include in include_directories:
        command.extend(("-I", str(include)))
    command.extend((str(source_path), "-o", str(binary_path)))
    if subprocess.run(command, check=False, capture_output=True).returncode:  # noqa: S603
        return None
    return binary_path


def _run_candidate(binary_path: Path | None) -> float:
    """Time one compiled candidate (its own CUDA-graph loop); -1 if it did not build or run."""
    if binary_path is None:
        return -1.0
    completed = subprocess.run([str(binary_path)], check=False, capture_output=True, text=True)  # noqa: S603
    try:
        return float(completed.stdout.strip())
    except ValueError:
        return -1.0


def _time_candidate(specialization: CutlassGemmI8Specialization, tile: Tile,
                    include_directories: Sequence[Path | str]) -> float:
    with TemporaryDirectory(prefix="lc0ex-i8-sweep-") as directory:
        return _run_candidate(_compile_candidate(specialization, tile, Path(directory), include_directories))


def _sweep(specialization: CutlassGemmI8Specialization, key: str,
           include_directories: Sequence[Path | str] = (CUTLASS_INCLUDE,)) -> Tile:
    limit = dynamic_shared_limit()
    _LOGGER.info("sweeping int8 CUTLASS tiles at m=%d n=%d k=%d (%s)", specialization.m, specialization.gemm_n,
                 specialization.k, specialization.family)
    best: tuple[float, Tile] | None = None
    candidates = [candidate for candidate in _SWEEP_CANDIDATES
                  if (candidate[0][0] + candidate[0][1]) * candidate[0][2] * candidate[2] <= limit]
    # B3: the candidates COMPILE in parallel (nvcc is the sweep's cost: ~30 s each) and are TIMED one after another,
    # so no two timings share the card. The picks are the serial sweep's; the wall time is one compile, not nine.
    with TemporaryDirectory(prefix="lc0ex-i8-sweep-") as root:
        directories = [Path(root) / str(index) for index in range(len(candidates))]
        for directory in directories:
            directory.mkdir()
        with ThreadPoolExecutor(max_workers=max(1, len(candidates))) as pool:
            binaries = list(pool.map(lambda pair: _compile_candidate(specialization, pair[0], pair[1],
                                                                     include_directories),
                                     zip(candidates, directories, strict=True)))
        for candidate, binary in zip(candidates, binaries, strict=True):
            microseconds = _run_candidate(binary)
            if microseconds <= 0.0:
                continue
            _LOGGER.info("  %s -> %.2f us", candidate, microseconds)
            if best is None or microseconds < best[0]:
                best = (microseconds, candidate)
    if best is None:
        message = f"no int8 tile compiled and ran for {key}"
        raise RuntimeError(message)
    _LOGGER.info("  best %s at %.2f us", best[1], best[0])
    _store_sweep_result(key, best[1])
    return best[1]


_SHARED_BYTES: dict[tuple[object, ...], int] = {}


def shared_storage_bytes(specialization: CutlassGemmI8Specialization, *,
                         include_directories: Sequence[Path | str] = (CUTLASS_INCLUDE,)) -> int:
    """`sizeof(SharedStorage)` from nvcc (a CUTLASS template computation), cached per tile and epilogue."""
    tile = select_tile(specialization)
    key = (tile, specialization.epilogue, specialization.output)
    cached = _SHARED_BYTES.get(key)
    if cached is not None:
        return cached
    source = _render(_PROBE_TEMPLATE, specialization, tile)
    with TemporaryDirectory(prefix="lc0ex-i8-smem-") as directory:
        source_path = Path(directory) / "probe.cu"
        binary_path = Path(directory) / "probe"
        source_path.write_text(source, encoding="utf-8")
        command = [str(_NVCC), "-std=c++17", "-O0", "--expt-relaxed-constexpr", "-w"]
        for include in include_directories:
            command.extend(("-I", str(include)))
        command.extend((str(source_path), "-o", str(binary_path)))
        subprocess.run(command, check=True)  # noqa: S603
        completed = subprocess.run([str(binary_path)], check=True, capture_output=True, text=True)  # noqa: S603
    measured = int(completed.stdout.strip())
    if measured > dynamic_shared_limit():
        message = f"int8 tile {tile} needs {measured} B of shared memory, over this device's opt-in ceiling"
        raise ValueError(message)
    _SHARED_BYTES[key] = measured
    return measured


def compile_cutlass_gemm_i8(specialization: CutlassGemmI8Specialization, *,
                            include_directories: Sequence[Path | str] = (CUTLASS_INCLUDE,)) -> KernelArtifact:
    """Compile one int8 GEMM specialization into a linker artifact."""
    threadblock, warp, _ = select_tile(specialization)
    cubin = compile_cuda(render_source(specialization), architecture=f"sm_{specialization.architecture}",
                         include_directories=include_directories, extra_arguments=("-w",))
    grid = (-(-specialization.m // threadblock[0]), -(-specialization.gemm_n // threadblock[1]), 1)
    parameters = [_POINTER] * 5
    if specialization.epilogue == "residual":
        parameters.append(_POINTER)
    if specialization.output == "i8":
        parameters.append(_POINTER)
    return artifact_from_cubin(
        cubin, function=entry_point_name(specialization), parameters=tuple(parameters), grid=grid,
        block=(_thread_count(threadblock, warp), 1, 1),
        dynamic_shared_memory_bytes=shared_storage_bytes(specialization, include_directories=include_directories),
    )


def cutlass_gemm_i8(  # noqa: PLR0913
    builder: ProgramBuilder,
    kernels: KernelCache,
    output: Buffer,
    codes: Buffer,
    weights: Buffer,
    scale: Buffer,
    bias: Buffer,
    specialization: CutlassGemmI8Specialization,
    *,
    skip: Buffer | None = None,
    prescale: Buffer | None = None,
) -> None:
    """Append one int8 GEMM site to an executable graph."""
    if (skip is not None) != (specialization.epilogue == "residual"):
        message = "cutlass_gemm_i8: a skip buffer goes with epilogue='residual' and only with it"
        raise ValueError(message)
    if (prescale is not None) != (specialization.output == "i8"):
        message = "cutlass_gemm_i8: a prescale buffer goes with output='i8' and only with it"
        raise ValueError(message)
    builder.set_target(lc0ex_pb2.Target.VENDOR_NVIDIA, f"sm_{specialization.architecture}")
    kernel = kernels.get(compile_cutlass_gemm_i8, specialization)
    arguments: tuple[Buffer, ...] = (output, codes, weights, scale, bias)
    if skip is not None:
        arguments += (skip,)
    if prescale is not None:
        arguments += (prescale,)
    builder.call(kernel, *arguments, readonly=list(arguments[1:]))
