"""CUTLASS dense GEMM specializations delivered as CUBIN artifacts.

Triton carries the FP16 fusion glue, but it will not carry FP8 or persistent
kernels on sm_89, so those specializations are compiled from CUTLASS by nvcc and
enter the graph through `artifact_from_cubin`. This module is the FP16 proof of
that path: same shapes and same graph position as `matmul`, different compiler.

CUTLASS's device-level API cannot be used here -- its Params constructor needs
the device SM count and the kernel's occupancy, both host-only -- so the kernel
body is built from the threadblock-level Mma and Epilogue, whose Params objects
construct on device from a layout alone.
"""

import json
import math
import logging
import os
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from lc0ex import Buffer, KernelArtifact, ProgramBuilder
from lc0ex.cubin_module_compiler import artifact_from_cubin, compile_cuda
from lc0ex.proto import lc0ex_pb2

from lczero_triton.bt4.kernels._activation import SOFTCAP_UNIT_SERIES_MIN
from lczero_triton.bt4.kernels._cache import KernelCache

_LOGGER = logging.getLogger(__name__)

_POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER
_WARP_SIZE = 32
# sm_89 allows 48 KiB of *static* shared memory per block, which caps a
# 128x64x32 tile at four pipeline stages. Everything here therefore goes
# through `extern __shared__` and declares the size in the artifact; the
# runtime raises CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES per node.
_STATIC_SHARED_LIMIT = 48 * 1024
# The hard per-block ceiling on sm_89 (and, unchanged, on sm_120).
_DYNAMIC_SHARED_LIMIT = 99 * 1024
_NVCC = Path("/usr/local/cuda-12.9/bin/nvcc")

CUTLASS_INCLUDE = Path.home() / "spsa" / "cutlass" / "include"

_TYPES_TEMPLATE = """\
#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/threadblock/default_mma.h>
#include <cutlass/epilogue/threadblock/default_epilogue_tensor_op.h>
// default_epilogue_with_broadcast.h names DefaultEpilogueSimt and the conv
// tile iterators in specializations it never instantiates here, but does not
// include them; without these three the header does not parse at all.
#include <cutlass/epilogue/threadblock/default_epilogue_simt.h>
#include <cutlass/epilogue/threadblock/predicated_tile_iterator_conv.h>
#include <cutlass/epilogue/threadblock/predicated_tile_iterator_strided_dgrad.h>
#include <cutlass/epilogue/threadblock/default_epilogue_with_broadcast.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/linear_combination_generic.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

// LC0's activations, transcribed from `_activation.py` so a fused CUTLASS
// epilogue stays in the same fidelity class as the Triton kernel it replaces.
// Triton lowers `tl.exp` to `mul by log2(e)` + `ex2.approx.f32` (CUDA's __expf)
// and `/` to `div.full.f32` (__fdividef). Those are the FAST forms, and using
// the precise ones is not merely slower -- measured at +6.2 us on FFN1 -- it is
// a different function, so bit-identity needs this exact pair.
namespace lc0act {{

#if defined(__CUDA_ARCH__)
#define LC0_EXP(x) __expf(x)
#define LC0_DIV(a, b) __fdividef((a), (b))
#else
#define LC0_EXP(x) expf(x)
#define LC0_DIV(a, b) ((a) / (b))
#endif

CUTLASS_HOST_DEVICE float mish(float x) {{
  const float e = LC0_EXP(x);
  const float n = e * e + 2.0f * e;
  const float d = LC0_DIV(x, n + 2.0f);
  return x <= -0.6f ? n * d : x - 2.0f * d;
}}
CUTLASS_HOST_DEVICE float swish(float x) {{ return LC0_DIV(x, 1.0f + LC0_EXP(-x)); }}

template <typename T> struct Op;
template <> struct Op<float> {{
  static const bool kIsHeavy = true;
  CUTLASS_HOST_DEVICE float operator()(float const& x) const {{ return {activation_expr}; }}
}};
template <typename T, int N> struct Op<cutlass::Array<T, N>> {{
  static const bool kIsHeavy = true;
  CUTLASS_HOST_DEVICE cutlass::Array<T, N> operator()(
      cutlass::Array<T, N> const& v) const {{
    Op<T> scalar;
    cutlass::Array<T, N> y;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) y[i] = scalar(v[i]);
    return y;
  }}
}};

}}  // namespace lc0act

namespace {{

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
// FP16 accumulate: the full-rate path on GeForce, and what the Triton kernels
// already do, so the two families stay numerically comparable.
using ElementAccumulator = cutlass::half_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

constexpr int kM = {m};
constexpr int kN = {n};
constexpr int kK = {k};

using ThreadblockShape = cutlass::gemm::GemmShape<{tile_m}, {tile_n}, {tile_k}>;
using WarpShape = cutlass::gemm::GemmShape<{warp_m}, {warp_n}, {warp_k}>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

using Mma = typename cutlass::gemm::threadblock::DefaultMma<
    ElementA, LayoutA, 8, ElementB, LayoutB, 8,
    ElementAccumulator, LayoutC,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    ThreadblockShape, WarpShape, InstructionShape, {stages},
    cutlass::arch::OpMultiplyAdd>::ThreadblockMma;

{epilogue_decl}

{epilogue_alias}

union SharedStorage {{
  typename Mma::SharedStorage main_loop;
  typename Epilogue::SharedStorage epilogue;
}};

}}  // namespace
"""

# `sizeof(SharedStorage)` is a CUTLASS template computation, so the only
# trustworthy way to get it into the artifact is to ask nvcc. The probe is a
# host program over the same typedefs; results are cached per tile.
_PROBE_TEMPLATE = (
    _TYPES_TEMPLATE
    + """
#include <cstdio>
int main() {{ std::printf("%zu\\n", sizeof(SharedStorage)); return 0; }}
"""
)

_KERNEL_TEMPLATE = (
    _TYPES_TEMPLATE
    + """
// Destination first, then the readonly operands: the lc0ex graph ABI, and the
// same order builder.call() emits for the Triton matmul family.
extern "C" __global__ __launch_bounds__({threads}) void {entry}(
    ElementC* d, const ElementA* a, const ElementB* b{extra_parameters}) {{
  // Dynamic, not static: a 128x64x32 tile at six stages needs 72 KiB, and
  // static shared memory stops at 48 KiB. CUTLASS's own kernels do this.
  extern __shared__ char lc0ex_shared_base[];
  SharedStorage& shared = *reinterpret_cast<SharedStorage*>(lc0ex_shared_base);

  const int tile_row = blockIdx.x;
  const int tile_column = blockIdx.y;
  if (tile_row * ThreadblockShape::kM >= kM) return;
  if (tile_column * ThreadblockShape::kN >= kN) return;

  const cutlass::MatrixCoord offset_a{{tile_row * ThreadblockShape::kM, 0}};
  const cutlass::MatrixCoord offset_b{{0, tile_column * ThreadblockShape::kN}};
  const int k_iterations = (kK + ThreadblockShape::kK - 1) / ThreadblockShape::kK;

  typename Mma::IteratorA iterator_a(
      typename Mma::IteratorA::Params(LayoutA(kK)),
      const_cast<ElementA*>(a), {{kM, kK}}, threadIdx.x, offset_a);
  typename Mma::IteratorB iterator_b(
      typename Mma::IteratorB::Params(LayoutB(kN)),
      const_cast<ElementB*>(b), {{kK, kN}}, threadIdx.x, offset_b);

  const int warp_index = threadIdx.x / 32;
  const int lane_index = threadIdx.x % 32;

  typename Mma::FragmentC accumulators;
  accumulators.clear();
  Mma mma(shared.main_loop, threadIdx.x, warp_index, lane_index);
  mma(k_iterations, accumulators, iterator_a, iterator_b, accumulators);

{output_op_decl}
  const cutlass::MatrixCoord offset_d{{tile_row * ThreadblockShape::kM,
                                     tile_column * ThreadblockShape::kN}};
  typename Epilogue::OutputTileIterator iterator_d(
      typename Epilogue::OutputTileIterator::Params(LayoutC(kN)), d,
      {{kM, kN}}, threadIdx.x, offset_d);
{source_iterator_decl}
  Epilogue epilogue(shared.epilogue, threadIdx.x, warp_index, lane_index);
{epilogue_call}
}}
"""
)

# (threadblock, warp, stages). Ordered widest first; the first tile that divides
# the problem sensibly is taken. There is no autotuning here yet -- the Triton
# family owns that, and this path exists to carry datatypes Triton cannot.
# The leading entry is cuBLAS's own choice for the encoder FFN2 shape, read off
# its kernel name (`..._128x64_ldg8_stages_32x6_nn`): six stages, 72 KiB.
_TILES: tuple[tuple[tuple[int, int, int], tuple[int, int, int], int], ...] = (
    ((128, 64, 32), (64, 32, 32), 6),
    ((128, 128, 32), (64, 64, 32), 3),
    ((128, 64, 32), (64, 32, 32), 4),
    ((64, 64, 32), (32, 32, 32), 4),
)


# Measured winners on sm_89 (RTX 4090), idle GPU, one slot per device.
# NOTE: these are per-architecture. On another SM count the wave quantisation
# differs and the table must be re-swept -- see LC0EX_SPEEDUPS.md.
# The best tile flips with M and the heuristic below cannot see it: at M=768 the
# card wants more and narrower CTAs, at M=2048 one full wave of 128x128, at
# M=8192 a wider tile again. Keyed by the exact (m, n, k) the builder asks for.
_MEASURED_TILES: dict[
    tuple[int, int, int, int, int],
    tuple[tuple[int, int, int], tuple[int, int, int], int],
] = {
    # encoder FFN2, batch 12 / 16 / 32 / 64 / 128
    (89, 128, 768, 1024, 1536): ((64, 128, 32), (32, 64, 32), 4),
    (89, 128, 1024, 1024, 1536): ((64, 128, 32), (32, 64, 32), 4),
    (89, 128, 2048, 1024, 1536): ((128, 128, 32), (64, 64, 32), 4),
    (89, 128, 4096, 1024, 1536): ((256, 128, 32), (64, 64, 32), 3),
    (89, 128, 8192, 1024, 1536): ((128, 256, 32), (64, 64, 32), 3),
    # encoder FFN1 (bias + Mish), batch 12 / 16 / 32 / 64 / 128
    (89, 128, 768, 1536, 1024): ((128, 128, 32), (64, 64, 32), 3),
    (89, 128, 1024, 1536, 1024): ((128, 128, 32), (64, 64, 32), 3),
    (89, 128, 2048, 1536, 1024): ((128, 128, 32), (64, 64, 32), 3),
    (89, 128, 4096, 1536, 1024): ((128, 128, 32), (64, 64, 32), 3),
    (89, 128, 8192, 1536, 1024): ((256, 128, 32), (64, 64, 32), 3),
    # encoder out-projection, batch 12 / 16 / 32 / 64 / 128
    (89, 128, 768, 1024, 1024): ((64, 128, 32), (32, 64, 32), 5),
    (89, 128, 1024, 1024, 1024): ((64, 128, 32), (32, 64, 32), 4),
    (89, 128, 2048, 1024, 1024): ((128, 128, 32), (64, 64, 32), 3),
    (89, 128, 4096, 1024, 1024): ((256, 128, 32), (64, 64, 32), 3),
    (89, 128, 8192, 1024, 1024): ((256, 128, 32), (64, 64, 32), 3),
}


def device_key() -> tuple[int, int]:
    """Return `(compute capability, SM count)` of the build device.

    Every tile in `_MEASURED_TILES` was swept against **128 SMs**. A 5090 has
    170 and a 5080 has 84, so a tile that is exactly one wave here is 0.75 of a
    wave there and 1.52 waves on the 5080. The table is a per-device fact and
    the key has to say so.
    """
    import torch  # noqa: PLC0415  # only needed on the build path.

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return (
        properties.major * 10 + properties.minor,
        properties.multi_processor_count,
    )


def dynamic_shared_limit() -> int:
    """Ask the device for its opt-in shared-memory ceiling.

    sm_89 and sm_120 both report 101,376 B and an A100 reports 166,912; a
    hard-coded constant is wrong on one of them whichever value is picked. This
    is the mistake `wip-20260825`'s pruner makes with `_MAX_SMEM_BYTES = 160 KB`.
    """
    try:
        import torch  # noqa: PLC0415

        index = torch.cuda.current_device()
        return int(
            torch.cuda.get_device_properties(index).shared_memory_per_block_optin
        )
    except (ImportError, AttributeError, RuntimeError):
        return _DYNAMIC_SHARED_LIMIT


def _tile_override() -> (
    tuple[tuple[int, int, int], tuple[int, int, int], int] | None
):
    """Read one explicit tile out of the environment, for sweeps."""
    raw = os.environ.get("LC0EX_CUTLASS_TILE", "")
    if not raw:
        return None
    values = [int(field) for field in re.split(r"[,x ]+", raw.strip()) if field]
    expected = 7
    if len(values) != expected:
        message = (
            "LC0EX_CUTLASS_TILE wants tile_m,tile_n,tile_k,warp_m,warp_n,"
            "warp_k,stages"
        )
        raise ValueError(message)
    return (
        (values[0], values[1], values[2]),
        (values[3], values[4], values[5]),
        values[6],
    )


@dataclass(frozen=True, slots=True)
class CutlassMatmulSpecialization:
    """Immutable row-major FP16 GEMM carried by a CUTLASS CUBIN."""

    m: int
    n: int
    k: int
    architecture: int
    has_bias: bool = False
    activation: str = "none"
    # Fuses `D = act(Acc + bias) * alpha + skip`, which is what wip-20260825's
    # `_matmul_skip_kernel` computes for the encoder out-projection and FFN2.
    has_skip: bool = False
    # Round 20b P2: sigmoid(x @ W_gate + b_gate) * (x @ W_up + b_up) from one [k, 2n]
    # weight matrix as a dual GEMM; requires has_bias (the [2n] gate|up bias).
    glu: bool = False
    # BT6-test (09-21): the lab's `ffn_softcap` on the sigmoid GLU -- both branches pass
    # through c*tanh(./c) before the product (`_activation` gate "glu_capped"). 0 = off,
    # and then the rendered source is byte-identical to the uncapped family's.
    glu_softcap: float = 0.0


def _select_tile(
    specialization: CutlassMatmulSpecialization,
) -> tuple[tuple[int, int, int], tuple[int, int, int], int]:
    """Pick the measured tile for this shape, or the widest that fits."""
    override = _tile_override()
    if override is not None:
        return override
    architecture, multiprocessors = device_key()
    shape = (specialization.m, specialization.n, specialization.k)
    # The sweep cache outranks the static table: its entries were measured on
    # this device with this epilogue family and this exact kernel, where the
    # table was swept for the plain and fused families only. Three of eight
    # residual shapes want a different tile from the table entry.
    cached = _cached_sweep(specialization, architecture, multiprocessors)
    if cached is not None:
        return cached
    measured = _MEASURED_TILES.get((architecture, multiprocessors, *shape))
    if measured is not None:
        return measured
    # A rung that has never been swept can borrow its nearest neighbour's tile
    # rather than pay a sweep. This is what makes a dense ladder affordable; it
    # is opt-in, and the donor is named in the log so it is never confused with
    # a measurement of this shape.
    if os.environ.get("LC0EX_CUTLASS_TILE_REUSE") == "1":
        nearest = _nearest_cached_tile(specialization, architecture, multiprocessors)
        if nearest is not None:
            tile, donor_m = nearest
            _LOGGER.info(
                "reusing the tile measured at m=%d for m=%d n=%d k=%d (%s): %s",
                donor_m, specialization.m, specialization.n, specialization.k,
                _epilogue_family(specialization), tile,
            )
            return tile

    swept = _swept_tile(specialization, architecture, multiprocessors)
    if swept is not None:
        return swept
    # The heuristic is the last resort and it is known to be wrong: on this very
    # card it picked the wrong tile at M=1024 and M=4096. Say so rather than
    # returning a number that looks measured.
    _LOGGER.warning(
        "no measured CUTLASS tile for sm_%d/%d SMs at m=%d n=%d k=%d, and the "
        "build-time sweep is disabled; falling back to the widest-fits "
        "heuristic, which is not a measured choice",
        architecture, multiprocessors, *shape,
    )
    for threadblock, warp, stages in _TILES:
        if threadblock[0] <= max(specialization.m, threadblock[0]) and (
            threadblock[1] <= max(specialization.n, threadblock[1])
        ):
            return threadblock, warp, stages
    return _TILES[-1]


def _thread_count(threadblock: tuple[int, int, int], warp: tuple[int, int, int]) -> int:
    """Return the block size implied by the threadblock and warp shapes."""
    warps = (threadblock[0] // warp[0]) * (threadblock[1] // warp[1])
    warps *= max(1, threadblock[2] // warp[2])
    return warps * _WARP_SIZE


def entry_point_name(specialization: CutlassMatmulSpecialization) -> str:
    """Return the `extern "C"` symbol for one specialization."""
    suffix = ""
    if specialization.has_bias:
        suffix += "_bias"
    if specialization.activation != "none":
        suffix += f"_{specialization.activation}"
    if specialization.glu:
        suffix += "_glu"
        if specialization.glu_softcap > 0.0:
            suffix += "_cap" + f"{specialization.glu_softcap:g}".replace(".", "p").replace("-", "m").replace("+", "")
    if specialization.has_skip:
        suffix += "_skip"
    return (
        f"lc0ex_cutlass_gemm_f16_m{specialization.m}"
        f"_n{specialization.n}_k{specialization.k}{suffix}"
    )


_SHARED_BYTES_CACHE: dict[tuple[object, ...], int] = {}


def shared_storage_bytes(
    specialization: CutlassMatmulSpecialization,
    *,
    include_directories: Sequence[Path | str] = (CUTLASS_INCLUDE,),
) -> int:
    """Return `sizeof(SharedStorage)` for one specialization, from nvcc."""
    threadblock, warp, stages = _select_tile(specialization)
    # The epilogue family is part of the key: EpilogueWithBroadcast does not
    # have the same SharedStorage as the plain one, and a stale hit here would
    # under-allocate dynamic shared memory with no diagnostic at all.
    key = (threadblock, warp, stages, _epilogue_family(specialization))
    cached = _SHARED_BYTES_CACHE.get(key)
    if cached is not None:
        return cached
    source = _render(_PROBE_TEMPLATE, specialization)
    with TemporaryDirectory(prefix="lc0ex-smem-") as directory:
        source_path = Path(directory) / "probe.cu"
        binary_path = Path(directory) / "probe"
        source_path.write_text(source, encoding="utf-8")
        command = [str(_NVCC), "-std=c++17", "-O0", "--expt-relaxed-constexpr"]
        for include in include_directories:
            command.extend(("-I", str(include)))
        command.extend((str(source_path), "-o", str(binary_path)))
        subprocess.run(command, check=True)  # noqa: S603  # nvcc path is pinned.
        completed = subprocess.run(  # noqa: S603
            [str(binary_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    measured = int(completed.stdout.strip())
    if measured > _DYNAMIC_SHARED_LIMIT:
        message = (
            f"tile {threadblock} x {warp} at {stages} stages needs "
            f"{measured} B of shared memory, over sm_89's "
            f"{_DYNAMIC_SHARED_LIMIT} B ceiling"
        )
        raise ValueError(message)
    _SHARED_BYTES_CACHE[key] = measured
    return measured


_ACTIVATION_EXPRESSIONS = {
    "none": "x",
    "mish": "lc0act::mish(x)",
    "swish": "lc0act::swish(x)",
    "relu": "x > 0.0f ? x : 0.0f",
}

# The plain path keeps CUTLASS's own LinearCombination with alpha=1, beta=0 and
# an FP16 compute type -- byte for byte what round 8 gated bit-identical. The
# bias/activation path needs an FP32 compute type (Triton's epilogue is FP32)
# and NoBetaScaling, which is exactly D = act(Acc + C).
_PLAIN_EPILOGUE = """using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
    ElementAccumulator, ElementAccumulator>;"""

_FUSED_EPILOGUE = """using ElementCompute = float;
using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationGeneric<
    lc0act::Op, ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
    ElementAccumulator, ElementCompute,
    cutlass::epilogue::thread::ScaleType::NoBetaScaling>;"""

_PLAIN_OUTPUT_OP = """  // alpha = 1, beta = 0, so the source tile is never read.
  EpilogueOutputOp output_op(typename EpilogueOutputOp::Params(
      ElementAccumulator(1), ElementAccumulator(0)));"""

_FUSED_OUTPUT_OP = """  // NoBetaScaling: D = activation(accumulator + source), source = the bias.
  EpilogueOutputOp output_op(typename EpilogueOutputOp::Params(
      ElementCompute(1), ElementCompute(1)));"""

# Stride 0 over the [N] bias vector: every row of the "source" tile reads the
# same vector, which is the broadcast, with no extra kernel and no extra buffer.
# The residual family. wip-20260825 folds the encoder's bias, its alpha scaling
# and its residual add into the GEMM epilogue, so a CUTLASS route that only
# fused the bias would be one kernel behind before it started. CUTLASS's
# EpilogueWithBroadcast is the primitive for it: one source *tile* (the skip)
# plus one broadcast *vector* (the bias), both consumed in one pass.
#
# The order below is `_matmul_skip_kernel`'s tail transcribed, and the order is
# the whole point -- bias before the activation, alpha after it, the skip last,
# every step in FP32 over an FP16 accumulator. Any other grouping is a
# different function at FP16 output precision.
_RESIDUAL_EPILOGUE = """using Lc0Acc = ElementAccumulator;
using Lc0Out = ElementC;
using Lc0Compute = float;

struct Lc0ResidualOp {
  using ElementOutput = Lc0Out;
  using ElementAccumulator = Lc0Acc;
  using ElementCompute = Lc0Compute;
  using ElementZ = Lc0Out;
  using ElementT = Lc0Out;

  static int const kElementsPerAccess = 128 / cutlass::sizeof_bits<Lc0Out>::value;
  static int const kCount = kElementsPerAccess;
  // Selects the single-source specialization, whose output functor takes
  // one source fragment and one broadcast fragment.
  static bool const kIsSingleSource = true;
  static bool const kStoreZ = true;
  // No second output, so TensorTileIterator is constructed and never stored.
  static bool const kStoreT = false;
  // Matches LinearCombinationGeneric<lc0act::Op>, which reports heavy.
  static bool const kIsHeavy = true;

  using FragmentAccumulator = cutlass::Array<ElementAccumulator, kElementsPerAccess>;
  using FragmentCompute = cutlass::Array<ElementCompute, kElementsPerAccess>;
  using FragmentC = cutlass::Array<ElementOutput, kElementsPerAccess>;
  using FragmentZ = cutlass::Array<ElementZ, kElementsPerAccess>;
  using FragmentT = cutlass::Array<ElementT, kElementsPerAccess>;

  struct Params {
    ElementCompute alpha;
  };

  ElementCompute alpha;

  CUTLASS_HOST_DEVICE
  explicit Lc0ResidualOp(Params const& params) : alpha(params.alpha) {}

  CUTLASS_HOST_DEVICE bool is_source_needed() const { return true; }
  CUTLASS_HOST_DEVICE void set_k_partition(int, int) {}

  CUTLASS_HOST_DEVICE
  void operator()(FragmentZ& frag_Z, FragmentT&, FragmentAccumulator const& AB,
                  FragmentC const& frag_C1, FragmentCompute const& V) const {
    lc0act::Op<ElementCompute> activation;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      ElementCompute value = ElementCompute(AB[i]) + V[i];
      value = activation(value) * alpha + ElementCompute(frag_C1[i]);
      frag_Z[i] = ElementZ(value);
    }
  }

  // Reached only if the skip tile is absent, which this graph never does.
  CUTLASS_HOST_DEVICE
  void operator()(FragmentZ& frag_Z, FragmentT&, FragmentAccumulator const& AB,
                  FragmentCompute const& V) const {
    lc0act::Op<ElementCompute> activation;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      frag_Z[i] = ElementZ(activation(ElementCompute(AB[i]) + V[i]) * alpha);
    }
  }
};

using EpilogueOutputOp = Lc0ResidualOp;"""

_PLAIN_EPILOGUE_ALIAS = """using Epilogue =
    typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
        ThreadblockShape, typename Mma::Operator,
        ThreadblockShape::kK / WarpShape::kK, EpilogueOutputOp,
        EpilogueOutputOp::kCount>::Epilogue;"""

# ElementTensor is only a type requirement here: kStoreT is false, so the
# tensor iterator is built and never written.
_RESIDUAL_EPILOGUE_ALIAS = """using Epilogue = typename cutlass::epilogue::threadblock::
    DefaultEpilogueWithBroadcastTensorOp<
        ThreadblockShape, typename Mma::Operator,
        ThreadblockShape::kK / WarpShape::kK, ElementC, ElementC, ElementC,
        EpilogueOutputOp, EpilogueOutputOp::kElementsPerAccess>::Epilogue;"""

_RESIDUAL_OUTPUT_OP = """  // alpha is a device scalar, read once per block, as tl.load(alpha) does.
  EpilogueOutputOp output_op(
      typename EpilogueOutputOp::Params{static_cast<Lc0Compute>(*alpha)});"""

# The skip is a full [M, N] tile at the destination's own offset. The tensor
# iterator is required by the signature and never stored (kStoreT = false).
_RESIDUAL_ITERATORS = """  typename Epilogue::OutputTileIterator iterator_skip(
      typename Epilogue::OutputTileIterator::Params(LayoutC(kN)),
      const_cast<ElementC*>(skip), {kM, kN}, threadIdx.x, offset_d);
  typename Epilogue::TensorTileIterator iterator_unused(
      typename Epilogue::TensorTileIterator::Params(LayoutC(kN)), d,
      {kM, kN}, threadIdx.x, offset_d);"""

_PLAIN_EPILOGUE_CALL = """  epilogue(output_op, iterator_d, accumulators, {source_iterator});"""

# The broadcast pointer is advanced to this threadblock's first column, which
# is the contract EpilogueWithBroadcast's own GEMM kernels observe; a null
# pointer would read as zeros, so an absent bias stays correct.
_RESIDUAL_EPILOGUE_CALL = """  epilogue(output_op, {broadcast_pointer}, iterator_d, accumulators,
           iterator_skip, iterator_unused,
           cutlass::MatrixCoord(kM, kN), offset_d);"""


_BIAS_ITERATOR = """  typename Epilogue::OutputTileIterator iterator_bias(
      typename Epilogue::OutputTileIterator::Params(LayoutC(0)),
      const_cast<ElementC*>(bias), {kM, kN}, threadIdx.x, offset_d);"""


def _epilogue_family(specialization: CutlassMatmulSpecialization) -> str:
    """Name the epilogue this specialization needs: plain, fused, residual or glu."""
    if specialization.glu:
        return "glu"
    if specialization.has_skip:
        return "residual"
    if specialization.has_bias or specialization.activation != "none":
        return "fused"
    return "plain"


def _render(
    template: str,
    specialization: CutlassMatmulSpecialization,
    tile: tuple[tuple[int, int, int], tuple[int, int, int], int] | None = None,
) -> str:
    """Fill one of the CUDA templates for a specialization.

    `tile` forces a candidate rather than consulting `_select_tile`, which is
    what lets the sweep render a kernel without recursing into tile selection.
    """
    threadblock, warp, stages = tile or _select_tile(specialization)
    family = _epilogue_family(specialization)
    if family == "glu":
        if not specialization.has_bias or specialization.has_skip:
            message = "the glu family takes a [2n] bias and no skip"
            raise ValueError(message)
        template = _GLU_TEMPLATES.get(id(template), template)
    if family == "residual":
        parameters = ", const ElementC* bias" if specialization.has_bias else ""
        parameters += ", const ElementC* skip, const ElementC* alpha"
        epilogue_call = _RESIDUAL_EPILOGUE_CALL.format(
            broadcast_pointer=(
                "bias + offset_d.column()" if specialization.has_bias else "nullptr"
            ),
        )
        epilogue_decl = _RESIDUAL_EPILOGUE
        epilogue_alias = _RESIDUAL_EPILOGUE_ALIAS
        output_op_decl = _RESIDUAL_OUTPUT_OP
        source_iterator_decl = _RESIDUAL_ITERATORS
    else:
        fused = family == "fused"
        parameters = ", const ElementC* bias" if specialization.has_bias else ""
        epilogue_call = _PLAIN_EPILOGUE_CALL.format(
            source_iterator=("iterator_bias" if specialization.has_bias else "iterator_d"),
        )
        epilogue_decl = _FUSED_EPILOGUE if fused else _PLAIN_EPILOGUE
        epilogue_alias = _PLAIN_EPILOGUE_ALIAS
        output_op_decl = _FUSED_OUTPUT_OP if fused else _PLAIN_OUTPUT_OP
        source_iterator_decl = _BIAS_ITERATOR if specialization.has_bias else ""
    return template.format(
        activation_expr=_ACTIVATION_EXPRESSIONS[specialization.activation],
        epilogue_decl=epilogue_decl,
        epilogue_alias=epilogue_alias,
        output_op_decl=output_op_decl,
        extra_parameters=parameters,
        source_iterator_decl=source_iterator_decl,
        epilogue_call=epilogue_call,
        m=specialization.m,
        n=specialization.n,
        k=specialization.k,
        tile_m=threadblock[0],
        tile_n=threadblock[1],
        tile_k=threadblock[2],
        warp_m=warp[0],
        warp_n=warp[1],
        warp_k=warp[2],
        stages=stages,
        threads=_thread_count(threadblock, warp),
        entry=entry_point_name(specialization),
        sweep_arguments=_sweep_arguments(specialization),
        b_columns="2 * kN" if specialization.glu else "kN",
        bias_count="2 * kN" if specialization.glu else "kN",
        glu_include=str(_GLU_EXAMPLE),
        **_glu_softcap_fields(specialization),
    )


_GLU_SOFTCAP_DECL = """// BT6-test `ffn_softcap`: c * tanh(x / c) in the two-sided exponential form of the Triton
// lowering (`_activation._softcap`), so both routes stay in one fidelity class.
constexpr float kGluSoftcap = {cap!r}f;
CUTLASS_HOST_DEVICE float lc0_glu_softcap(float x) {{
  const float magnitude = LC0_DIV(x < 0.0f ? -x : x, kGluSoftcap);
  const float decay = LC0_EXP(-2.0f * magnitude);
  const float tangent = LC0_DIV(1.0f - decay, 1.0f + decay);
  return kGluSoftcap * (x < 0.0f ? -tangent : tangent);
}}
// The gate is a sigmoid, so it lies in (0, 1): `_activation._softcap_unit`, the same rule and the same bound.
CUTLASS_HOST_DEVICE float lc0_glu_softcap_unit(float g) {{
{unit_body}
}}
"""
_GLU_SOFTCAP_UNIT_SERIES = """  const float ratio = g * (1.0f / kGluSoftcap);
  const float square = ratio * ratio;
  return g * (1.0f - square * ((1.0f / 3.0f) - square * (2.0f / 15.0f)));"""
_GLU_SOFTCAP_UNIT_GENERAL = "  return lc0_glu_softcap(g);"


def _glu_softcap_fields(specialization: CutlassMatmulSpecialization) -> dict[str, str]:
    """The four template fields of the GLU functor; with no cap they render the uncapped source unchanged."""
    cap = float(specialization.glu_softcap)
    if cap < 0.0 or cap != cap or cap == float("inf"):
        message = f"glu_softcap must be a finite value >= 0 (0 = off); got {cap}"
        raise ValueError(message)
    if cap > 0.0 and not specialization.glu:
        message = "glu_softcap caps the two GLU branches; it needs glu=True"
        raise ValueError(message)
    if cap == 0.0:
        return {
            "glu_softcap_decl": "",
            "glu_gate_expr": "gate",
            "glu_up_expr_i": "static_cast<float>(rhs[i])",
            "glu_up_expr": "static_cast<float>(rhs)",
        }
    unit_body = _GLU_SOFTCAP_UNIT_SERIES if cap >= SOFTCAP_UNIT_SERIES_MIN else _GLU_SOFTCAP_UNIT_GENERAL
    return {
        "glu_softcap_decl": _GLU_SOFTCAP_DECL.format(cap=cap, unit_body=unit_body),
        "glu_gate_expr": "lc0_glu_softcap_unit(gate)",
        "glu_up_expr_i": "lc0_glu_softcap(static_cast<float>(rhs[i]))",
        "glu_up_expr": "lc0_glu_softcap(static_cast<float>(rhs))",
    }


def _sweep_arguments(specialization: CutlassMatmulSpecialization) -> str:
    """The kernel's argument list for the sweep harness, in signature order."""
    arguments = [
        "static_cast<ElementC*>(d)",
        "static_cast<const ElementA*>(a)",
        "static_cast<const ElementB*>(b)",
    ]
    if specialization.has_bias:
        arguments.append("static_cast<const ElementC*>(bias)")
    if specialization.has_skip:
        arguments.append("static_cast<const ElementC*>(skip)")
        arguments.append("static_cast<const ElementC*>(alpha)")
    return ", ".join(arguments)


# Timing only: the operands are zeroed, which is fine because a GEMM's runtime
# does not depend on its values, and it keeps the harness to one compile.
_SWEEP_MAIN = r"""
#include <cuda_runtime.h>
#include <cstdio>

int main() {{
  void* d = nullptr;
  void* a = nullptr;
  void* b = nullptr;
  void* bias = nullptr;
  void* skip = nullptr;
  void* alpha = nullptr;
  if (cudaMalloc(&d, (size_t)kM * kN * sizeof(ElementC)) != cudaSuccess ||
      cudaMalloc(&a, (size_t)kM * kK * sizeof(ElementA)) != cudaSuccess ||
      cudaMalloc(&b, (size_t)kK * {b_columns} * sizeof(ElementB)) != cudaSuccess ||
      cudaMalloc(&bias, (size_t){bias_count} * sizeof(ElementC)) != cudaSuccess ||
      cudaMalloc(&skip, (size_t)kM * kN * sizeof(ElementC)) != cudaSuccess ||
      cudaMalloc(&alpha, sizeof(ElementC)) != cudaSuccess) {{
    std::printf("-1\n");
    return 0;
  }}
  cudaMemset(d, 0, (size_t)kM * kN * sizeof(ElementC));
  cudaMemset(a, 0, (size_t)kM * kK * sizeof(ElementA));
  cudaMemset(b, 0, (size_t)kK * {b_columns} * sizeof(ElementB));
  cudaMemset(bias, 0, (size_t){bias_count} * sizeof(ElementC));
  cudaMemset(skip, 0, (size_t)kM * kN * sizeof(ElementC));
  cudaMemset(alpha, 0, sizeof(ElementC));

  const int shared_bytes = (int)sizeof(SharedStorage);
  if (cudaFuncSetAttribute((const void*)&{entry},
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           shared_bytes) != cudaSuccess) {{
    std::printf("-1\n");
    return 0;
  }}
  const dim3 grid((kM + ThreadblockShape::kM - 1) / ThreadblockShape::kM,
                  (kN + ThreadblockShape::kN - 1) / ThreadblockShape::kN, 1);
  const dim3 block({threads}, 1, 1);

  for (int i = 0; i < 20; ++i) {{
    {entry}<<<grid, block, shared_bytes>>>({sweep_arguments});
  }}
  if (cudaDeviceSynchronize() != cudaSuccess) {{
    std::printf("-1\n");
    return 0;
  }}
  // Timed as lc0ex serves: 200 launches captured in ONE CUDA graph, the best of 5 replays (a plain-stream loop rounds
  // every kernel period up to a multiple of 2.048 us on some sm_120 drivers; backend sm_120 report 09-28 §3).
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  for (int i = 0; i < 200; ++i) {{
    {entry}<<<grid, block, shared_bytes, stream>>>({sweep_arguments});
  }}
  cudaStreamEndCapture(stream, &graph);
  if (cudaGraphInstantiate(&exec, graph, 0) != cudaSuccess) {{
    std::printf("-1\n");
    return 0;
  }}
  cudaGraphLaunch(exec, stream);
  cudaStreamSynchronize(stream);
  cudaEvent_t start;
  cudaEvent_t stop;
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

_SWEEP_TEMPLATE = _KERNEL_TEMPLATE + _SWEEP_MAIN


# ---------------------------------------------------------------------------
# The `glu` family (round 20b P2): sigmoid(x @ Wg + bg) * (x @ Wu + bu) as one
# CUTLASS dual GEMM (examples/45_dual_gemm). See `CutlassMatmulSpecialization.glu`.
_LC0ACT_MARKER = "}}  // namespace lc0act\n"
_LC0ACT_HEAD = _TYPES_TEMPLATE[: _TYPES_TEMPLATE.index(_LC0ACT_MARKER) + len(_LC0ACT_MARKER)]
_GLU_EXAMPLE = CUTLASS_INCLUDE.parent / "examples" / "45_dual_gemm"

_GLU_TYPES_TEMPLATE = _LC0ACT_HEAD + """
#include "{glu_include}/threadblock/dual_mma_multistage.h"
#include "{glu_include}/threadblock/dual_epilogue.h"

namespace {{

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementAccumulator = cutlass::half_t;
using ElementCompute = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

constexpr int kM = {m};
constexpr int kN = {n};
constexpr int kK = {k};

using ThreadblockShape = cutlass::gemm::GemmShape<{tile_m}, {tile_n}, {tile_k}>;
using WarpShape = cutlass::gemm::GemmShape<{warp_m}, {warp_n}, {warp_k}>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

using GluMma = typename cutlass::gemm::threadblock::DefaultMma<
    ElementA, LayoutA, 8, ElementB, LayoutB, 8,
    ElementAccumulator, LayoutC,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    ThreadblockShape, WarpShape, InstructionShape, {stages},
    cutlass::arch::OpMultiplyAdd>::ThreadblockMma;

// D = accumulator + bias in FP32, the fused family's NoBetaScaling form with the
// identity activation.
using GluOp0 = cutlass::epilogue::thread::LinearCombinationGeneric<
    lc0act::Op, ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
    ElementAccumulator, ElementCompute,
    cutlass::epilogue::thread::ScaleType::NoBetaScaling>;

// sigmoid(gate) * up, the lab's GLU (`_activation.apply_glu`, gate "glu"), written with
// the same fast exp/div pair as the Triton lowering.
{glu_softcap_decl}struct Lc0SigmoidAndMul {{
  using ElementOutput = ElementC;
  using ElementAccumulator = ElementC;
  using ElementCompute = float;
  static int const kCount = 128 / cutlass::sizeof_bits<ElementC>::value;
  using FragmentOutput = cutlass::Array<ElementOutput, kCount>;
  using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
  struct Params {{}};
  CUTLASS_HOST_DEVICE explicit Lc0SigmoidAndMul(Params const&) {{}}
  CUTLASS_HOST_DEVICE bool is_source_needed() const {{ return true; }}
  CUTLASS_HOST_DEVICE void set_k_partition(int, int) {{}}
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const& lhs, FragmentAccumulator const& rhs) const {{
    FragmentOutput out;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {{
      const float gate = LC0_DIV(1.0f, 1.0f + LC0_EXP(-static_cast<float>(lhs[i])));
      out[i] = ElementOutput({glu_gate_expr} * {glu_up_expr_i});
    }}
    return out;
  }}
  CUTLASS_HOST_DEVICE
  ElementOutput operator()(ElementAccumulator const& lhs, ElementAccumulator const& rhs) const {{
    const float gate = LC0_DIV(1.0f, 1.0f + LC0_EXP(-static_cast<float>(lhs)));
    return ElementOutput({glu_gate_expr} * {glu_up_expr});
  }}
}};

}}  // namespace

// Resolved inside cutlass::gemm / cutlass::epilogue exactly as the example's own
// device header resolves them (threadblock::, SharedMemoryClearOption).
namespace cutlass {{ namespace gemm {{ namespace lc0glu {{
using DualMma = threadblock::DualMmaMultistage<
    typename GluMma::Shape,
    typename GluMma::IteratorA, typename GluMma::SmemIteratorA, GluMma::kCacheOpA,
    typename GluMma::IteratorB, typename GluMma::SmemIteratorB, GluMma::kCacheOpB,
    typename GluMma::IteratorB, typename GluMma::SmemIteratorB,
    typename GluMma::ElementC, typename GluMma::LayoutC,
    typename GluMma::Policy, typename GluMma::Policy,
    GluMma::kStages, SharedMemoryClearOption::kNone>;
}} }} }}

namespace cutlass {{ namespace epilogue {{ namespace lc0glu {{
static int const kPartitionsK = ThreadblockShape::kK / WarpShape::kK;
using Epilogue0 = typename threadblock::DefaultEpilogueTensorOp<
    ThreadblockShape, typename cutlass::gemm::lc0glu::DualMma::Operator0, kPartitionsK,
    GluOp0, GluOp0::kCount>::Epilogue;
using DualEpilogue = threadblock::DualEpilogue<
    typename Epilogue0::Shape,
    typename Epilogue0::WarpMmaOperator,
    Epilogue0::kPartitionsK,
    typename Epilogue0::OutputTileIterator,
    typename Epilogue0::AccumulatorFragmentIterator,
    typename Epilogue0::WarpTileIterator,
    typename Epilogue0::SharedLoadIterator,
    GluOp0, GluOp0, Lc0SigmoidAndMul,
    typename Epilogue0::Padding,
    false, false,
    Epilogue0::kFragmentsPerIteration,
    true>;
}} }} }}

namespace {{
union SharedStorage {{
  typename cutlass::gemm::lc0glu::DualMma::SharedStorage main_loop;
  typename cutlass::epilogue::lc0glu::DualEpilogue::SharedStorage epilogue;
}};
}}  // namespace
"""

_GLU_PROBE_TEMPLATE = _GLU_TYPES_TEMPLATE + _PROBE_TEMPLATE[len(_TYPES_TEMPLATE):]

_GLU_KERNEL_TEMPLATE = _GLU_TYPES_TEMPLATE + """
extern "C" __global__ __launch_bounds__({threads}) void {entry}(
    ElementC* d, const ElementA* a, const ElementB* b, const ElementC* bias) {{
  extern __shared__ char lc0ex_shared_base[];
  SharedStorage& shared = *reinterpret_cast<SharedStorage*>(lc0ex_shared_base);
  using DualMma = cutlass::gemm::lc0glu::DualMma;
  using DualEpilogue = cutlass::epilogue::lc0glu::DualEpilogue;
  using OutputIterator = typename cutlass::epilogue::lc0glu::Epilogue0::OutputTileIterator;

  const int tile_row = blockIdx.x;
  const int tile_column = blockIdx.y;
  if (tile_row * ThreadblockShape::kM >= kM) return;
  if (tile_column * ThreadblockShape::kN >= kN) return;

  const cutlass::MatrixCoord offset_a{{tile_row * ThreadblockShape::kM, 0}};
  const cutlass::MatrixCoord offset_b{{0, tile_column * ThreadblockShape::kN}};
  const int k_iterations = (kK + ThreadblockShape::kK - 1) / ThreadblockShape::kK;

  // One [K, 2N] row-major weight matrix, gate columns first: both operands step
  // rows by 2N, and the up operand starts N columns in.
  typename DualMma::IteratorA iterator_a(
      typename DualMma::IteratorA::Params(LayoutA(kK)),
      const_cast<ElementA*>(a), {{kM, kK}}, threadIdx.x, offset_a);
  typename DualMma::IteratorB0 iterator_gate(
      typename DualMma::IteratorB0::Params(LayoutB(2 * kN)),
      const_cast<ElementB*>(b), {{kK, kN}}, threadIdx.x, offset_b);
  typename DualMma::IteratorB1 iterator_up(
      typename DualMma::IteratorB1::Params(LayoutB(2 * kN)),
      const_cast<ElementB*>(b + kN), {{kK, kN}}, threadIdx.x, offset_b);

  const int warp_index = threadIdx.x / 32;
  const int lane_index = threadIdx.x % 32;

  typename DualMma::FragmentC gate_accumulators;
  typename DualMma::FragmentC up_accumulators;
  gate_accumulators.clear();
  up_accumulators.clear();
  DualMma mma(shared.main_loop, threadIdx.x, warp_index, lane_index);
  mma(k_iterations, gate_accumulators, up_accumulators, iterator_a, iterator_gate, iterator_up,
      gate_accumulators, up_accumulators);

  GluOp0 bias_op(typename GluOp0::Params(ElementCompute(1), ElementCompute(1)));
  Lc0SigmoidAndMul gate_op{{typename Lc0SigmoidAndMul::Params()}};
  const cutlass::MatrixCoord offset_d{{tile_row * ThreadblockShape::kM,
                                     tile_column * ThreadblockShape::kN}};
  OutputIterator iterator_d(typename OutputIterator::Params(LayoutC(kN)), d,
                            {{kM, kN}}, threadIdx.x, offset_d);
  OutputIterator iterator_gate_bias(typename OutputIterator::Params(LayoutC(0)),
                                    const_cast<ElementC*>(bias), {{kM, kN}}, threadIdx.x, offset_d);
  OutputIterator iterator_up_bias(typename OutputIterator::Params(LayoutC(0)),
                                  const_cast<ElementC*>(bias + kN), {{kM, kN}}, threadIdx.x, offset_d);
  OutputIterator sources[2] = {{iterator_gate_bias, iterator_up_bias}};
  DualEpilogue epilogue(shared.epilogue, threadIdx.x, warp_index, lane_index);
  // D0 and D1 are not stored (kStoreD0 = kStoreD1 = false); only D2 reaches `d`.
  epilogue(bias_op, bias_op, gate_op, iterator_d, iterator_d, iterator_d,
           gate_accumulators, up_accumulators, sources, true);
}}
"""

_GLU_SWEEP_TEMPLATE = _GLU_KERNEL_TEMPLATE + _SWEEP_MAIN

_GLU_TEMPLATES = {
    id(_KERNEL_TEMPLATE): _GLU_KERNEL_TEMPLATE,
    id(_PROBE_TEMPLATE): _GLU_PROBE_TEMPLATE,
    id(_SWEEP_TEMPLATE): _GLU_SWEEP_TEMPLATE,
}

# The candidates worth trying on an unknown device: every tile the sm_89 sweep
# ever chose, plus the narrow and wide ends, since the wave count -- and so the
# winner -- moves with the SM count.
_SWEEP_CANDIDATES: tuple[
    tuple[tuple[int, int, int], tuple[int, int, int], int], ...
] = (
    ((64, 64, 32), (32, 32, 32), 4),
    ((64, 128, 32), (32, 64, 32), 4),
    ((64, 128, 32), (32, 64, 32), 5),
    ((128, 64, 32), (64, 32, 32), 4),
    ((128, 64, 32), (64, 32, 32), 6),
    ((128, 128, 32), (64, 64, 32), 3),
    ((128, 256, 32), (64, 64, 32), 3),
    ((256, 128, 32), (64, 64, 32), 3),
)

_SWEEP_CACHE_PATH = Path(
    os.environ.get(
        "LC0EX_CUTLASS_TILE_CACHE",
        str(Path.home() / ".cache" / "lc0ex" / "cutlass_tiles.json"),
    )
)


# One process has to resolve one specialization to ONE tile. `_select_tile` is
# consulted three times for a single kernel -- when the dynamic shared memory is
# probed (`shared_storage_bytes`), when the body is rendered (`_render`) and again
# in `compile_cutlass_matmul` for the launch grid -- so reopening the file at each
# of those lets a concurrent builder's rewrite land between them. Round 22 built
# two ladders beside a third build against one shared JSON and compiled a rung-8
# QKV GEMM for one tile while launching it with another's shared memory: every
# rung gated clean and every backendbench batch died on an invalid __shared__
# write. The file is therefore read once per process and per path, and a sweep
# this process ran outranks anything that appears on disk afterwards -- the tile
# it compiled for is the tile it must launch.
_SWEEP_CACHE_MEMO: dict[Path, dict[str, list[object]]] = {}
# What THIS process measured, so a store can merge onto a file other builders
# have written to since, without ever adopting their answer for our own shapes.
_SWEEP_CACHE_OWN: dict[str, object] = {}


def _read_sweep_cache_file() -> dict[str, list[object]]:
    """Read the on-disk sweep results, tolerating a missing or corrupt file."""
    try:
        loaded = json.loads(_SWEEP_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _load_sweep_cache() -> dict[str, list[object]]:
    """Return this process's view of the sweep cache, read from disk once.

    The memo is keyed on the path, so pointing `LC0EX_CUTLASS_TILE_CACHE` at
    another file still takes effect; only a rewrite of the file this process is
    already using is ignored, which is exactly the race.
    """
    cache = _SWEEP_CACHE_MEMO.get(_SWEEP_CACHE_PATH)
    if cache is None:
        cache = _read_sweep_cache_file()
        _SWEEP_CACHE_MEMO.clear()
        _SWEEP_CACHE_MEMO[_SWEEP_CACHE_PATH] = cache
    return cache


def _replace_sweep_cache_file(cache: dict[str, object]) -> None:
    """Write the cache beside the target and rename it on, so readers see one state."""
    temporary = _SWEEP_CACHE_PATH.with_name(f"{_SWEEP_CACHE_PATH.name}.{os.getpid()}.tmp")
    try:
        # O_CREAT respects the umask, and an existing cache keeps the mode it had:
        # `mkstemp` would hand back at 0600 a file that several builds -- and, on a
        # shared box, several users -- have always been able to read.
        descriptor = os.open(temporary, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o666)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(cache, indent=1))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.chmod(temporary, _SWEEP_CACHE_PATH.stat().st_mode & 0o777)
        except OSError:
            pass
        os.replace(temporary, _SWEEP_CACHE_PATH)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _store_sweep_result(key: str, tile: object) -> None:
    """Add one measured tile to this process's view and to the on-disk cache."""
    cache = _load_sweep_cache()
    cache[key] = tile
    _SWEEP_CACHE_OWN[key] = tile
    # Merge onto what is on disk NOW: another builder's entries survive, ours win.
    merged: dict[str, object] = {**_read_sweep_cache_file(), **_SWEEP_CACHE_OWN}
    try:
        _SWEEP_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _replace_sweep_cache_file(merged)
    except OSError:
        _LOGGER.warning("could not write the tile cache at %s", _SWEEP_CACHE_PATH)


def _time_candidate(
    specialization: CutlassMatmulSpecialization,
    tile: tuple[tuple[int, int, int], tuple[int, int, int], int],
    include_directories: Sequence[Path | str],
) -> float:
    """Compile and time one candidate tile; a negative result means unusable."""
    source = _render(_SWEEP_TEMPLATE, specialization, tile=tile)
    with TemporaryDirectory(prefix="lc0ex-sweep-") as directory:
        source_path = Path(directory) / "sweep.cu"
        binary_path = Path(directory) / "sweep"
        source_path.write_text(source, encoding="utf-8")
        command = [
            str(_NVCC), "-std=c++17", "-O3", "--expt-relaxed-constexpr", "-w",
            f"-arch=sm_{specialization.architecture}",
        ]
        for include in include_directories:
            command.extend(("-I", str(include)))
        command.extend((str(source_path), "-o", str(binary_path)))
        if subprocess.run(command, check=False, capture_output=True).returncode:  # noqa: S603
            return -1.0
        completed = subprocess.run(  # noqa: S603
            [str(binary_path)], check=False, capture_output=True, text=True
        )
    try:
        return float(completed.stdout.strip())
    except ValueError:
        return -1.0


def _sweep_key(
    specialization: CutlassMatmulSpecialization,
    architecture: int,
    multiprocessors: int,
) -> str:
    """The cache key: device, shape and epilogue family."""
    return "/".join(
        str(part)
        for part in (
            architecture, multiprocessors, specialization.m, specialization.n,
            specialization.k, _epilogue_family(specialization),
        )
    )


def _cached_sweep(
    specialization: CutlassMatmulSpecialization,
    architecture: int,
    multiprocessors: int,
) -> tuple[tuple[int, int, int], tuple[int, int, int], int] | None:
    """Return a previously swept tile for this exact kernel, if there is one."""
    cached = _load_sweep_cache().get(
        _sweep_key(specialization, architecture, multiprocessors)
    )
    if cached is None:
        return None
    threadblock, warp, stages = cached
    return (tuple(threadblock), tuple(warp), stages)  # type: ignore[return-value]


def _nearest_cached_tile(
    specialization: CutlassMatmulSpecialization,
    architecture: int,
    multiprocessors: int,
) -> tuple[tuple[tuple[int, int, int], tuple[int, int, int], int], int] | None:
    """Return a neighbouring rung's measured tile and the M it was measured at.

    Only entries for the same device, the same `(n, k)` and the same epilogue
    family are eligible -- the tile depends on all three (R28). Nearest is taken
    in log-M, because the tile tracks the wave count, which is multiplicative.
    """
    family = _epilogue_family(specialization)
    prefix = f"{architecture}/{multiprocessors}/"
    suffix = f"/{specialization.n}/{specialization.k}/{family}"
    best: tuple[float, int, object] | None = None
    for key, tile in _load_sweep_cache().items():
        if not key.startswith(prefix) or not key.endswith(suffix):
            continue
        try:
            donor_m = int(key[len(prefix) :].split("/", 1)[0])
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
    return ((tuple(threadblock), tuple(warp), stages), best[1])  # type: ignore[return-value]


def _swept_tile(
    specialization: CutlassMatmulSpecialization,
    architecture: int,
    multiprocessors: int,
    *,
    include_directories: Sequence[Path | str] = (CUTLASS_INCLUDE,),
) -> tuple[tuple[int, int, int], tuple[int, int, int], int] | None:
    """Measure the candidate tiles on this device and keep the fastest.

    This is what makes an artifact self-tuning on a card the lane has never
    seen. It costs one nvcc compile per candidate -- minutes per shape -- and
    the result is cached on disk, so it is paid once per (device, shape,
    epilogue family). `LC0EX_CUTLASS_TILE_SWEEP=0` disables it.
    """
    if os.environ.get("LC0EX_CUTLASS_TILE_SWEEP") == "0":
        return None
    key = _sweep_key(specialization, architecture, multiprocessors)
    cached = _cached_sweep(specialization, architecture, multiprocessors)
    if cached is not None:
        return cached

    _LOGGER.info(
        "sweeping CUTLASS tiles for sm_%d/%d SMs at m=%d n=%d k=%d (%s)",
        architecture, multiprocessors, specialization.m, specialization.n,
        specialization.k, _epilogue_family(specialization),
    )
    best: tuple[float, tuple[tuple[int, int, int], tuple[int, int, int], int]] | None = None
    for candidate in _SWEEP_CANDIDATES:
        microseconds = _time_candidate(specialization, candidate, include_directories)
        if microseconds <= 0.0:
            continue
        _LOGGER.info("  %s -> %.2f us", candidate, microseconds)
        if best is None or microseconds < best[0]:
            best = (microseconds, candidate)
    if best is None:
        return None
    _LOGGER.info("  best %s at %.2f us", best[1], best[0])
    _store_sweep_result(key, best[1])
    return best[1]


def render_source(specialization: CutlassMatmulSpecialization) -> str:
    """Render the kernel translation unit for one specialization."""
    return _render(_KERNEL_TEMPLATE, specialization)


def compile_cutlass_matmul(
    specialization: CutlassMatmulSpecialization,
    *,
    include_directories: Sequence[Path | str] = (CUTLASS_INCLUDE,),
) -> KernelArtifact:
    """Compile one CUTLASS GEMM specialization into a linker artifact."""
    threadblock, warp, _ = _select_tile(specialization)
    cubin = compile_cuda(
        render_source(specialization),
        architecture=f"sm_{specialization.architecture}",
        include_directories=include_directories,
    )
    grid = (
        (specialization.m + threadblock[0] - 1) // threadblock[0],
        (specialization.n + threadblock[1] - 1) // threadblock[1],
        1,
    )
    return artifact_from_cubin(
        cubin,
        function=entry_point_name(specialization),
        parameters=(_POINTER, _POINTER, _POINTER)
        + ((_POINTER,) if specialization.has_bias else ())
        + ((_POINTER, _POINTER) if specialization.has_skip else ()),
        grid=grid,
        block=(_thread_count(threadblock, warp), 1, 1),
        dynamic_shared_memory_bytes=shared_storage_bytes(
            specialization,
            include_directories=include_directories,
        ),
    )


# The rendered GEMM declares AlignmentA = AlignmentB = 8, so each operand's row must
# be a whole number of 8-element groups. A width that is not -- the lab static net's
# dff of 683 -- compiles, sweeps and counts as a CUTLASS node, then faults at launch
# with CUDA_ERROR_MISALIGNED_ADDRESS (round 20, rig46 sm_120).
CUTLASS_ALIGNMENT = 8


def cutlass_supports(n: int, k: int) -> bool:
    """Return whether both GEMM widths satisfy the rendered kernel's alignment."""
    return n % CUTLASS_ALIGNMENT == 0 and k % CUTLASS_ALIGNMENT == 0


def cutlass_matmul(
    builder: ProgramBuilder,
    kernels: KernelCache,
    output: Buffer,
    activations: Buffer,
    weights: Buffer,
    specialization: CutlassMatmulSpecialization,
    *,
    bias: Buffer | None = None,
    skip: Buffer | None = None,
    alpha: Buffer | None = None,
) -> None:
    """Append one CUTLASS row-major dense matrix multiplication."""
    if not cutlass_supports(specialization.n, specialization.k):
        message = (
            f"CUTLASS GEMM widths n={specialization.n}, k={specialization.k} are not "
            f"multiples of the rendered alignment {CUTLASS_ALIGNMENT}; the kernel would "
            "fault at launch with CUDA_ERROR_MISALIGNED_ADDRESS. Use matmul."
        )
        raise ValueError(message)
    if (bias is not None) != specialization.has_bias:
        message = "CutlassMatmulSpecialization.has_bias must match the bias argument."
        raise ValueError(message)
    if ((skip is not None) and (alpha is not None)) != specialization.has_skip:
        message = (
            "CutlassMatmulSpecialization.has_skip requires both a skip and an "
            "alpha buffer."
        )
        raise ValueError(message)
    builder.set_target(
        lc0ex_pb2.Target.VENDOR_NVIDIA,
        f"sm_{specialization.architecture}",
    )
    kernel = kernels.get(compile_cutlass_matmul, specialization)
    arguments: tuple[Buffer, ...] = (output, activations, weights)
    if bias is not None:
        arguments += (bias,)
    if specialization.has_skip and skip is not None and alpha is not None:
        arguments += (skip, alpha)
    builder.call(kernel, *arguments, readonly=list(arguments[1:]))
