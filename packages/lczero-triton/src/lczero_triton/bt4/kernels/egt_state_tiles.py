"""EGT2 edge-state read tiles per block, and the FP16 copy of the state (round 21, K2c).

Why this exists
---------------
K2's head program (`attention_egt`) reads the whole FP32 edge state of its sample -- 16 tiles of [64, 64], 262 KB --
and contracts it three times (edge read, door, gate). The state is per sample, so the 32 head programs of one sample
re-read the same 262 KB; at batch 64 that is ~540 MB per block from L2, and the K2 report's split puts it at 131 of
168 us per call. Nothing in the architecture asks for that: each head needs three [64, 64] tiles, which are three
linear maps of the state that depend only on the block's three [heads, 16] plans.

Two ways to read less, both served through the same `edge_state` pointer of `attention_egt`:

* **the FP16 copy** (`cast_state`): the head program keeps its loop but loads 16 FP16 tiles (131 KB). The FP32
  state stays the accumulation chain of the sites; the copy is what the blocks read. One cast per state write
  (the prologue's seed and each of the three sites).
* **the read tiles** (`state_tiles`): one planar kernel per block computes, per sample, the three tiles of every
  head -- ``[heads, 16] x [16, 4096]`` for each of ``edge_read/w``, ``door/w`` and ``gate/w`` in FP32 (IEEE), stored
  FP16 at ``((sample * round_heads + head) * 3 + term) * 4096 + cell`` -- and the head program loads 3 x 8 KB.

The tiles are only cheaper while they stay in L2 (K2's round 2: 6.4 us against 20.4 at batch 8, where they fit; 138
us against 97.6 at batch 64, where a 100 MB FP32 buffer streamed from DRAM). `round_heads` bounds the buffer: the
block's attention runs in ``heads // round_heads`` rounds of (tiles for this round's heads, attention over them), so
the live tile buffer is ``batch * round_heads * 3 * 8 KB`` whatever the batch. `reverse` writes the tiles in
descending sample order, so the attention kernel -- which walks samples ascending -- meets the most recently written
ones first (LRU-friendly when the buffer is near the L2 size).

Both kernels are planar, one program per (sample, run of `pixels` cells), K3's shape. Neither is in place. Every
`tl.dot` pins ``input_precision="ieee"``: Triton 3.7 runs FP32 dots in TF32 by default.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import torch
import triton
import triton.language as tl
from lc0ex import Buffer, KernelArtifact, ProgramBuilder
from lc0ex.proto import lc0ex_pb2
from lc0ex.triton_module_compiler import artifact_from_triton

from lczero_triton.bt4.kernels._cache import KernelCache

# r23 F8: the e4m3 copy of the edge state (item F8). e4m3 (`tl.float8e4nv`) has a 3-bit mantissa, so its error is
# RELATIVE and the scale buys range, not precision: the per-channel scale exists to keep every channel inside
# [2^-6, 448] x scale, where the format is normal. The amax below is this family's, measured over the lane's own
# onnxruntime dumps (`ref/egt/e0.bin`, 1,024 positions, and `ref/egt_k2/gcap/state{0,1,2}.bin`, 64 positions x
# 4,096 cells x 16 channels per stage): the per-channel max over the seed and both dumped post-site stages.
# `EGT_STATE_SAFETY` is headroom for positions outside that draw; it costs nothing (measured: the term error is
# the same at 1.0, 1.25, 2.0 and 4.0) and clipping is zero at every one of them.
E4M3_MAX = 448.0
EGT_STATE_SAFETY = 2.0
EGT_STATE_AMAX = (1.8659, 1.6913, 3.1821, 1.8750, 2.6229, 2.6097, 1.6319, 2.9423,
                  2.8932, 2.3692, 1.7170, 1.8367, 2.6965, 2.9409, 2.2355, 3.4956)


def state_scales(amax: tuple[float, ...] = EGT_STATE_AMAX, safety: float = EGT_STATE_SAFETY) -> tuple[float, ...]:
    """The per-channel e4m3 scale: a stored unit is `scale` of the state's units."""
    return tuple(value * safety / E4M3_MAX for value in amax)


# r23b I8: the int8 copy (design §8). int8 spends every bit on the mantissa, so for it the scale IS the precision --
# the F8 report (§3.5) measured int8 with a TIGHT per-channel scale 4-6x more accurate than e4m3 at the same byte.
# Codes are rounded half up and clamped to +-127 (symmetric: -128 is never written), so a state value past the amax
# saturates silently, as e4m3's conversion does.
INT8_MAX = 127.0
EGT_STATE_SAFETY_I8 = 1.0


def state_scales_i8(amax: tuple[float, ...] = EGT_STATE_AMAX,
                    safety: float = EGT_STATE_SAFETY_I8) -> tuple[float, ...]:
    """r23b I8: the per-channel int8 scale: a stored code is `scale` of the state's units."""
    return tuple(value * safety / INT8_MAX for value in amax)

_POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER
CELLS = 64 * 64
HEADS = 32
STATES = 16
TERMS = 3
# The three block plans the tiles are built from, in kernel argument order (`attention_egt.BLOCK_TABLES` names).
TILE_TABLES = ("edge_read/w", "door/w", "gate/w")
# (pixels per program, num_warps). Runs of 256 cells and above are excluded on K3's finding (pathological at 4
# warps, and 1024-cell programs compile for minutes).
_CONFIGURATIONS = ((32, 2), (64, 2), (64, 4), (128, 4), (128, 8))


@triton.jit
def _store_term(  # noqa: PLR0913
    tiles, values, weights, sample, head_index, channels, cells, term: tl.constexpr, round_heads: tl.constexpr,
    states: tl.constexpr, head_base: tl.constexpr, tiles_f32: tl.constexpr,
) -> None:
    """One term's tiles for this run of cells and every head of the round: ``W[round, 16] . state[16, pixels]``.

    `head_index` has ``max(round_heads, 16)`` rows because ``tl.dot`` needs 16; rows past the round are masked.
    """
    live = head_index < round_heads
    weight = tl.load(weights + (head_base + head_index[:, None]) * states + channels[None, :],
                     mask=live[:, None], other=0.0)
    result = tl.dot(weight, values, input_precision="ieee")  # [rows, pixels]
    offsets = ((sample * round_heads + head_index[:, None]) * 3 + term) * 4096 + cells[None, :]
    if tiles_f32:
        tl.store(tiles + offsets, result, mask=live[:, None])
    else:
        tl.store(tiles + offsets, result.to(tl.float16), mask=live[:, None])


@triton.jit
def _state_tiles_body(  # noqa: PLR0913
    tiles,
    state,
    read_weights,
    door_weights,
    gate_weights,
    batch_count: tl.constexpr,
    round_heads: tl.constexpr,
    states: tl.constexpr,
    head_base: tl.constexpr,
    tiles_f32: tl.constexpr,
    reverse: tl.constexpr,
    pixels: tl.constexpr,
) -> None:
    """The read, door and gate tiles of one run of ``pixels`` cells of one sample, for the round's heads."""
    program = tl.program_id(0)
    runs: tl.constexpr = 4096 // pixels
    sample = program // runs
    if reverse:
        sample = batch_count - 1 - sample
    cells = (program % runs) * pixels + tl.arange(0, pixels)
    channels = tl.arange(0, states)
    rows: tl.constexpr = 16 if round_heads < 16 else round_heads  # tl.dot's minimum M
    head_index = tl.arange(0, rows)
    values = tl.load(state + (sample * states + channels[:, None]) * 4096 + cells[None, :]).to(tl.float32)
    _store_term(tiles, values, read_weights, sample, head_index, channels, cells, 0, round_heads, states, head_base,
                tiles_f32)
    _store_term(tiles, values, door_weights, sample, head_index, channels, cells, 1, round_heads, states, head_base,
                tiles_f32)
    _store_term(tiles, values, gate_weights, sample, head_index, channels, cells, 2, round_heads, states, head_base,
                tiles_f32)


@triton.jit
def _cast_state_body(
    output,
    state,
    batch_count: tl.constexpr,  # noqa: ARG001  # Autotune key and launch grid.
    states: tl.constexpr,
    pixels: tl.constexpr,
) -> None:
    """FP32 state -> FP16 copy, one run of ``pixels`` cells of one sample, all channels."""
    program = tl.program_id(0)
    runs: tl.constexpr = 4096 // pixels
    sample = program // runs
    cells = (program % runs) * pixels + tl.arange(0, pixels)
    channels = tl.arange(0, states)
    offsets = (sample * states + channels[:, None]) * 4096 + cells[None, :]
    tl.store(output + offsets, tl.load(state + offsets).to(tl.float16))


@triton.jit
def _cast_state_cell_major_body(
    output,
    state,
    batch_count: tl.constexpr,  # noqa: ARG001  # Autotune key and launch grid.
    states: tl.constexpr,
    pixels: tl.constexpr,
) -> None:
    """v9: FP32 planar state -> FP16 CELL-MAJOR copy ``[batch, 64, 64, states]`` (a cell's channels contiguous: one 32-byte read
    per cell in `egt_v9.cu`), one run of ``pixels`` cells of one sample."""
    program = tl.program_id(0)
    runs: tl.constexpr = 4096 // pixels
    sample = program // runs
    cells = (program % runs) * pixels + tl.arange(0, pixels)
    channels = tl.arange(0, states)
    values = tl.load(state + (sample * states + channels[:, None]) * 4096 + cells[None, :]).to(tl.float16)
    tl.store(output + (sample * 4096 + cells[None, :]) * states + channels[:, None], values)


@triton.jit
def _cast_state_f8_body(
    output,
    state,
    batch_count: tl.constexpr,  # noqa: ARG001  # Autotune key and launch grid.
    states: tl.constexpr,
    pixels: tl.constexpr,
    inverse_scales: tl.constexpr,
) -> None:
    """r23 F8: e4m3 copy of one run of ``pixels`` cells of one sample, a channel at a time.

    The scale is a compile-time scalar per channel, so the store is one multiply and one saturating convert; the
    head program multiplies the scale back into its plan weight. Unlike the FP16 copy this kernel walks channels
    in a static loop -- a constexpr tuple cannot be a vector -- which is the same traffic in half the write bytes.
    """
    program = tl.program_id(0)
    runs: tl.constexpr = 4096 // pixels
    sample = program // runs
    cells = (program % runs) * pixels + tl.arange(0, pixels)
    for channel in tl.static_range(states):
        offsets = (sample * states + channel) * 4096 + cells
        tl.store(output + offsets, (tl.load(state + offsets) * inverse_scales[channel]).to(tl.float8e4nv))


@triton.jit
def _cast_state_i8_body(
    output,
    state,
    batch_count: tl.constexpr,  # noqa: ARG001  # Autotune key and launch grid.
    states: tl.constexpr,
    pixels: tl.constexpr,
    inverse_scales: tl.constexpr,
) -> None:
    """r23b I8: int8 copy of one run of ``pixels`` cells of one sample, a channel at a time (F8's shape).

    The code is ``clamp(floor(x / scale + 0.5), -127, 127)``; the head program multiplies the scale back into its
    plan weight, as for e4m3.
    """
    program = tl.program_id(0)
    runs: tl.constexpr = 4096 // pixels
    sample = program // runs
    cells = (program % runs) * pixels + tl.arange(0, pixels)
    for channel in tl.static_range(states):
        offsets = (sample * states + channel) * 4096 + cells
        code = tl.clamp(tl.floor(tl.load(state + offsets) * inverse_scales[channel] + 0.5), -127.0, 127.0)
        tl.store(output + offsets, code.to(tl.int8))


_state_tiles_kernel = triton.autotune(
    configs=[triton.Config({"pixels": pixels}, num_warps=warps) for pixels, warps in _CONFIGURATIONS],
    key=["batch_count", "round_heads", "states", "head_base", "tiles_f32", "reverse"],
    cache_results=True,
)(_state_tiles_body)

_cast_state_kernel = triton.autotune(
    configs=[triton.Config({"pixels": pixels}, num_warps=warps) for pixels, warps in _CONFIGURATIONS],
    key=["batch_count", "states"],
    cache_results=True,
)(_cast_state_body)
_cast_state_cell_major_kernel = triton.autotune(
    configs=[triton.Config({"pixels": pixels}, num_warps=warps) for pixels, warps in _CONFIGURATIONS],
    key=["batch_count", "states"],
    cache_results=True,
)(_cast_state_cell_major_body)


_cast_state_f8_kernel = triton.autotune(
    configs=[triton.Config({"pixels": pixels}, num_warps=warps) for pixels, warps in _CONFIGURATIONS],
    key=["batch_count", "states"],
    cache_results=True,
)(_cast_state_f8_body)


_cast_state_i8_kernel = triton.autotune(
    configs=[triton.Config({"pixels": pixels}, num_warps=warps) for pixels, warps in _CONFIGURATIONS],
    key=["batch_count", "states"],
    cache_results=True,
)(_cast_state_i8_body)


@dataclass(frozen=True, slots=True)
class StateTilesSpecialization:
    """Immutable read-tile specialization: `batch_count` samples, `round_heads` heads from `head_base`."""

    batch_count: int
    architecture: int
    round_heads: int = HEADS
    head_base: int = 0
    states: int = STATES
    tiles_f32: bool = False
    reverse: bool = False

    def __post_init__(self) -> None:
        """Reject widths a ``tl.arange`` or ``tl.dot`` tile cannot take (rounds below 16 heads pad the dot)."""
        if self.states < 16 or self.states & (self.states - 1):  # noqa: PLR2004
            message = f"StateTilesSpecialization.states={self.states} must be a power of two >= 16 (tl.dot)"
            raise ValueError(message)
        if self.round_heads <= 0 or self.round_heads & (self.round_heads - 1):
            message = f"StateTilesSpecialization.round_heads={self.round_heads} must be a power of two"
            raise ValueError(message)
        if self.head_base < 0 or self.head_base % self.round_heads:
            message = f"head_base={self.head_base} must be a multiple of round_heads={self.round_heads}"
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class CastStateCellMajorSpecialization:
    """v9: the FP32 -> FP16 cell-major state copy (`egt_v9.cu` reads it)."""

    batch_count: int
    architecture: int
    states: int = STATES


@dataclass(frozen=True, slots=True)
class CastStateSpecialization:
    """Immutable FP32 -> FP16 state copy specialization."""

    batch_count: int
    architecture: int
    states: int = STATES


@dataclass(frozen=True, slots=True)
class CastStateF8Specialization:
    """r23 F8: the e4m3 state copy. `scales` is one build-time constant per channel (`state_scales`)."""

    batch_count: int
    architecture: int
    scales: tuple[float, ...]
    states: int = STATES

    def __post_init__(self) -> None:
        """A scale per channel, all positive: a zero or a missing one is a silently wrong copy."""
        if len(self.scales) != self.states or any(scale <= 0.0 for scale in self.scales):
            message = f"CastStateF8Specialization.scales needs {self.states} positive entries"
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class CastStateI8Specialization:
    """r23b I8: the int8 state copy. `scales` is one build-time constant per channel (`state_scales_i8`)."""

    batch_count: int
    architecture: int
    scales: tuple[float, ...]
    states: int = STATES

    def __post_init__(self) -> None:
        """A scale per channel, all positive: a zero or a missing one is a silently wrong copy."""
        if len(self.scales) != self.states or any(scale <= 0.0 for scale in self.scales):
            message = f"CastStateI8Specialization.scales needs {self.states} positive entries"
            raise ValueError(message)


def tiles_bytes(specialization: StateTilesSpecialization) -> int:
    """The tile buffer one round writes, in bytes."""
    element = 4 if specialization.tiles_f32 else 2
    return element * specialization.batch_count * specialization.round_heads * TERMS * CELLS


def _planar_grid(configuration: Mapping[str, object]) -> tuple[int]:
    return (cast("int", configuration["batch_count"]) * CELLS // cast("int", configuration["pixels"]),)


def launch_state_tiles(  # noqa: PLR0913
    tiles: torch.Tensor,
    state: torch.Tensor,
    read_weights: torch.Tensor,
    door_weights: torch.Tensor,
    gate_weights: torch.Tensor,
    specialization: StateTilesSpecialization,
) -> object:
    """Launch on torch tensors (tests, benchmarks, `compile_state_tiles`)."""
    return _state_tiles_kernel[_planar_grid](
        tiles, state, read_weights, door_weights, gate_weights, specialization.batch_count,
        specialization.round_heads, specialization.states, specialization.head_base, specialization.tiles_f32,
        specialization.reverse,
    )


def launch_cast_state(output: torch.Tensor, state: torch.Tensor, specialization: CastStateSpecialization) -> object:
    """Launch on torch tensors."""
    return _cast_state_kernel[_planar_grid](output, state, specialization.batch_count, specialization.states)


def launch_cast_state_cell_major(output: torch.Tensor, state: torch.Tensor,
                                 specialization: CastStateCellMajorSpecialization) -> object:
    """Launch on torch tensors: `output` is ``[batch, 64, 64, states]`` FP16."""
    return _cast_state_cell_major_kernel[_planar_grid](output, state, specialization.batch_count, specialization.states)


def launch_cast_state_f8(output: torch.Tensor, state: torch.Tensor,
                         specialization: CastStateF8Specialization) -> object:
    """Launch the e4m3 copy on torch tensors."""
    return _cast_state_f8_kernel[_planar_grid](output, state, specialization.batch_count, specialization.states,
                                               inverse_scales=tuple(1.0 / scale for scale in specialization.scales))


def launch_cast_state_i8(output: torch.Tensor, state: torch.Tensor,
                         specialization: CastStateI8Specialization) -> object:
    """r23b I8: launch the int8 copy on torch tensors."""
    return _cast_state_i8_kernel[_planar_grid](output, state, specialization.batch_count, specialization.states,
                                               inverse_scales=tuple(1.0 / scale for scale in specialization.scales))


def compile_state_tiles(specialization: StateTilesSpecialization) -> KernelArtifact:
    """Autotune and compile one read-tile specialization."""
    batch, states, heads = specialization.batch_count, specialization.states, specialization.round_heads
    tile_type = torch.float32 if specialization.tiles_f32 else torch.float16
    tiles = torch.empty((batch, heads, TERMS, 64, 64), dtype=tile_type, device="cuda")
    state = torch.zeros((batch, states, 64, 64), dtype=torch.float32, device="cuda")
    weights = [torch.zeros((specialization.head_base + heads, states), dtype=torch.float32, device="cuda")
               for _ in TILE_TABLES]
    compiled = launch_state_tiles(tiles, state, *weights, specialization)
    pixels = cast("int", _state_tiles_kernel.best_config.kwargs["pixels"])
    return artifact_from_triton(compiled, grid=(batch * CELLS // pixels, 1, 1), parameters=(_POINTER,) * 5,
                                autotuner=_state_tiles_kernel)


def compile_cast_state(specialization: CastStateSpecialization) -> KernelArtifact:
    """Autotune and compile one state-copy specialization."""
    batch, states = specialization.batch_count, specialization.states
    output = torch.empty((batch, states, 64, 64), dtype=torch.float16, device="cuda")
    state = torch.zeros((batch, states, 64, 64), dtype=torch.float32, device="cuda")
    compiled = launch_cast_state(output, state, specialization)
    pixels = cast("int", _cast_state_kernel.best_config.kwargs["pixels"])
    return artifact_from_triton(compiled, grid=(batch * CELLS // pixels, 1, 1), parameters=(_POINTER,) * 2,
                                autotuner=_cast_state_kernel)


def compile_cast_state_cell_major(specialization: CastStateCellMajorSpecialization) -> KernelArtifact:
    """Autotune and compile one cell-major state-copy specialization."""
    batch, states = specialization.batch_count, specialization.states
    output = torch.empty((batch, 64, 64, states), dtype=torch.float16, device="cuda")
    state = torch.zeros((batch, states, 64, 64), dtype=torch.float32, device="cuda")
    compiled = launch_cast_state_cell_major(output, state, specialization)
    pixels = cast("int", _cast_state_cell_major_kernel.best_config.kwargs["pixels"])
    return artifact_from_triton(compiled, grid=(batch * CELLS // pixels, 1, 1), parameters=(_POINTER,) * 2,
                                autotuner=_cast_state_cell_major_kernel)


def compile_cast_state_f8(specialization: CastStateF8Specialization) -> KernelArtifact:
    """Autotune and compile one e4m3 state-copy specialization."""
    batch, states = specialization.batch_count, specialization.states
    output = torch.empty((batch, states, 64, 64), dtype=torch.float8_e4m3fn, device="cuda")
    state = torch.zeros((batch, states, 64, 64), dtype=torch.float32, device="cuda")
    compiled = launch_cast_state_f8(output, state, specialization)
    pixels = cast("int", _cast_state_f8_kernel.best_config.kwargs["pixels"])
    return artifact_from_triton(compiled, grid=(batch * CELLS // pixels, 1, 1), parameters=(_POINTER,) * 2,
                                autotuner=_cast_state_f8_kernel)


def compile_cast_state_i8(specialization: CastStateI8Specialization) -> KernelArtifact:
    """r23b I8: autotune and compile one int8 state-copy specialization."""
    batch, states = specialization.batch_count, specialization.states
    output = torch.empty((batch, states, 64, 64), dtype=torch.int8, device="cuda")
    state = torch.zeros((batch, states, 64, 64), dtype=torch.float32, device="cuda")
    compiled = launch_cast_state_i8(output, state, specialization)
    pixels = cast("int", _cast_state_i8_kernel.best_config.kwargs["pixels"])
    return artifact_from_triton(compiled, grid=(batch * CELLS // pixels, 1, 1), parameters=(_POINTER,) * 2,
                                autotuner=_cast_state_i8_kernel)


def egt_state_tiles(  # noqa: PLR0913
    builder: ProgramBuilder,
    kernels: KernelCache,
    tiles: Buffer,
    state: Buffer,
    tables: Mapping[str, Buffer],
    specialization: StateTilesSpecialization,
) -> None:
    """Append one round's tile build. `tables` maps TILE_TABLES' short names to the block's persistent plans."""
    builder.set_target(lc0ex_pb2.Target.VENDOR_NVIDIA, f"sm_{specialization.architecture}")
    kernel = kernels.get(compile_state_tiles, specialization)
    reads = (state, *(tables[name] for name in TILE_TABLES))
    builder.call(kernel, tiles, *reads, readonly=reads)


def cast_state_i8(
    builder: ProgramBuilder,
    kernels: KernelCache,
    output: Buffer,
    state: Buffer,
    specialization: CastStateI8Specialization,
) -> None:
    """Append one FP32 -> int8 copy of the state (r23b I8: int8 read mode)."""
    builder.set_target(lc0ex_pb2.Target.VENDOR_NVIDIA, f"sm_{specialization.architecture}")
    kernel = kernels.get(compile_cast_state_i8, specialization)
    builder.call(kernel, output, state, readonly=(state,))


def cast_state_f8(
    builder: ProgramBuilder,
    kernels: KernelCache,
    output: Buffer,
    state: Buffer,
    specialization: CastStateF8Specialization,
) -> None:
    """Append one FP32 -> e4m3 copy of the state (r23 F8: e4m3 read mode)."""
    builder.set_target(lc0ex_pb2.Target.VENDOR_NVIDIA, f"sm_{specialization.architecture}")
    kernel = kernels.get(compile_cast_state_f8, specialization)
    builder.call(kernel, output, state, readonly=(state,))


def cast_state_cell_major(
    builder: ProgramBuilder,
    kernels: KernelCache,
    output: Buffer,
    state: Buffer,
    specialization: CastStateCellMajorSpecialization,
) -> None:
    """Append one FP32 -> FP16 cell-major copy of the state (v9's read)."""
    builder.set_target(lc0ex_pb2.Target.VENDOR_NVIDIA, f"sm_{specialization.architecture}")
    kernel = kernels.get(compile_cast_state_cell_major, specialization)
    builder.call(kernel, output, state, readonly=(state,))


def cast_state(
    builder: ProgramBuilder,
    kernels: KernelCache,
    output: Buffer,
    state: Buffer,
    specialization: CastStateSpecialization,
) -> None:
    """Append one FP32 -> FP16 copy of the state."""
    builder.set_target(lc0ex_pb2.Target.VENDOR_NVIDIA, f"sm_{specialization.architecture}")
    kernel = kernels.get(compile_cast_state, specialization)
    builder.call(kernel, output, state, readonly=(state,))
