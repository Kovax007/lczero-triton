"""EGT2 triplet operator at the edge update sites (item E, K4; round 22 K4b: the fused forms).

The lab's `triplet_path` and `triplet_ag` 512x15 exports add one branch to each edge update site, between K3's
readback and K3's edge FFN. Per sample, with ``c`` an edge channel, ``h`` one of 4 triplet heads, ``d`` one of 4 dot
channels and ``i``/``j``/``k`` squares::

    n        = rms(e_hat)                                   rms(x) = x / sqrt(mean_c x^2 + 1e-6)
    v        = W_v n            [32, 64, 64]                /encoder{b}/edge_site/triplet/value/w  [32, 16]
    eg       = W_g n + b_g      [16, 64, 64]                /encoder{b}/edge_site/triplet/gate/w   [16, 16]
                                                            /encoder{b}/edge_site/triplet/gate/b   [16]
    e_in, g_in, e_out, g_out = split(eg, 4)                 4 x [4, 64, 64], in that order
    a_in     = softmax_k(e_in[h, i, k]) * sigmoid(g_in)     softmax over the LAST axis
    a_out    = softmax_k(e_out[h, k, i]) * sigmoid(g_out)   softmax over the ROW axis
    v_in, v_out = v[:16], v[16:]        each [4, 4, 64, 64] = [h, d, i, j]
    va_in    = sum_k a_in[h, i, k] v_in[h, d, k, j]         `path`; `ag` reads v_in[h, d, j, k]
    va_out   = sum_k a_out[h, k, i] v_out[h, d, j, k]       `path`; `ag` reads v_out[h, d, k, j]
    e_hat2   = e_hat + W_o (va_in | va_out)                 /encoder{b}/edge_site/triplet/out/w    [16, 32]

Both contraction orders are one flag, not two kernels: every case is ``A[i, k] V[k, j]`` once the operands are read
in the right order. Loading the outward branch's ``e_out``/``g_out`` tiles transposed turns its row softmax into the
same last-axis softmax and its ``a_out[h, k, i]`` into ``A[i, k]``, so only ``V`` still swaps, on
``(direction + ag) mod 2``: `path` swaps outward, `ag` swaps inward.

The triplet exports have no gate cap (that is `attention_egt`'s `cap` flag, in the logit gate, not here), and the
softmaxes and the sigmoid are FP32, as training kept them.

Forms (round 22 K4b)
--------------------
``TripletSiteSpecialization.form`` selects how the operator is staged; ``buffer_bytes`` sizes what each form needs.

* ``"scratch"`` (K4, as served in round 21): ``prep`` -> ``contract`` -> ``out``. ``prep`` writes ``v`` (33.5 MB at
  batch 64) and ``eg`` (16.8 MB) to scratch, ``contract`` reads them back and overwrites ``v`` with ``va`` in place,
  ``out`` adds ``W_o va`` into e_hat in place. About 235 MB of traffic per site at batch 64, most of it DRAM.
  ``triplet_site_with_readback`` (K4b step a) replaces K3's separate ``edge_site`` stage "readback" and ``prep`` by
  one planar stage, ``readback_prep``: it computes ``e_hat = e + O_e H`` with K3's own dot, stores it, and goes
  straight into rms -> ``v``, ``eg`` without re-reading e_hat. One launch and one 16.8 MB state read fewer per site;
  e_hat and e_hat2 are bit-identical to the split chain (the dots are the same expressions).
* ``"fused"`` (K4b step b): ``fused`` -> ``out``, no scratch. One program per (sample, direction, head, group of
  ``group`` dot channels) reads the sample's 16 planes once, row-major, and because every map is linear in the
  state, ``W rms(x) = inv * (W x)``: it accumulates the sum of squares, the head's logit and door maps and its value
  maps as [64, 64] FMA accumulators over the 16 channels, scales by ``inv = 1 / sqrt(mean + eps)``, builds ``A``
  (softmax x sigmoid, FP32) and runs the ``A V[d]`` contractions as separate ``tl.dot`` calls, writing only ``va``.
  The outward branch's transposed reads (and ``V``'s swap) are ``tl.trans`` on the accumulated tiles, selected
  by the program's direction. The seven FP32 accumulators are the register floor: 215-255 registers per thread at
  8 warps whatever the organisation (a constexpr direction with static transposed loads, a head pair or both
  directions per program, 4 or 16 warps all measured equal or worse -- K4b report). Two dot classes: ``dot="ieee"``
  (K4's class, FP32 FMA -- which SPILLS at this register floor and loses to the scratch chain) and ``dot="fp16"``
  -- ``A`` and ``V`` rounded once to FP16 and contracted on the tensor cores with FP32 accumulation, the class of
  K2's q.k and of K2c's FP16 state copy, spill-free and 2x faster: gate it at the served rule (5e-3 vs ORT, then
  the end-to-end KL/top-1), never at K4's 1e-5. The softmax, the sigmoid and the logit/door maps stay FP32 in
  both. ``state_f16`` reads e_hat from an FP16 copy (131 KB per sample instead of 262), which ``triplet_readback``
  writes next to the FP32 e_hat; the same served class. Traffic per sample per site: 8 programs x 131-262 KB read
  (L2-served: the programs of a sample share the state) + 512 KB of ``va`` written. ``out`` is unchanged and
  reads ``va`` from its own buffer (``buffer_bytes()["va"]``).
* ``out_ffn`` (K4b step c, `triplet_out_ffn`): K3's stage "ffn" with ``out`` folded in -- one planar program
  computes ``e_hat2 = e_hat + W_o va`` (K4's dot, IEEE) and goes straight into K3's FFN and rms, writing e' to the
  site's output buffer; e_hat2 never reaches memory (-16.8 MB written, -16.8 MB read and one launch per site at
  batch 64). e_hat2 itself is the same dot expression as `out`'s; e' is in class with K3's stage (the FFN dots run
  on the dot-output layout, K4b (a)'s reduction-order effect).

``tl.dot`` defaults to TF32 on FP32 inputs in this Triton (``default_dot_input_precision = "tf32"``), which puts a
~9e-4 floor under every FP32-class gate; every dot here pins ``input_precision="ieee"`` (K2 and K3 both hit this).
In-place stages (``contract`` on ``values``, ``out`` on the state) are autotuned with ``restore_value``: without it
every candidate after the first contracts an already-contracted tensor or adds the residual again, and the FIRST
run of a specialization (a build) is wrong while every later one is right (K4 report section 3). ``readback_prep``
and ``fused`` write only buffers they do not read, so their replays are idempotent.

Layouts (fixed interfaces)
--------------------------
* ``state`` (e_hat, updated in place): FP32 ``[B, 16, 64, 64]``, index ``sample * 16 * 4096 + c * 4096 + 64 i + j``
  (K1's e0 layout, K3's site layout).
* ``values`` (v, then va in place; form "scratch"): FP32 ``[B, 32, 64, 64]``, channel ``direction * 16 + h * 4 + d``:
  ``v_in``'s 16 channels first, then ``v_out``'s, which is both the export's split order and its concatenation order.
  ``va`` (form "fused") has the same shape and channel order in its own buffer.
* ``gates`` (eg; form "scratch"): FP32 ``[B, 16, 64, 64]``, channel ``direction * 8 + h`` for the logits and
  ``direction * 8 + 4 + h`` for the doors -- the export's split order ``e_in, g_in, e_out, g_out``.
* ``logits`` (K2's H, ``readback_prep`` only): FP32 ``[B * 32, 64, 64]``, index ``(sample * 32 + head) * 4096 + cell``.
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
CELLS = 64 * 64
STATES = 16
TRIPLET_HEADS = 4
TRIPLET_DOTS = 4
DIRECTIONS = 2
# The block's attention heads, K2's H layout: what the readback (K3, and `readback_prep` here) contracts.
SITE_HEADS = 32
# The site FFN's hidden width (K3's `HIDDEN`), for `out_ffn`.
SITE_HIDDEN = 64
# K3's FFN tables, the site tables `out_ffn` reads after the readback one (`edge_site.SITE_TABLES[1:]`).
FFN_TABLES = ("ffn/dense1/w", "ffn/dense1/b", "ffn/dense2/w")
# `const_1343` / `const_1350` at the site after block 3 and their twins at sites 7 and 11 (the same rms epsilon K3
# and the prologue use).
EPSILON = 1e-6
SITE_BLOCKS = (3, 7, 11)
# The plan names each stage reads, below `/encoder{b}/edge_site/` (`lab._names._plan_egt_site`), in argument order.
TRIPLET_TABLES = ("triplet/value/w", "triplet/gate/w", "triplet/gate/b", "triplet/out/w")
# K3's readback table, the one site table `readback_prep` reads (`edge_site.SITE_TABLES[0]`).
READBACK_TABLE = "readback/w"

Contraction = Literal["path", "ag"]
_CONTRACTIONS: dict[str, int] = {"path": 0, "ag": 1}
Form = Literal["scratch", "fused"]
_FORMS = ("scratch", "fused")
Dot = Literal["ieee", "fp16"]
_DOTS: dict[str, int] = {"ieee": 0, "fp16": 1}
# (pixels per program, num_warps) for the planar stages: K3's finalists plus the 1-warp runs the K4
# microbenchmark preferred (batch 64: prep 35.3 us at 64 cells x 1 warp against 43.8 at 2 warps).
_PLANAR_CONFIGURATIONS = ((32, 1), (32, 2), (64, 1), (64, 2), (64, 4))
# (dot channels per program, num_warps) for the contraction. One dot channel per program wins by a wide margin,
# measured at batch 64: 82 us at `fused` = 1, against 856 at 2 and 1,058 at 4 (their best warp counts), because a
# fused `tl.dot` of [64, 64] x [64, 128] or x [64, 256] is exactly the large float block K1 and K3 both found
# pathological. `fused` = 2 and 4 are excluded from autotune: they never win and cost 15-77 s each to compile.
_CONTRACT_CONFIGURATIONS = ((1, 4), (1, 8), (1, 16), (1, 32))
# (dot channels per program, num_warps) for the fused stage: the two configurations that do not spill in the K4b
# sweep (batch 64, FP16 dots: 4 x 8 warps 77 us, 2 x 4 warps 75 us; 16 warps, 4 warps with 4 channels, a head
# pair or both directions per program spill 8-25x and are not offered). Two channels per program halves the live
# value tiles at the price of a second state read and a second logit/door pass.
_FUSED_CONFIGURATIONS = ((4, 8), (2, 4))


def triplet_table_names(after_block: int) -> tuple[str, ...]:
    """Return the four plan names one site's triplet reads, in kernel argument order."""
    return tuple(f"/encoder{after_block}/edge_site/{table}" for table in TRIPLET_TABLES)


def ffn_table_names(after_block: int) -> tuple[str, ...]:
    """Return the plan names of the site's three FFN tables (K3's), which `out_ffn` reads, in argument order."""
    return tuple(f"/encoder{after_block}/edge_site/{table}" for table in FFN_TABLES)


def readback_table_name(after_block: int) -> str:
    """Return the plan name of the site's readback table (K3's), which `readback_prep` reads."""
    return f"/encoder{after_block}/edge_site/{READBACK_TABLE}"


@triton.jit
def _rms_columns(values, states: tl.constexpr, epsilon: tl.constexpr):
    """``x / sqrt(mean_c x^2 + eps)`` over axis 0 of a ``[states, cells]`` tile (the export's Mul/ReduceMean/Div)."""
    mean_square = tl.sum(values * values, axis=0) * (1.0 / states)
    return values * (1.0 / tl.sqrt(mean_square + epsilon))[None, :]


@triton.jit
def _triplet_prep_body(  # noqa: PLR0913
    values,
    gates,
    state,
    value_weight,
    gate_weight,
    gate_bias,
    batch_count: tl.constexpr,  # noqa: ARG001  # Autotune key and launch grid.
    states: tl.constexpr,
    epsilon: tl.constexpr,
    pixels: tl.constexpr,
) -> None:
    """Write ``v = W_v rms(e_hat)`` and ``eg = W_g rms(e_hat) + b_g`` for one run of ``pixels`` cells."""
    program = tl.program_id(0)
    runs: tl.constexpr = 4096 // pixels
    sample = program // runs
    cells = (program % runs) * pixels + tl.arange(0, pixels)
    channels = tl.arange(0, states)
    wide = tl.arange(0, 2 * states)
    planes = channels[:, None] * 4096 + cells[None, :]
    normed = _rms_columns(tl.load(state + sample * states * 4096 + planes), states, epsilon)
    value = tl.dot(tl.load(value_weight + wide[:, None] * states + channels[None, :]), normed,
                   input_precision="ieee")
    tl.store(values + sample * 2 * states * 4096 + wide[:, None] * 4096 + cells[None, :], value)
    gate = tl.dot(tl.load(gate_weight + channels[:, None] * states + channels[None, :]), normed,
                  input_precision="ieee")
    tl.store(gates + sample * states * 4096 + planes, gate + tl.load(gate_bias + channels)[:, None])


@triton.jit
def _triplet_readback_prep_body(  # noqa: PLR0913
    values,
    gates,
    output,
    state,
    logits,
    readback_weight,
    value_weight,
    gate_weight,
    gate_bias,
    batch_count: tl.constexpr,  # noqa: ARG001  # Autotune key and launch grid.
    site_heads: tl.constexpr,
    states: tl.constexpr,
    epsilon: tl.constexpr,
    pixels: tl.constexpr,
    site_heads_pad: tl.constexpr = 0,
) -> None:
    """K4b (a): ``e_hat = e + O_e H`` (K3's readback dot) into ``output``, then ``prep`` from the same registers.

    One run of ``pixels`` cells of one sample. ``output`` is the buffer K3's stage "readback" would have written; the
    e_hat tile never leaves the program between the readback and the rms.
    """
    program = tl.program_id(0)
    runs: tl.constexpr = 4096 // pixels
    sample = program // runs
    cells = (program % runs) * pixels + tl.arange(0, pixels)
    channels = tl.arange(0, states)
    wide = tl.arange(0, 2 * states)
    planes = channels[:, None] * 4096 + cells[None, :]
    current = tl.load(state + sample * states * 4096 + planes)  # e, [states, pixels]
    if site_heads_pad == 0:
        head_index = tl.arange(0, site_heads)
        logit_values = tl.load(logits + (sample * site_heads + head_index[:, None]) * 4096 + cells[None, :])
        readback_values = tl.load(readback_weight + channels[:, None] * site_heads + head_index[None, :])
    else:
        # A head count that is not a power of two: the dot's K axis padded, the missing heads read as zero.
        head_index = tl.arange(0, site_heads_pad)
        live = head_index < site_heads
        logit_values = tl.load(logits + (sample * site_heads + head_index[:, None]) * 4096 + cells[None, :],
                               mask=live[:, None], other=0.0)
        readback_values = tl.load(readback_weight + channels[:, None] * site_heads + head_index[None, :],
                                  mask=live[None, :], other=0.0)
    e_hat = tl.dot(readback_values, logit_values, current, input_precision="ieee")  # K3's expression, verbatim
    tl.store(output + sample * states * 4096 + planes, e_hat)
    normed = _rms_columns(e_hat, states, epsilon)
    value = tl.dot(tl.load(value_weight + wide[:, None] * states + channels[None, :]), normed,
                   input_precision="ieee")
    tl.store(values + sample * 2 * states * 4096 + wide[:, None] * 4096 + cells[None, :], value)
    gate = tl.dot(tl.load(gate_weight + channels[:, None] * states + channels[None, :]), normed,
                  input_precision="ieee")
    tl.store(gates + sample * states * 4096 + planes, gate + tl.load(gate_bias + channels)[:, None])


@triton.jit
def _triplet_contract_body(  # noqa: PLR0913
    values,
    gates,
    batch_count: tl.constexpr,  # noqa: ARG001  # Autotune key and launch grid.
    states: tl.constexpr,
    heads: tl.constexpr,
    dots: tl.constexpr,
    contraction: tl.constexpr,
    fused: tl.constexpr,
) -> None:
    """Replace ``fused`` value channels of one (sample, direction, head) with their contracted ``va`` tiles."""
    groups: tl.constexpr = dots // fused
    # The 2 is `DIRECTIONS`, inlined: jit code cannot read a module global (K2's trap 5).
    per_sample: tl.constexpr = 2 * heads * groups
    program = tl.program_id(0)
    sample = program // per_sample
    within = program % per_sample
    direction = within // (heads * groups)
    head = (within // groups) % heads
    group = within % groups

    rows = tl.arange(0, 64)
    columns = tl.arange(0, 64)
    # The inward branch reads its tiles row-major; the outward branch reads them transposed, which turns its row
    # softmax into the same last-axis softmax and `a_out[h, k, i]` into `A[i, k]`.
    row_major = rows[:, None] * 64 + columns[None, :]
    column_major = columns[None, :] * 64 + rows[:, None]
    cell = row_major * (1 - direction) + column_major * direction
    gate_base = sample * states * 4096 + direction * 2 * heads * 4096
    logit = tl.load(gates + gate_base + head * 4096 + cell)
    door = tl.load(gates + gate_base + (heads + head) * 4096 + cell)
    exponentials = tl.exp(logit - tl.max(logit, axis=1)[:, None])
    weights = exponentials / tl.sum(exponentials, axis=1)[:, None]
    weights = weights * (1.0 / (1.0 + tl.exp(-door)))

    lanes = tl.arange(0, fused * 64)
    lane_dot = lanes // 64
    lane_column = lanes % 64
    value_base = (sample * 2 * states + (direction * heads + head) * dots + group * fused) * 4096
    planes = lane_dot[None, :] * 4096
    straight = rows[:, None] * 64 + lane_column[None, :]
    swapped = lane_column[None, :] * 64 + rows[:, None]
    # `path` swaps the outward branch's values, `ag` the inward one's; every other case reads them as stored.
    swap = (direction + contraction) % 2
    source = tl.load(values + value_base + planes + straight * (1 - swap) + swapped * swap)
    # In place: this program is the only one that touches these channels, and it has read them all by now.
    tl.store(values + value_base + planes + straight,
             tl.dot(weights, source, input_precision="ieee"))


@triton.jit
def _triplet_readback_body(  # noqa: PLR0913
    output,
    copy,
    state,
    logits,
    readback_weight,
    batch_count: tl.constexpr,  # noqa: ARG001  # Autotune key and launch grid.
    site_heads: tl.constexpr,
    states: tl.constexpr,
    pixels: tl.constexpr,
    site_heads_pad: tl.constexpr = 0,
) -> None:
    """K3's readback (``e_hat = e + O_e H``, the same expression) into ``output``, plus an FP16 copy for `fused`."""
    program = tl.program_id(0)
    runs: tl.constexpr = 4096 // pixels
    sample = program // runs
    cells = (program % runs) * pixels + tl.arange(0, pixels)
    channels = tl.arange(0, states)
    planes = channels[:, None] * 4096 + cells[None, :]
    current = tl.load(state + sample * states * 4096 + planes)
    if site_heads_pad == 0:
        head_index = tl.arange(0, site_heads)
        logit_values = tl.load(logits + (sample * site_heads + head_index[:, None]) * 4096 + cells[None, :])
        readback_values = tl.load(readback_weight + channels[:, None] * site_heads + head_index[None, :])
    else:
        # A head count that is not a power of two: the dot's K axis padded, the missing heads read as zero.
        head_index = tl.arange(0, site_heads_pad)
        live = head_index < site_heads
        logit_values = tl.load(logits + (sample * site_heads + head_index[:, None]) * 4096 + cells[None, :],
                               mask=live[:, None], other=0.0)
        readback_values = tl.load(readback_weight + channels[:, None] * site_heads + head_index[None, :],
                                  mask=live[None, :], other=0.0)
    e_hat = tl.dot(readback_values, logit_values, current, input_precision="ieee")
    tl.store(output + sample * states * 4096 + planes, e_hat)
    tl.store(copy + sample * states * 4096 + planes, e_hat.to(tl.float16))


@triton.jit
def _store_contraction(base, d, weights, value, swap, tile, dot: tl.constexpr) -> None:
    """``va[d] = A V[d]``, ``V`` read transposed when the contraction order says so.

    ``dot`` = 0: FP32 FMA in IEEE (K4's class); 1: FP16 operands on the tensor cores, FP32 accumulation.
    """
    value = tl.where(swap, tl.trans(value), value)
    if dot == 1:
        tl.store(base + d * 4096 + tile, tl.dot(weights.to(tl.float16), value.to(tl.float16)))
    else:
        tl.store(base + d * 4096 + tile, tl.dot(weights, value, input_precision="ieee"))


@triton.jit
def _triplet_fused_body(  # noqa: PLR0913
    va,
    state,
    value_weight,
    gate_weight,
    gate_bias,
    batch_count: tl.constexpr,  # noqa: ARG001  # Autotune key and launch grid.
    states: tl.constexpr,
    heads: tl.constexpr,
    dots: tl.constexpr,
    contraction: tl.constexpr,
    epsilon: tl.constexpr,
    dot: tl.constexpr,
    group: tl.constexpr,
) -> None:
    """K4b (b): ``va`` for ``group`` dot channels of one (sample, direction, head), from one read of the state."""
    groups: tl.constexpr = dots // group
    # The 2 is `DIRECTIONS`, inlined: jit code cannot read a module global (K2's trap 5).
    per_sample: tl.constexpr = 2 * heads * groups
    program = tl.program_id(0)
    sample = program // per_sample
    within = program % per_sample
    direction = within // (heads * groups)
    head = (within // groups) % heads
    chunk = within % groups
    rows = tl.arange(0, 64)
    columns = tl.arange(0, 64)
    tile = rows[:, None] * 64 + columns[None, :]
    plane = state + sample * states * 4096 + tile
    # The export's split order: logits at channel direction * 8 + h, doors at direction * 8 + 4 + h of `eg`; the
    # value rows at direction * 16 + h * 4 + d of W_v.
    gate_row = direction * 2 * heads + head
    door_row = gate_row + heads
    value_row = (direction * heads + head) * dots + chunk * group
    square = tl.zeros((64, 64), dtype=tl.float32)
    logit = tl.zeros((64, 64), dtype=tl.float32)
    door = tl.zeros((64, 64), dtype=tl.float32)
    v0 = tl.zeros((64, 64), dtype=tl.float32)
    v1 = tl.zeros((64, 64), dtype=tl.float32)
    v2 = tl.zeros((64, 64), dtype=tl.float32)
    v3 = tl.zeros((64, 64), dtype=tl.float32)
    for c in tl.static_range(states):
        x = tl.load(plane + c * 4096).to(tl.float32)
        square += x * x
        logit += tl.load(gate_weight + gate_row * states + c) * x
        door += tl.load(gate_weight + door_row * states + c) * x
        v0 += tl.load(value_weight + value_row * states + c) * x
        v1 += tl.load(value_weight + (value_row + 1) * states + c) * x
        if group > 2:
            v2 += tl.load(value_weight + (value_row + 2) * states + c) * x
            v3 += tl.load(value_weight + (value_row + 3) * states + c) * x
    # The maps are linear in the state, so the rms scale applies after the channel sum: W rms(x) = inv (W x).
    inverse = 1.0 / tl.sqrt(square * (1.0 / states) + epsilon)
    logit = logit * inverse + tl.load(gate_bias + gate_row)
    door = door * inverse + tl.load(gate_bias + door_row)
    # The outward branch's tiles are read transposed (K4 loads them column-major), which turns its row softmax
    # into the same last-axis softmax and `a_out[h, k, i]` into `A[i, k]`.
    transposed = direction == 1
    logit = tl.where(transposed, tl.trans(logit), logit)
    door = tl.where(transposed, tl.trans(door), door)
    exponentials = tl.exp(logit - tl.max(logit, axis=1)[:, None])
    weights = exponentials / tl.sum(exponentials, axis=1)[:, None]
    weights = weights * (1.0 / (1.0 + tl.exp(-door)))
    # `path` swaps the outward branch's values, `ag` the inward one's; every other case reads them as stored.
    swap = ((direction + contraction) % 2) == 1
    base = va + (sample * 2 * states + value_row) * 4096
    _store_contraction(base, 0, weights, v0 * inverse, swap, tile, dot)
    _store_contraction(base, 1, weights, v1 * inverse, swap, tile, dot)
    if group > 2:
        _store_contraction(base, 2, weights, v2 * inverse, swap, tile, dot)
        _store_contraction(base, 3, weights, v3 * inverse, swap, tile, dot)


@triton.jit
def _triplet_out_body(  # noqa: PLR0913
    state,
    values,
    output_weight,
    batch_count: tl.constexpr,  # noqa: ARG001  # Autotune key and launch grid.
    states: tl.constexpr,
    pixels: tl.constexpr,
) -> None:
    """Add ``W_o va`` into e_hat in place for one run of ``pixels`` cells, giving e_hat2."""
    program = tl.program_id(0)
    runs: tl.constexpr = 4096 // pixels
    sample = program // runs
    cells = (program % runs) * pixels + tl.arange(0, pixels)
    channels = tl.arange(0, states)
    wide = tl.arange(0, 2 * states)
    planes = channels[:, None] * 4096 + cells[None, :]
    # FP32 va, or FP16 (`va_f16`: `triplet_mma`'s FP16 store) widened here; the dot stays IEEE FP32
    branch = tl.load(values + sample * 2 * states * 4096 + wide[:, None] * 4096 + cells[None, :]).to(tl.float32)
    weight = tl.load(output_weight + channels[:, None] * 2 * states + wide[None, :])
    current = tl.load(state + sample * states * 4096 + planes)
    tl.store(state + sample * states * 4096 + planes,
             tl.dot(weight, branch, current, input_precision="ieee"))


@triton.jit
def _triplet_out_ffn_body(  # noqa: PLR0913
    output,
    state,
    values,
    output_weight,
    dense1_weight,
    dense1_bias,
    dense2_weight,
    batch_count: tl.constexpr,  # noqa: ARG001  # Autotune key and launch grid.
    states: tl.constexpr,
    hidden: tl.constexpr,
    epsilon: tl.constexpr,
    pixels: tl.constexpr,
) -> None:
    """K4b (c): ``e' = rms(e_hat2 + W2 relu(W1 rms(e_hat2) + b1))`` with ``e_hat2 = e_hat + W_o va`` in registers.

    One run of ``pixels`` cells of one sample; K3's stage "ffn" body with `out`'s dot ahead of it. ``output`` is
    the site's e' buffer (the buffer e came from), ``state`` e_hat, both K3's layouts.
    """
    program = tl.program_id(0)
    runs: tl.constexpr = 4096 // pixels
    sample = program // runs
    cells = (program % runs) * pixels + tl.arange(0, pixels)
    channels = tl.arange(0, states)
    wide = tl.arange(0, 2 * states)
    planes = channels[:, None] * 4096 + cells[None, :]
    branch = tl.load(values + sample * 2 * states * 4096 + wide[:, None] * 4096 + cells[None, :])
    weight = tl.load(output_weight + channels[:, None] * 2 * states + wide[None, :])
    current = tl.load(state + sample * states * 4096 + planes)
    values_ = tl.dot(weight, branch, current, input_precision="ieee")  # e_hat2, K4's `out` expression
    units = tl.arange(0, hidden)
    weight1 = tl.load(dense1_weight + units[:, None] * states + channels[None, :])
    bias1 = tl.load(dense1_bias + units)
    weight2 = tl.load(dense2_weight + channels[:, None] * hidden + units[None, :])
    activation = tl.dot(weight1, _rms_columns(values_, states, epsilon), input_precision="ieee")
    activation = tl.maximum(activation + bias1[:, None], 0.0)
    values_ = tl.dot(weight2, activation, values_, input_precision="ieee")  # e_til
    tl.store(output + sample * states * 4096 + planes, _rms_columns(values_, states, epsilon))


_triplet_prep_kernel = triton.autotune(
    configs=[triton.Config({"pixels": pixels}, num_warps=warps) for pixels, warps in _PLANAR_CONFIGURATIONS],
    key=["batch_count", "states"],
    cache_results=True,
)(_triplet_prep_body)

_triplet_readback_prep_kernel = triton.autotune(
    configs=[triton.Config({"pixels": pixels}, num_warps=warps) for pixels, warps in _PLANAR_CONFIGURATIONS],
    key=["batch_count", "site_heads", "states"],
    cache_results=True,
    # Not in place: it reads `state` (e) and `logits`, writes `output` (e_hat), `values` and `gates`.
)(_triplet_readback_prep_body)

_triplet_contract_kernel = triton.autotune(
    configs=[triton.Config({"fused": fused}, num_warps=warps) for fused, warps in _CONTRACT_CONFIGURATIONS],
    key=["batch_count", "states", "heads", "dots", "contraction"],
    cache_results=True,
    # In place: this stage replaces `values` with `va`, so every autotune candidate after the first would contract
    # an already-contracted tensor. Without this the FIRST run of a specialization is wrong and every later run is
    # right, because autotuning only happens once -- see `policy_egt_bias` for the same guard.
    restore_value=["values"],
)(_triplet_contract_body)

_triplet_readback_kernel = triton.autotune(
    configs=[triton.Config({"pixels": pixels}, num_warps=warps) for pixels, warps in _PLANAR_CONFIGURATIONS],
    key=["batch_count", "site_heads", "states"],
    cache_results=True,
    # Not in place: it reads `state` (e) and `logits`, writes `output` (e_hat) and its FP16 `copy`.
)(_triplet_readback_body)

_triplet_fused_kernel = triton.autotune(
    configs=[triton.Config({"group": group}, num_warps=warps) for group, warps in _FUSED_CONFIGURATIONS],
    key=["batch_count", "states", "heads", "dots", "contraction", "dot"],
    cache_results=True,
    # Not in place: it reads `state` and writes only `va`, so a replay rewrites the same values.
)(_triplet_fused_body)

# K3's site configurations: the FFN's [64, pixels] hidden tile is what bounds this planar stage (K3 report).
_OUT_FFN_CONFIGURATIONS = ((32, 2), (32, 4), (64, 2), (64, 4))

_triplet_out_ffn_kernel = triton.autotune(
    configs=[triton.Config({"pixels": pixels}, num_warps=warps) for pixels, warps in _OUT_FFN_CONFIGURATIONS],
    key=["batch_count", "states", "hidden"],
    cache_results=True,
    # Not in place: it reads `state` (e_hat) and `values` (va) and writes `output` (e').
)(_triplet_out_ffn_body)

_triplet_out_kernel = triton.autotune(
    configs=[triton.Config({"pixels": pixels}, num_warps=warps) for pixels, warps in _PLANAR_CONFIGURATIONS],
    key=["batch_count", "states"],
    cache_results=True,
    # In place: this stage adds the branch into e_hat, so an unrestored candidate adds it again.
    restore_value=["state"],
)(_triplet_out_body)


@dataclass(frozen=True, slots=True)
class TripletSiteSpecialization:
    """Immutable triplet specialization (the exports' widths by default; `contraction` comes from the reader).

    `form` picks the staging ("scratch": K4's prep/contract/out with two scratch buffers; "fused": K4b's fused
    prep+contract writing only `va`, then out). `site_heads` is the block's head count, read by `readback_prep`.
    """

    batch_count: int
    architecture: int
    contraction: Contraction = "path"
    states: int = STATES
    heads: int = TRIPLET_HEADS
    dots: int = TRIPLET_DOTS
    form: Form = "scratch"
    site_heads: int = SITE_HEADS
    # Form "fused" only: the contraction's operand class, and whether e_hat is read from an FP16 copy.
    dot: Dot = "ieee"
    state_f16: bool = False
    # The site FFN's hidden width (K3's `hidden`), read by `out_ffn` (K4b c) only.
    site_hidden: int = SITE_HIDDEN
    # `out` reads va as FP16 (`triplet_mma` with `va_f16`); form "fused" only.
    va_f16: bool = False

    def __post_init__(self) -> None:
        """Reject widths a ``tl.arange`` tile cannot take, and a value width the two directions cannot fill."""
        if not 0 < self.site_heads <= 64:  # noqa: PLR2004
            message = f"TripletSiteSpecialization.site_heads={self.site_heads} must be in 1..64"
            raise ValueError(message)
        for name in ("states", "heads", "dots", "site_hidden"):
            width = getattr(self, name)
            if width <= 0 or width & (width - 1):
                message = f"TripletSiteSpecialization.{name}={width} must be a power of two"
                raise ValueError(message)
        if self.heads * self.dots != self.states:
            message = (f"TripletSiteSpecialization: heads={self.heads} x dots={self.dots} must equal "
                       f"states={self.states}; the two directions fill the 2 x states value channels")
            raise ValueError(message)
        if self.contraction not in _CONTRACTIONS:
            message = f"unknown contraction {self.contraction!r}; expected one of {sorted(_CONTRACTIONS)}"
            raise ValueError(message)
        if self.form not in _FORMS:
            message = f"unknown form {self.form!r}; expected one of {_FORMS}"
            raise ValueError(message)
        if self.form == "fused" and self.dots != TRIPLET_DOTS:
            message = f"form 'fused' is written for dots={TRIPLET_DOTS} (four value accumulators per program)"
            raise ValueError(message)
        if self.dot not in _DOTS:
            message = f"unknown dot {self.dot!r}; expected one of {sorted(_DOTS)}"
            raise ValueError(message)
        if self.va_f16 and self.form != "fused":
            message = "va_f16 (FP16 va from triplet_mma) belongs to form 'fused'"
            raise ValueError(message)
        if self.form != "fused" and (self.dot != "ieee" or self.state_f16):
            message = "dot='fp16' and state_f16 belong to form 'fused'"
            raise ValueError(message)



def buffer_bytes(specialization: TripletSiteSpecialization) -> dict[str, int]:
    """Return the buffers the calls touch, in bytes.

    Form "scratch": `values` and `gates` are scratch and ``va`` reuses `values` (0 bytes of its own). Form "fused":
    neither scratch buffer exists and ``va`` has its own buffer of `values`' size.
    """
    cells = specialization.batch_count * CELLS
    state = 4 * specialization.states * cells
    wide = 4 * 2 * specialization.states * cells
    if specialization.form == "fused":
        copy = 2 * specialization.states * cells if specialization.state_f16 else 0
        return {"state": state, "values": 0, "gates": 0, "va": wide, "copy": copy}
    return {"state": state, "values": wide, "gates": 4 * specialization.states * cells, "va": 0, "copy": 0}


def _planar_grid(configuration: Mapping[str, object]) -> tuple[int]:
    """Return one program per (sample, run of cells)."""
    return (cast("int", configuration["batch_count"]) * CELLS // cast("int", configuration["pixels"]),)


def _contract_grid(configuration: Mapping[str, object]) -> tuple[int]:
    """Return one program per (sample, direction, head, group of fused dot channels)."""
    batch = cast("int", configuration["batch_count"])
    heads = cast("int", configuration["heads"])
    dots = cast("int", configuration["dots"])
    return (batch * DIRECTIONS * heads * (dots // cast("int", configuration["fused"])),)


def _fused_programs(batch: int, heads: int, dots: int, group: int) -> int:
    """Programs of the fused stage: one per (sample, direction, head, group of `group` dot channels)."""
    return batch * DIRECTIONS * heads * (dots // group)


def _fused_grid(configuration: Mapping[str, object]) -> tuple[int]:
    """Return the fused stage's grid for one autotune configuration."""
    return (_fused_programs(cast("int", configuration["batch_count"]), cast("int", configuration["heads"]),
                            cast("int", configuration["dots"]), cast("int", configuration["group"])),)


def _site_heads_pad(site_heads: int) -> int:
    """0 for a power-of-two head count (the unpadded kernel), else the next power of two (at least 16, tl.dot)."""
    if site_heads & (site_heads - 1) == 0:
        return 0
    return max(16, 1 << site_heads.bit_length())


def _inputs(specialization: TripletSiteSpecialization) -> dict[str, torch.Tensor]:
    """Return zeroed tensors of every buffer and table shape, for autotuning and compilation."""
    batch, states, site_heads = specialization.batch_count, specialization.states, specialization.site_heads
    return {
        "state": torch.zeros((batch, states, 64, 64), dtype=torch.float32, device="cuda"),
        "copy": torch.zeros((batch, states, 64, 64), dtype=torch.float16, device="cuda"),
        "output": torch.zeros((batch, states, 64, 64), dtype=torch.float32, device="cuda"),
        "values": torch.zeros((batch, 2 * states, 64, 64), dtype=torch.float32, device="cuda"),
        "gates": torch.zeros((batch, states, 64, 64), dtype=torch.float32, device="cuda"),
        "logits": torch.zeros((batch * site_heads, 64, 64), dtype=torch.float32, device="cuda"),
        "readback_weight": torch.zeros((states, site_heads), dtype=torch.float32, device="cuda"),
        "value_weight": torch.zeros((2 * states, states), dtype=torch.float32, device="cuda"),
        "gate_weight": torch.zeros((states, states), dtype=torch.float32, device="cuda"),
        "gate_bias": torch.zeros(states, dtype=torch.float32, device="cuda"),
        "output_weight": torch.zeros((states, 2 * states), dtype=torch.float32, device="cuda"),
        "dense1_weight": torch.zeros((specialization.site_hidden, states), dtype=torch.float32, device="cuda"),
        "dense1_bias": torch.zeros(specialization.site_hidden, dtype=torch.float32, device="cuda"),
        "dense2_weight": torch.zeros((states, specialization.site_hidden), dtype=torch.float32, device="cuda"),
    }


def compile_triplet_prep(specialization: TripletSiteSpecialization) -> KernelArtifact:
    """Autotune and compile the rms and the two projections (one artifact per `KernelCache.get`)."""
    tensors = _inputs(specialization)
    compiled = _triplet_prep_kernel[_planar_grid](
        tensors["values"], tensors["gates"], tensors["state"], tensors["value_weight"], tensors["gate_weight"],
        tensors["gate_bias"], specialization.batch_count, specialization.states, EPSILON,
    )
    pixels = cast("int", _triplet_prep_kernel.best_config.kwargs["pixels"])
    return artifact_from_triton(
        compiled, grid=(specialization.batch_count * CELLS // pixels, 1, 1), parameters=(_POINTER,) * 6,
        autotuner=_triplet_prep_kernel,
    )


def compile_triplet_readback_prep(specialization: TripletSiteSpecialization) -> KernelArtifact:
    """Autotune and compile K4b (a): K3's readback fused with the rms and the two projections."""
    tensors = _inputs(specialization)
    compiled = _triplet_readback_prep_kernel[_planar_grid](
        tensors["values"], tensors["gates"], tensors["output"], tensors["state"], tensors["logits"],
        tensors["readback_weight"], tensors["value_weight"], tensors["gate_weight"], tensors["gate_bias"],
        specialization.batch_count, specialization.site_heads, specialization.states, EPSILON,
        site_heads_pad=_site_heads_pad(specialization.site_heads),
    )
    pixels = cast("int", _triplet_readback_prep_kernel.best_config.kwargs["pixels"])
    return artifact_from_triton(
        compiled, grid=(specialization.batch_count * CELLS // pixels, 1, 1), parameters=(_POINTER,) * 9,
        autotuner=_triplet_readback_prep_kernel,
    )


def compile_triplet_contract(specialization: TripletSiteSpecialization) -> KernelArtifact:
    """Autotune and compile the softmax-gated contraction."""
    tensors = _inputs(specialization)
    compiled = _triplet_contract_kernel[_contract_grid](
        tensors["values"], tensors["gates"], specialization.batch_count, specialization.states,
        specialization.heads, specialization.dots, _CONTRACTIONS[specialization.contraction],
    )
    fused = cast("int", _triplet_contract_kernel.best_config.kwargs["fused"])
    groups = specialization.dots // fused
    grid = specialization.batch_count * DIRECTIONS * specialization.heads * groups
    return artifact_from_triton(
        compiled, grid=(grid, 1, 1), parameters=(_POINTER,) * 2, autotuner=_triplet_contract_kernel,
    )


def compile_triplet_readback(specialization: TripletSiteSpecialization) -> KernelArtifact:
    """Autotune and compile K3's readback with the FP16 e_hat copy the `state_f16` fused stage reads."""
    tensors = _inputs(specialization)
    compiled = _triplet_readback_kernel[_planar_grid](
        tensors["output"], tensors["copy"], tensors["state"], tensors["logits"], tensors["readback_weight"],
        specialization.batch_count, specialization.site_heads, specialization.states,
        site_heads_pad=_site_heads_pad(specialization.site_heads),
    )
    pixels = cast("int", _triplet_readback_kernel.best_config.kwargs["pixels"])
    return artifact_from_triton(
        compiled, grid=(specialization.batch_count * CELLS // pixels, 1, 1), parameters=(_POINTER,) * 5,
        autotuner=_triplet_readback_kernel,
    )


def launch_triplet_fused(  # noqa: PLR0913
    va: torch.Tensor, state: torch.Tensor, value_weight: torch.Tensor, gate_weight: torch.Tensor,
    gate_bias: torch.Tensor, specialization: TripletSiteSpecialization,
) -> object:
    """Launch the fused stage on torch tensors (tests, benchmarks, `compile_triplet_fused`).

    `state` is e_hat, FP32 or (with `state_f16`) its FP16 copy: the pointer's element type specializes the kernel.
    """
    return _triplet_fused_kernel[_fused_grid](
        va, state, value_weight, gate_weight, gate_bias, specialization.batch_count, specialization.states,
        specialization.heads, specialization.dots, _CONTRACTIONS[specialization.contraction], EPSILON,
        _DOTS[specialization.dot],
    )


def compile_triplet_fused(specialization: TripletSiteSpecialization) -> KernelArtifact:
    """Autotune and compile K4b (b): rms, the head's maps, softmax x sigmoid and the contractions in one program."""
    tensors = _inputs(specialization)
    state = tensors["copy"] if specialization.state_f16 else tensors["state"]
    compiled = launch_triplet_fused(tensors["values"], state, tensors["value_weight"], tensors["gate_weight"],
                                    tensors["gate_bias"], specialization)
    group = cast("int", _triplet_fused_kernel.best_config.kwargs["group"])
    grid = _fused_programs(specialization.batch_count, specialization.heads, specialization.dots, group)
    return artifact_from_triton(compiled, grid=(grid, 1, 1), parameters=(_POINTER,) * 5,
                                autotuner=_triplet_fused_kernel)


def compile_triplet_out(specialization: TripletSiteSpecialization) -> KernelArtifact:
    """Autotune and compile the output projection that adds the branch into e_hat."""
    tensors = _inputs(specialization)
    values = tensors["values"].half() if specialization.va_f16 else tensors["values"]
    compiled = _triplet_out_kernel[_planar_grid](
        tensors["state"], values, tensors["output_weight"], specialization.batch_count,
        specialization.states,
    )
    pixels = cast("int", _triplet_out_kernel.best_config.kwargs["pixels"])
    return artifact_from_triton(
        compiled, grid=(specialization.batch_count * CELLS // pixels, 1, 1), parameters=(_POINTER,) * 3,
        autotuner=_triplet_out_kernel,
    )


def compile_triplet_out_ffn(specialization: TripletSiteSpecialization) -> KernelArtifact:
    """Autotune and compile K4b (c): `out` folded into K3's FFN stage."""
    tensors = _inputs(specialization)
    compiled = _triplet_out_ffn_kernel[_planar_grid](
        tensors["output"], tensors["state"], tensors["values"], tensors["output_weight"], tensors["dense1_weight"],
        tensors["dense1_bias"], tensors["dense2_weight"], specialization.batch_count, specialization.states,
        specialization.site_hidden, EPSILON,
    )
    pixels = cast("int", _triplet_out_ffn_kernel.best_config.kwargs["pixels"])
    return artifact_from_triton(
        compiled, grid=(specialization.batch_count * CELLS // pixels, 1, 1), parameters=(_POINTER,) * 7,
        autotuner=_triplet_out_ffn_kernel,
    )


def triplet_out_ffn(  # noqa: PLR0913
    builder: ProgramBuilder,
    kernels: KernelCache,
    output: Buffer,
    state: Buffer,
    va: Buffer,
    tables: Mapping[str, Buffer],
    ffn_tables: Mapping[str, Buffer],
    specialization: TripletSiteSpecialization,
    *,
    after_block: int,
) -> None:
    """K4b (c): append `out` folded into K3's stage "ffn": ``output`` = e' from ``state`` (e_hat) and ``va``.

    Replaces `out` (inside `triplet_site`) and `edge_site` stage "ffn" together: call `triplet_site` with
    ``out=False`` first, then this. `tables` are the triplet's plans, `ffn_tables` the site's (`ffn_table_names`).
    """
    if output is state or output is va or state is va:
        message = "out_ffn needs distinct buffers: e_hat, va and the e' output"
        raise ValueError(message)
    _set_target(builder, specialization)
    output_weight = tables[triplet_table_names(after_block)[3]]
    dense1_weight, dense1_bias, dense2_weight = (ffn_tables[name] for name in ffn_table_names(after_block))
    builder.call(kernels.get(compile_triplet_out_ffn, specialization), output, state, va, output_weight,
                 dense1_weight, dense1_bias, dense2_weight,
                 readonly=[state, va, output_weight, dense1_weight, dense1_bias, dense2_weight])


def _set_target(builder: ProgramBuilder, specialization: TripletSiteSpecialization) -> None:
    builder.set_target(lc0ex_pb2.Target.VENDOR_NVIDIA, f"sm_{specialization.architecture}")


def _append_out(  # noqa: PLR0913
    builder: ProgramBuilder, kernels: KernelCache, state: Buffer, va: Buffer, output_weight: Buffer,
    specialization: TripletSiteSpecialization,
) -> None:
    builder.call(kernels.get(compile_triplet_out, specialization), state, va, output_weight,
                 readonly=[va, output_weight])


def triplet_out(  # noqa: PLR0913
    builder: ProgramBuilder, kernels: KernelCache, state: Buffer, va: Buffer, tables: Mapping[str, Buffer],
    specialization: TripletSiteSpecialization, *, after_block: int,
) -> None:
    """Append `out` alone: ``state`` (e_hat) += ``W_o va`` in place, ``va`` FP32 or (`va_f16`) FP16 (after `triplet_mma`)."""
    if va is state:
        message = "triplet_out needs distinct buffers: the edge state and va"
        raise ValueError(message)
    _set_target(builder, specialization)
    _append_out(builder, kernels, state, va, tables[triplet_table_names(after_block)[3]], specialization)


def triplet_site(  # noqa: PLR0913
    builder: ProgramBuilder,
    kernels: KernelCache,
    state: Buffer,
    values: Buffer,
    gates: Buffer | None,
    tables: Mapping[str, Buffer],
    specialization: TripletSiteSpecialization,
    *,
    after_block: int,
    copy: Buffer | None = None,
    out: bool = True,
) -> None:
    """Append the calls that add one site's triplet branch into ``state`` (e_hat) in place.

    With ``out=False`` the `out` stage is left to `triplet_out_ffn` (K4b c) and ``state`` stays e_hat.

    `tables` maps the plan names of `triplet_table_names(after_block)` to their persistent FP32 buffers. Form
    "scratch" (K4): prep, contract, out; `values` and `gates` are scratch, sized by `buffer_bytes`, and `values`
    also carries `va`. Form "fused" (K4b): fused, out; `values` is the ``va`` buffer
    (`buffer_bytes()["va"]`) and `gates` is unused (pass None); with `state_f16`, `copy` is the FP16 e_hat the
    fused launches read (`buffer_bytes()["copy"]`, written by `triplet_readback`). Run this between `edge_site`
    stage "readback" (or `triplet_readback`), which writes e_hat, and stage "ffn", which reads it.
    """
    if values is state or gates is state or (gates is not None and values is gates):
        message = "the triplet needs distinct buffers: the edge state, the values (or va) buffer and the gates scratch"
        raise ValueError(message)
    _set_target(builder, specialization)
    value_weight, gate_weight, gate_bias, output_weight = (tables[name] for name in triplet_table_names(after_block))
    if specialization.form == "fused":
        if specialization.state_f16 != (copy is not None):
            message = "form 'fused' takes the FP16 e_hat copy exactly when state_f16 is set"
            raise ValueError(message)
        source = copy if copy is not None else state
        builder.call(kernels.get(compile_triplet_fused, specialization), values, source, value_weight, gate_weight,
                     gate_bias, readonly=[source, value_weight, gate_weight, gate_bias])
        if out:
            _append_out(builder, kernels, state, values, output_weight, specialization)
        return
    if gates is None:
        message = "form 'scratch' needs the gates scratch buffer"
        raise ValueError(message)
    builder.call(kernels.get(compile_triplet_prep, specialization), values, gates, state, value_weight, gate_weight,
                 gate_bias, readonly=[state, value_weight, gate_weight, gate_bias])
    builder.call(kernels.get(compile_triplet_contract, specialization), values, gates, readonly=[gates])
    if out:
        _append_out(builder, kernels, state, values, output_weight, specialization)


def triplet_site_with_readback(  # noqa: PLR0913
    builder: ProgramBuilder,
    kernels: KernelCache,
    output: Buffer,
    state: Buffer,
    logits: Buffer,
    values: Buffer,
    gates: Buffer,
    tables: Mapping[str, Buffer],
    readback_weight: Buffer,
    specialization: TripletSiteSpecialization,
    *,
    after_block: int,
) -> None:
    """K4b (a): append readback_prep, contract and out -- K3's stage "readback" fused into K4's prep.

    ``output`` receives e_hat (and, after `out`, e_hat2 in place); ``state`` is e and ``logits`` the block's H, both
    read only. `readback_weight` is the site's `readback/w` plan (`readback_table_name(after_block)`). Form must be
    "scratch" (the two scratch buffers are still written and read). `edge_site` stage "ffn" closes the site.
    """
    if specialization.form != "scratch":
        message = "triplet_site_with_readback stages K4b (a) on the scratch chain; use form 'scratch'"
        raise ValueError(message)
    if output is state or values is state or gates is state or values is gates or output in (values, gates):
        message = "the triplet needs distinct buffers: e, e_hat, the values scratch and the gates scratch"
        raise ValueError(message)
    _set_target(builder, specialization)
    value_weight, gate_weight, gate_bias, output_weight = (tables[name] for name in triplet_table_names(after_block))
    builder.call(kernels.get(compile_triplet_readback_prep, specialization), values, gates, output, state, logits,
                 readback_weight, value_weight, gate_weight, gate_bias,
                 readonly=[state, logits, readback_weight, value_weight, gate_weight, gate_bias])
    builder.call(kernels.get(compile_triplet_contract, specialization), values, gates, readonly=[gates])
    _append_out(builder, kernels, output, values, output_weight, specialization)


def triplet_readback(  # noqa: PLR0913
    builder: ProgramBuilder,
    kernels: KernelCache,
    output: Buffer,
    copy: Buffer,
    state: Buffer,
    logits: Buffer,
    readback_weight: Buffer,
    specialization: TripletSiteSpecialization,
) -> None:
    """Append K3's readback with the FP16 e_hat copy: ``output`` = e_hat (FP32), ``copy`` = e_hat (FP16).

    Replaces `edge_site` stage "readback" ahead of a `state_f16` fused triplet; e_hat is the same expression and
    the same bits. ``state`` is e and ``logits`` the block's H, both read only.
    """
    if output is state or copy is state or output is copy:
        message = "triplet_readback needs distinct buffers: e, e_hat and its FP16 copy"
        raise ValueError(message)
    _set_target(builder, specialization)
    builder.call(kernels.get(compile_triplet_readback, specialization), output, copy, state, logits,
                 readback_weight, readonly=[state, logits, readback_weight])

