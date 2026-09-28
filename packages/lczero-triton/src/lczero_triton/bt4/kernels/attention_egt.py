"""EGT2 attention (item E, K2): the 34-channel logit assembly, edge-state reads, door and gate, fused per head.

What one block computes (the item E map, section 3, with R0's folds; per position z, head h, query i, key j):

    F = qk_scale[h] (q_i . k_j) + sum_c attack[h,c] E[c,i,j] + sum_c (k_j . key[h,c]) E[c,i,j]
        + sum_c (q_i . query[h,c]) E[c,i,j] + S[i,j] (sum_c scaled_coefficients[h,c] E[c,i,j] + constant_bias[h,i,j])
    H = (F + sum_c edge_read[h,c] e[c,i,j]) * (1 + sum_c door[h,c] e[c,i,j])          <- FP32, exported at 3/7/11
    A = softmax_j(H) * gate,   gate = min(2 sigmoid(sum_c gate_w[h,c] e[c,i,j] + gate_b[h]), 1) if cap else 2 sigmoid(.)
    out = sum_j A[h,i,j] v[h,j]           (the out-projection, skip, norms and GLU that follow are static P1-P3)

The tables are R0's FP32 plans `/encoder{i}/mha/egt/*` (`lab/_names.py`), read as planned: no new folds.
Inputs: Q, K, V packed ``[q | k | v]`` per row (P3, FP16 served); S ``[batch, 64, 64]`` (FP16 served, FP32 allowed);
e ``[batch, 16, 64, 64]`` (FP32, K1's default); E as K1's packed ``uint64 [batch, 64, 64]`` and, for the fast path,
the edge list built from it once per batch.

Choices, by measured time (one RTX 5080, batch 64 x 32 heads, kernel time; `REPORT_itemE_K2`)
------------------------------------------------------------------------------------------
* **q.k**: FP16 inputs into an FP32 accumulator (`out_dtype=tl.float32`), static's form: 25.0 us for the q.k-only
  baseline, against 26.9 us for FP32 TF32 and 39.1 us for FP32 IEEE. Every dot pins `input_precision="ieee"`:
  Triton 3.7's default for FP32 inputs is TF32 (9e-4 relative on this data), which the FP32-class gates would see.
* **E contraction: a per-batch bit list.** E is sparse (358 set bits per position, median; 568 max of 139,264 on
  the reference set) and fixed for all 15 blocks. `egt_edge_list` lists the set bits once per batch, as
  (cell, channel) in cell order, with each cell's [start, end) span. Per head: ``Qq = q.query^T + attack`` and
  ``Qk = k.key^T`` as two narrow dots each (channels 0-31 and 32-33), one contribution per listed bit
  ``Qq[i,c] + Qk[j,c] + S[i,j] G[c]`` (four gathers, one load), FP32 prefix sums over the list, and the dense tile
  ``sums[end] - sums[start]``. Measured with q.k: 51.3 us (capacity 1024), against 234 us for 6-bit code-group tables
  (12 table dots), 440 us for 34 bit planes on the split words and 1,974 us for 4-bit group tables; batched dots over
  ``[64, 32, 64]`` bit blocks do not run (2.1 MB of L1 shared memory against 99 KB). The prefix sums cost precision:
  1.5e-6 of peak on the E terms against 1.5e-7 dense, inside the map's 1e-5 FP32-class rule.
* **e reads: in the head program.** 16 FP32 state tiles loaded and contracted three times: 97.6 us, against 136.5 us
  for loading three precomputed FP32 read tiles alone (plus a 154.5 us GEMM to make them; FP16 tiles 77 + 166 us and
  2e-3 relative). A ``[16, 4096]`` block does not fit L1.
* **Softmax and gate in FP32; output FP16**, as static. No clamp before the softmax (the export's FP32 H is finite).

Capacity and the exact fallback (K2b)
-------------------------------------
A list of `capacity` slots holds a position's set bits exactly when ``counts[z] <= capacity - 1`` (the last slot is
the prefix sums' zero element). **Above that the head program takes a per-position runtime branch** and contracts that
position's E terms densely instead: 34 bit planes over the split 17-bit words of E, gathering the same ``Qq``/``Qk``
tables, in FP32 (the same bit-plane form the E-contraction study measured at 440 us). So the kernel is exact for any E, and the
capacity only trades speed:
positions past it cost the dense path. The fast path's arithmetic is unchanged by the branch, bit for bit (checked
against the pre-fallback kernel at every num_warps, in both precisions), and the branch costs about 12 % on it
(177 -> 198 us at batch 64 with H export, num_warps 4); with every position on the dense path the call takes 576 us.

Edge list (the K2 per-batch buffers; `egt_edge_list`)
------------------------------------------------------
``cells int16 [batch, capacity]`` (8 * row + column), ``channels int8 [batch, capacity]``, ``prefix int16
[batch, 2, 4096]`` (start, end), ``counts int32 [batch]`` (set bits per position, never clamped: it is what the
kernel branches on). Three kernels, all over 64 cells (per-sample 4,096-cell builders compiled for minutes): per
(position, row) set-bit totals, per position their running sum, per (position, row) the list.

Edge-state read modes (round 21, K2c)
-------------------------------------
`edge_state` is either the state itself (K2: 16 tiles loaded and contracted in the head program; FP32 or, with
`state_f32=False`, the FP16 copy `egt_state_tiles.cast_state` writes) or, with `state_tiles`, the block's
precomputed read tiles: three [64, 64] tiles per head, FP16, at ``((sample * round_heads + head_in_round) * 3 +
term) * 4096`` (`egt_state_tiles`). The head program then loads 3 x 8 KB instead of 16 x 16 KB. `round_heads`
splits the block into ``heads // round_heads`` rounds of (tiles, attention) so the tile buffer stays inside L2 at
any batch: the grid is ``samples * round_heads`` programs and `head_base` names the round's first head.

H export (the K3 interface, fixed)
----------------------------------
With `export_h`, H (post-door, pre-softmax) goes to FP32 ``[batch * heads, 64, 64]``, row-major, at
``(sample * heads + head) * 4096 + 64 * i + j``: the layout of static's smolgen logits, read by `edge_site`.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import cast

import torch
import triton
import triton.language as tl
from lc0ex import Buffer, KernelArtifact, ProgramBuilder
from lc0ex.proto import lc0ex_pb2
from lc0ex.triton_module_compiler import artifact_from_triton

from lczero_triton.bt4.kernels._cache import KernelCache

_POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER
TOKENS = 64
CELLS = TOKENS * TOKENS
EDGE_CHANNELS = 34
BASE_CHANNELS = 17
STATE_CHANNELS = 16
DEFAULT_CAPACITY = 1024

# Logit terms, combinable, so tests can enable them one at a time.
EGT_QK = 1  # qk_scale * q.k
EGT_ATTACK = 2  # attack . E
EGT_KEY = 4  # (k . key) . E
EGT_QUERY = 8  # (q . query) . E
EGT_SCALED = 16  # S * (scaled_coefficients . E + constant_bias)
EGT_EDGE_READ = 32  # + edge_read . e
EGT_DOOR = 64  # * (1 + door . e)
EGT_GATE = 128  # softmax * gate(gate_w . e + gate_b)
EGT_FULL = 255
EGT_EDGE_TERMS = EGT_ATTACK | EGT_KEY | EGT_QUERY | EGT_SCALED
EGT_STATE_TERMS = EGT_EDGE_READ | EGT_DOOR | EGT_GATE

# The block's R0 plans, in kernel argument order: f"{prefix}/mha/egt/{name}".
BLOCK_TABLES = ("qk_scale", "attack", "key", "query", "scaled_coefficients", "constant_bias", "edge_read/w", "door/w",
                "gate/w", "gate/b")


def block_table_names(prefix: str) -> tuple[str, ...]:
    """Return the plan names one block's kernel reads, for an encoder prefix such as ``/encoder3``."""
    return tuple(f"{prefix}/mha/egt/{name}" for name in BLOCK_TABLES)


# ---------------------------------------------------------------- the edge list (once per batch)
@triton.jit
def _row_population(code):
    """Set bits per cell of one row of packed E (split into its two 17-bit halves: K1's uint64 decode trap)."""
    low = (code & 0x1FFFF).to(tl.int32)  # jit code reads no module globals: 17 = BASE_CHANNELS
    high = (code >> tl.full((), 17, tl.uint64)).to(tl.int32)
    population = tl.zeros((64,), dtype=tl.int32)
    for channel in tl.static_range(17):
        population += ((low >> channel) & 1) + ((high >> channel) & 1)
    return low, high, population


@triton.autotune(configs=[triton.Config({}, num_warps=warps) for warps in (1, 2, 4)], key=["batch_count"],
                 cache_results=True)
@triton.jit
def _egt_edge_count_kernel(row_totals, edges, batch_count: tl.constexpr) -> None:  # noqa: ARG001
    """Set bits of one row of one position."""
    program = tl.program_id(0)  # sample * 64 + row
    columns = tl.arange(0, 64)
    code = tl.load(edges + program * 64 + columns)
    _, _, population = _row_population(code)
    tl.store(row_totals + program, tl.sum(population, 0))


@triton.autotune(configs=[triton.Config({}, num_warps=warps) for warps in (1, 2, 4)], key=["batch_count"],
                 cache_results=True)
@triton.jit
def _egt_edge_offset_kernel(row_starts, counts, row_totals, batch_count: tl.constexpr) -> None:  # noqa: ARG001
    """Where each row's bits start in the position's list, and the position's count (never clamped)."""
    sample = tl.program_id(0)
    rows = tl.arange(0, 64)
    totals = tl.load(row_totals + sample * 64 + rows)
    ends = tl.cumsum(totals, 0)
    tl.store(row_starts + sample * 64 + rows, ends - totals)
    tl.store(counts + sample, tl.sum(totals, 0))


@triton.autotune(configs=[triton.Config({}, num_warps=warps) for warps in (1, 2, 4)],
                 key=["batch_count", "capacity"], cache_results=True)
@triton.jit
def _egt_edge_list_kernel(  # noqa: PLR0913
    cells,
    channels,
    prefix,
    edges,
    row_starts,
    batch_count: tl.constexpr,  # noqa: ARG001
    capacity: tl.constexpr,
) -> None:
    """List one row's set bits (cell order, channel order within a cell) and write its cells' [start, end) spans.

    Positions past ``capacity - 1`` are dropped and the spans clamp; the kernel detects that from `counts` and
    contracts such a position densely instead.
    """
    program = tl.program_id(0)
    sample = program // 64
    row = program % 64
    columns = tl.arange(0, 64)
    code = tl.load(edges + program * 64 + columns)
    low, high, population = _row_population(code)
    start = tl.load(row_starts + program) + tl.cumsum(population, 0) - population
    limit = capacity - 1
    tl.store(prefix + sample * 2 * 4096 + row * 64 + columns, tl.minimum(start, limit).to(tl.int16))
    tl.store(prefix + sample * 2 * 4096 + 4096 + row * 64 + columns, tl.minimum(start + population, limit).to(tl.int16))
    index = (row * 64 + columns).to(tl.int16)
    base_cells = cells + sample * capacity
    base_channels = channels + sample * capacity
    position = start
    for channel in tl.static_range(17):
        bit = (low >> channel) & 1
        keep = (bit == 1) & (position < limit)
        tl.store(base_cells + position, index, mask=keep)
        tl.store(base_channels + position, tl.full((64,), channel, tl.int8), mask=keep)
        position += bit
    for channel in tl.static_range(17):
        bit = (high >> channel) & 1
        keep = (bit == 1) & (position < limit)
        tl.store(base_cells + position, index, mask=keep)
        tl.store(base_channels + position, tl.full((64,), 17 + channel, tl.int8), mask=keep)
        position += bit


# ---------------------------------------------------------------- the E terms, list path and dense fallback (K2e)
@triton.jit
def _list_terms(  # noqa: PLR0913
    cells, channels, prefix, counts, normalization, row_flat, row_narrow_flat, column_flat, column_narrow_flat,
    scaled_coefficients, coefficient_vector, sample, head, grid,
    capacity: tl.constexpr, use_query: tl.constexpr, use_attack: tl.constexpr, use_key: tl.constexpr,
    use_scaled: tl.constexpr, coefficient_vectors: tl.constexpr, scaled_stream: tl.constexpr,
):
    """The four E terms of one head over the position's set-bit list: ``[4096]`` dense, K2's prefix-sum form."""
    bits = tl.arange(0, capacity)
    valid = bits < tl.minimum(tl.load(counts + sample), capacity - 1)
    cell = tl.where(valid, tl.load(cells + sample * capacity + bits).to(tl.int32), 0)
    channel = tl.where(valid, tl.load(channels + sample * capacity + bits).to(tl.int32), 0)
    is_wide = channel < 32
    wide_channel = tl.minimum(channel, 31)
    narrow_channel = tl.maximum(channel - 32, 0)
    contribution = tl.zeros((capacity,), dtype=tl.float32)
    if use_query or use_attack:
        row = cell >> 6
        contribution += tl.where(is_wide, tl.gather(row_flat, row * 32 + wide_channel, 0),
                                 tl.gather(row_narrow_flat, row * 2 + narrow_channel, 0))
    if use_key:
        column = cell & 63
        contribution += tl.where(is_wide, tl.gather(column_flat, column * 32 + wide_channel, 0),
                                 tl.gather(column_narrow_flat, column * 2 + narrow_channel, 0))
    if use_scaled:
        if coefficient_vectors:
            coefficient = tl.gather(coefficient_vector, channel, 0)
        else:
            coefficient = tl.load(scaled_coefficients + head * 34 + channel)
        if scaled_stream:
            scaled_contribution = tl.where(valid, coefficient, 0.0)
        else:
            contribution += tl.gather(normalization, cell, 0) * coefficient
    contribution = tl.where(valid, contribution, 0.0)
    # sums[b] = contributions of bits 0..b-1, so a cell's total is sums[end] - sums[start].
    shifted = tl.where(bits > 0, tl.gather(contribution, tl.maximum(bits - 1, 0), 0), 0.0)
    sums = tl.cumsum(shifted, 0)
    start = tl.load(prefix + sample * 2 * 4096 + grid).to(tl.int32)
    end = tl.load(prefix + sample * 2 * 4096 + 4096 + grid).to(tl.int32)
    dense = tl.gather(sums, end, 0) - tl.gather(sums, start, 0)
    if use_scaled and scaled_stream:
        shifted_scaled = tl.where(bits > 0, tl.gather(scaled_contribution, tl.maximum(bits - 1, 0), 0), 0.0)
        sums_scaled = tl.cumsum(shifted_scaled, 0)
        dense += normalization * (tl.gather(sums_scaled, end, 0) - tl.gather(sums_scaled, start, 0))
    return dense


@triton.jit
def _dense_terms(  # noqa: PLR0913
    edges, normalization, row_flat, row_narrow_flat, column_flat, column_narrow_flat, scaled_coefficients, sample,
    head, grid, use_scaled: tl.constexpr,
):
    """The same four terms over E's 34 bit planes directly (K2b's exact fallback past capacity - 1)."""
    code = tl.load(edges + sample * 4096 + grid)
    low = (code & 0x1FFFF).to(tl.int32)
    high = (code >> tl.full((), 17, tl.uint64)).to(tl.int32)
    row_index = grid >> 6
    column_index = grid & 63
    dense = tl.zeros((4096,), dtype=tl.float32)
    for channel in tl.static_range(34):
        if channel < 32:
            coefficient = tl.gather(row_flat, row_index * 32 + channel, 0)
            coefficient += tl.gather(column_flat, column_index * 32 + channel, 0)
        else:
            coefficient = tl.gather(row_narrow_flat, row_index * 2 + (channel - 32), 0)
            coefficient += tl.gather(column_narrow_flat, column_index * 2 + (channel - 32), 0)
        if use_scaled:
            coefficient += normalization * tl.load(scaled_coefficients + head * 34 + channel)
        if channel < 17:
            bit = ((low >> channel) & 1).to(tl.float32)
        else:
            bit = ((high >> (channel - 17)) & 1).to(tl.float32)
        dense += bit * coefficient
    return dense


# ---------------------------------------------------------------- attention, one program per (sample, head)
_WARPS = (2, 4, 8, 16)


@triton.autotune(
    configs=[triton.Config({}, num_warps=warps) for warps in _WARPS],
    key=["batch_count", "heads", "head_dim", "capacity", "cap", "export_h", "export_weights", "use_qk", "use_attack",
         "use_key", "use_query", "use_scaled", "use_read", "use_door", "use_gate", "state_tiles", "round_heads",
         "head_base", "coefficient_vectors", "scaled_stream", "overflow_exact", "overflow_only",
         "state_f8", "state_i8", "quant_output", "output_gate"],
    cache_results=True,
)
@triton.jit
def _attention_egt_kernel(  # noqa: PLR0913, PLR0915, C901
    output,
    logits,
    weights,
    qkv,
    cells,
    channels,
    prefix,
    counts,
    edges,
    edge_norm,
    edge_state,
    qk_scale,
    attack,
    key_table,
    query_table,
    scaled_coefficients,
    constant_bias,
    read_weights,
    door_weights,
    gate_weights,
    gate_bias,
    quant_prescale,
    batch_count: tl.constexpr,  # noqa: ARG001  # Autotune key and launch grid.
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    capacity: tl.constexpr,
    cap: tl.constexpr,
    export_h: tl.constexpr,
    export_weights: tl.constexpr,
    use_qk: tl.constexpr,
    use_attack: tl.constexpr,
    use_key: tl.constexpr,
    use_query: tl.constexpr,
    use_scaled: tl.constexpr,
    use_edges: tl.constexpr,
    use_read: tl.constexpr,
    use_door: tl.constexpr,
    use_gate: tl.constexpr,
    use_state: tl.constexpr,
    state_tiles: tl.constexpr,
    round_heads: tl.constexpr,
    head_base: tl.constexpr,
    coefficient_vectors: tl.constexpr,
    scaled_stream: tl.constexpr,
    overflow_exact: tl.constexpr,
    overflow_only: tl.constexpr,
    state_f8: tl.constexpr = False,
    state_scales: tl.constexpr = (),
    state_i8: tl.constexpr = False,
    quant_output: tl.constexpr = False,
    output_gate: tl.constexpr = False,
    gate_scale: tl.constexpr = 2.0,
) -> None:
    """One head's EGT2 attention block end to end.

    `quant_output` (Q1, round 26): the block's output is written as the out-projection's int8 operand -- the
    `attn_out` codes `floor(o_j * r_j + 0.5)` clamped to +-127, `r` = `quant_prescale` over the output channels --
    instead of FP16, so the conversion pass and the FP16 round trip of the attention output disappear.

    `output_gate` (O on the EGT2 blocks, round 26b): `qkv` is the packed `[q | k | v | g]` (stride 4 width) and the
    attended tile is multiplied by `gate_scale * sigmoid(g)` in FP32 before the store -- the attention OUTPUT gate
    (`mha_output_gate`), not the edge door's per-head gate (`use_gate`).
    """
    program = tl.program_id(0)
    sample = program // round_heads
    if overflow_only:
        # K2e: the correction launch. Positions inside the list's capacity were served exactly by the main launch.
        if tl.load(counts + sample) <= capacity - 1:
            return
    head = head_base + program % round_heads
    matrix = sample * heads + head  # the (sample, head) index of the H and A exports, whatever the round
    width = heads * head_dim
    stride = 3 * width
    if output_gate:
        stride = 4 * width
    squares = tl.arange(0, 64)
    depth = tl.arange(0, head_dim)
    direct = squares[:, None] * 64 + squares[None, :]
    row_base = sample * 64 * stride + head * head_dim
    query = tl.load(qkv + row_base + squares[:, None] * stride + depth[None, :])
    key = tl.load(qkv + row_base + width + squares[:, None] * stride + depth[None, :])

    scores = tl.zeros((64, 64), dtype=tl.float32)
    if use_qk:
        scores += tl.dot(query, tl.trans(key), input_precision="ieee", out_dtype=tl.float32) * tl.load(qk_scale + head)

    if use_edges:
        grid = tl.arange(0, 4096)
        wide = tl.arange(0, 32)
        narrow = tl.arange(0, 2)
        pair = head * 34 * head_dim
        normalization = tl.load(edge_norm + sample * 4096 + grid).to(tl.float32)
        # Qq and Qk per channel, shared by the list path and the dense fallback.
        if use_query or use_attack:
            row_wide = tl.zeros((64, 32), dtype=tl.float32)
            row_narrow = tl.zeros((64, 2), dtype=tl.float32)
            if use_query:
                table = tl.load(query_table + pair + wide[None, :] * head_dim + depth[:, None]).to(query.dtype)
                row_wide += tl.dot(query, table, input_precision="ieee", out_dtype=tl.float32)
                table = tl.load(query_table + pair + (32 + narrow[None, :]) * head_dim + depth[:, None]).to(query.dtype)
                row_narrow += tl.dot(query, table, input_precision="ieee", out_dtype=tl.float32)
            if use_attack:
                row_wide += tl.load(attack + head * 34 + wide)[None, :]
                row_narrow += tl.load(attack + head * 34 + 32 + narrow)[None, :]
            row_flat = tl.reshape(row_wide, (2048,))
            row_narrow_flat = tl.reshape(row_narrow, (128,))
        else:
            row_flat = tl.zeros((2048,), dtype=tl.float32)
            row_narrow_flat = tl.zeros((128,), dtype=tl.float32)
        if use_key:
            table = tl.load(key_table + pair + wide[None, :] * head_dim + depth[:, None]).to(key.dtype)
            column_flat = tl.reshape(tl.dot(key, table, input_precision="ieee", out_dtype=tl.float32), (2048,))
            table = tl.load(key_table + pair + (32 + narrow[None, :]) * head_dim + depth[:, None]).to(key.dtype)
            column_narrow_flat = tl.reshape(tl.dot(key, table, input_precision="ieee", out_dtype=tl.float32), (128,))
        else:
            column_flat = tl.zeros((2048,), dtype=tl.float32)
            column_narrow_flat = tl.zeros((128,), dtype=tl.float32)

        # A list of `capacity` slots is exact up to capacity - 1 set bits; past that this position takes the dense
        # fallback, which reads E directly (K2b). Non-overflowing positions run exactly the arithmetic they ran
        # before the fallback existed. K2e: `overflow_exact=False` compiles the branch out (list path only).
        padded = tl.arange(0, 64)
        coefficient_vector = tl.load(scaled_coefficients + head * 34 + padded, mask=padded < 34, other=0.0)
        if overflow_only:
            dense = _dense_terms(edges, normalization, row_flat, row_narrow_flat, column_flat,
                                 column_narrow_flat, scaled_coefficients, sample, head, grid, use_scaled)
        elif overflow_exact:
            if tl.load(counts + sample) > capacity - 1:
                dense = _dense_terms(edges, normalization, row_flat, row_narrow_flat, column_flat,
                                     column_narrow_flat, scaled_coefficients, sample, head, grid, use_scaled)
            else:
                dense = _list_terms(cells, channels, prefix, counts, normalization, row_flat, row_narrow_flat,
                                    column_flat, column_narrow_flat, scaled_coefficients, coefficient_vector,
                                    sample, head, grid, capacity, use_query, use_attack, use_key, use_scaled,
                                    coefficient_vectors, scaled_stream)
        else:
            dense = _list_terms(cells, channels, prefix, counts, normalization, row_flat, row_narrow_flat,
                                column_flat, column_narrow_flat, scaled_coefficients, coefficient_vector,
                                sample, head, grid, capacity, use_query, use_attack, use_key, use_scaled,
                                coefficient_vectors, scaled_stream)
        if use_scaled:
            dense += normalization * tl.load(constant_bias + head * 4096 + grid)
        scores += tl.reshape(dense, (64, 64))

    if use_state:
        if state_tiles:
            # K2c: this head's three read tiles, built once per (sample, round) by `egt_state_tiles`.
            tile_base = (sample * round_heads + (head - head_base)) * 3 * 4096
            read = tl.load(edge_state + tile_base + direct).to(tl.float32)
            door = tl.load(edge_state + tile_base + 4096 + direct).to(tl.float32)
            gate = tl.load(edge_state + tile_base + 8192 + direct).to(tl.float32)
        else:
            read = tl.zeros((64, 64), dtype=tl.float32)
            door = tl.zeros((64, 64), dtype=tl.float32)
            gate = tl.zeros((64, 64), dtype=tl.float32)
            state_base = sample * 16 * 4096
            for state_channel in tl.static_range(16):
                tile = tl.load(edge_state + state_base + state_channel * 4096 + direct).to(tl.float32)
                if state_f8 or state_i8:  # r23b I8: the int8 copy folds its scale exactly as e4m3 does
                    # r23 F8: e4m3 stores the channel in units of `state_scales[c]`. The scale is a compile-time
                    # constant, so it folds into the plan's SCALAR weight -- one extra multiply per (term,
                    # channel), never one per element of the [64, 64] tile.
                    if use_read:
                        read += (tl.load(read_weights + head * 16 + state_channel)
                                 * state_scales[state_channel]) * tile
                    if use_door:
                        door += (tl.load(door_weights + head * 16 + state_channel)
                                 * state_scales[state_channel]) * tile
                    if use_gate:
                        gate += (tl.load(gate_weights + head * 16 + state_channel)
                                 * state_scales[state_channel]) * tile
                else:
                    if use_read:
                        read += tl.load(read_weights + head * 16 + state_channel) * tile
                    if use_door:
                        door += tl.load(door_weights + head * 16 + state_channel) * tile
                    if use_gate:
                        gate += tl.load(gate_weights + head * 16 + state_channel) * tile
        if use_read:
            scores += read
        if use_door:
            scores = scores * (1.0 + door)

    if export_h:
        tl.store(logits + matrix * 4096 + direct, scores)

    maximum = tl.max(scores, axis=1)
    exponentials = tl.exp(scores - maximum[:, None])
    probabilities = exponentials / tl.sum(exponentials, axis=1)[:, None]
    if use_gate:
        gate += tl.load(gate_bias + head)
        if cap:
            probabilities = probabilities * tl.where(gate >= 0.0, 1.0, 2.0 * tl.sigmoid(gate))
        else:
            probabilities = probabilities * (2.0 * tl.sigmoid(gate))
    if export_weights:
        tl.store(weights + matrix * 4096 + direct, probabilities)

    values = tl.load(qkv + row_base + 2 * width + squares[:, None] * stride + depth[None, :])
    attended = tl.dot(probabilities.to(values.dtype), values, input_precision="ieee", out_dtype=tl.float32)
    if output_gate:
        # O: `gate_scale * sigmoid(g)` per (square, channel) of this head, in FP32 on the FP32 accumulator, as
        # `attention_static` does; the lab computes the gate in float32 because it multiplies the whole branch.
        pre_activation = tl.load(qkv + row_base + 3 * width + squares[:, None] * stride + depth[None, :])
        attended = attended * (gate_scale * tl.sigmoid(pre_activation.to(tl.float32)))
    destination = output + sample * 64 * width + squares[:, None] * width + head * head_dim + depth[None, :]
    if quant_output:
        # The same rule as every int8 the artifact writes (`layer_norm`, `quantise_operand`, `cutlass_gemm_i8`).
        prescale = tl.load(quant_prescale + head * head_dim + depth)
        codes = tl.clamp(tl.floor(attended * prescale[None, :] + 0.5), -127.0, 127.0)
        tl.store(destination, codes.to(tl.int8))
    else:
        tl.store(destination, attended.to(tl.float16))


# ---------------------------------------------------------------- specializations, compile, builder
@dataclass(frozen=True, slots=True)
class EgtEdgeListSpecialization:
    """Immutable edge-list specialization: one per batch size and capacity."""

    batch_size: int
    architecture: int
    capacity: int = DEFAULT_CAPACITY


@dataclass(frozen=True, slots=True)
class AttentionEgtSpecialization:
    """Immutable EGT2 attention specialization.

    `batch_count` is samples x heads (one program per head block, as `attention_static`). `cap` selects the capped
    gate (gcap) or the plain one (the triplet exports). `export_h` writes H for the next update site. `norm_f32` and
    `state_f32` are the precisions of S and e (K1's defaults: FP16 and FP32). `export_weights` (tests only) writes A.
    `capacity` is the edge list's; positions with more set bits than `capacity - 1` take the exact dense fallback.
    """

    batch_count: int
    heads: int
    head_dim: int
    architecture: int
    cap: bool
    export_h: bool = False
    logit_terms: int = EGT_FULL
    capacity: int = DEFAULT_CAPACITY
    norm_f32: bool = False
    state_f32: bool = True
    export_weights: bool = False
    # K2c: `edge_state` holds the precomputed read tiles of `round_heads` heads from `head_base` (0 = all heads).
    state_tiles: bool = False
    round_heads: int = 0
    head_base: int = 0
    # K2e: the list path's three switches (defaults = K2b's arithmetic, bit for bit).
    coefficient_vectors: bool = False
    scaled_stream: bool = False
    overflow_exact: bool = True
    # K2e step 2: the correction launch (dense path for overflow positions only; everything else returns).
    overflow_only: bool = False
    # r23 F8: `edge_state` is the e4m3 copy and `state_scales` its per-channel scale (a build-time constant).
    state_f8: bool = False
    state_scales: tuple[float, ...] = ()
    # r23b I8: `edge_state` is the int8 copy; `state_scales` is then its per-channel int8 scale.
    state_i8: bool = False
    # Q1 (round 26): the output is the out-projection's int8 operand, through the `attn_out` vector.
    quant_output: bool = False
    # O (round 26b): the attention output gate as the fourth lane of the packed projection, `[q | k | v | g]`.
    output_gate: bool = False
    gate_scale: float = 2.0


@dataclass(frozen=True, slots=True)
class EgtEdgeList:
    """The per-batch edge-list buffers every block reads, and the two scratch buffers that build them."""

    cells: Buffer
    channels: Buffer
    prefix: Buffer
    counts: Buffer
    row_totals: Buffer
    row_starts: Buffer


def _check(specialization: AttentionEgtSpecialization) -> None:
    if not 0 < specialization.logit_terms <= EGT_FULL:
        message = f"logit_terms={specialization.logit_terms} is out of range (1..{EGT_FULL})"
        raise ValueError(message)
    depth = specialization.head_dim
    if depth < 16 or depth & (depth - 1):  # noqa: PLR2004
        message = f"head_dim={depth} must be a power of two >= 16 (tl.dot and tl.arange)"
        raise ValueError(message)
    _check_capacity(specialization.capacity)
    if specialization.state_i8:
        # r23b I8: the int8 copy is its own read mode, like the e4m3 one.
        if specialization.state_f8 or specialization.state_f32 or specialization.state_tiles:
            message = "state_i8 excludes state_f8, state_f32 and state_tiles"
            raise ValueError(message)
        if len(specialization.state_scales) != STATE_CHANNELS:
            message = f"state_i8 needs {STATE_CHANNELS} scales, got {len(specialization.state_scales)}"
            raise ValueError(message)
    elif specialization.state_f8:
        # r23 F8: the e4m3 copy is its own read mode -- never the FP32 state, never the precomputed tiles.
        if specialization.state_f32 or specialization.state_tiles:
            message = "state_f8 excludes state_f32 and state_tiles"
            raise ValueError(message)
        if len(specialization.state_scales) != STATE_CHANNELS:
            message = f"state_f8 needs {STATE_CHANNELS} scales, got {len(specialization.state_scales)}"
            raise ValueError(message)
    elif specialization.state_scales:
        message = "state_scales without state_f8"
        raise ValueError(message)
    heads, rounds, base = specialization.heads, round_heads_of(specialization), specialization.head_base
    if rounds <= 0 or heads % rounds:
        message = f"round_heads={rounds} must divide heads={heads}"
        raise ValueError(message)
    if base < 0 or base % rounds or base >= heads:
        message = f"head_base={base} must be a multiple of round_heads={rounds} below heads={heads}"
        raise ValueError(message)


def round_heads_of(specialization: "AttentionEgtSpecialization") -> int:
    """The heads one launch covers: `round_heads`, or every head when it is 0."""
    return specialization.round_heads or specialization.heads


def grid_size(specialization: "AttentionEgtSpecialization") -> int:
    """Programs per launch: one per (sample, head of the round)."""
    return specialization.batch_count // specialization.heads * round_heads_of(specialization)


def _check_capacity(capacity: int) -> None:
    if capacity < 64 or capacity > 16384 or capacity & (capacity - 1):  # noqa: PLR2004
        message = f"capacity={capacity} must be a power of two in [64, 16384] (int16 spans)"
        raise ValueError(message)


def term_flags(logit_terms: int) -> dict[str, bool]:
    """The kernel's term constexprs for a logit-term set."""
    return {
        "use_qk": bool(logit_terms & EGT_QK),
        "use_attack": bool(logit_terms & EGT_ATTACK),
        "use_key": bool(logit_terms & EGT_KEY),
        "use_query": bool(logit_terms & EGT_QUERY),
        "use_scaled": bool(logit_terms & EGT_SCALED),
        "use_edges": bool(logit_terms & EGT_EDGE_TERMS),
        "use_read": bool(logit_terms & EGT_EDGE_READ),
        "use_door": bool(logit_terms & EGT_DOOR),
        "use_gate": bool(logit_terms & EGT_GATE),
        "use_state": bool(logit_terms & EGT_STATE_TERMS),
    }


def buffer_bytes(specialization: AttentionEgtSpecialization) -> dict[str, int]:
    """Execution buffers one block call writes, and the per-batch edge list it reads, in bytes."""
    samples = specialization.batch_count // specialization.heads
    sizes = {
        "output": (1 if specialization.quant_output else 2) * samples * TOKENS * specialization.heads
        * specialization.head_dim,
        "logits (export_h)": 4 * specialization.batch_count * CELLS if specialization.export_h else 0,
    }
    return sizes | edge_list_bytes(EgtEdgeListSpecialization(samples, specialization.architecture,
                                                              specialization.capacity))


def edge_list_bytes(specialization: EgtEdgeListSpecialization) -> dict[str, int]:
    """The per-batch edge-list buffers, in bytes."""
    samples, capacity = specialization.batch_size, specialization.capacity
    return {
        "edge_list cells": 2 * samples * capacity,
        "edge_list channels": samples * capacity,
        "edge_list prefix": 2 * 2 * samples * CELLS,
        "edge_list counts": 4 * samples,
        "edge_list row_totals + row_starts (scratch)": 2 * 4 * samples * TOKENS,
    }


def _autotune_grid(configuration: Mapping[str, object]) -> tuple[int]:
    """One program per (sample, head of the round)."""
    samples = cast("int", configuration["batch_count"]) // cast("int", configuration["heads"])
    return (samples * cast("int", configuration["round_heads"]),)


def _row_grid(configuration: Mapping[str, object]) -> tuple[int]:
    """One program per (position, row)."""
    return (cast("int", configuration["batch_count"]) * TOKENS,)


def _sample_grid(configuration: Mapping[str, object]) -> tuple[int]:
    """One program per position."""
    return (cast("int", configuration["batch_count"]),)


def launch_attention_egt(  # noqa: PLR0913
    output: torch.Tensor,
    qkv: torch.Tensor,
    edge_list: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    edges: torch.Tensor,
    edge_norm: torch.Tensor,
    edge_state: torch.Tensor,
    tables: Mapping[str, torch.Tensor],
    specialization: AttentionEgtSpecialization,
    logits: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    quant_prescale: torch.Tensor | None = None,
) -> object:
    """Launch the kernel on torch tensors (tests, probes and `compile_attention_egt`); `tables` keyed by BLOCK_TABLES."""
    _check(specialization)
    if specialization.quant_output != (quant_prescale is not None):
        message = "quant_output and a quant_prescale tensor must come together"
        raise ValueError(message)
    placeholder = tables["qk_scale"]
    return _attention_egt_kernel[_autotune_grid](
        output,
        logits if logits is not None else placeholder,
        weights if weights is not None else placeholder,
        qkv,
        *edge_list,
        edges,
        edge_norm,
        edge_state,
        *(tables[name] for name in BLOCK_TABLES),
        quant_prescale if quant_prescale is not None else placeholder,
        specialization.batch_count,
        specialization.heads,
        specialization.head_dim,
        specialization.capacity,
        specialization.cap,
        specialization.export_h,
        specialization.export_weights,
        **term_flags(specialization.logit_terms),
        state_tiles=specialization.state_tiles,
        round_heads=round_heads_of(specialization),
        head_base=specialization.head_base,
        coefficient_vectors=specialization.coefficient_vectors,
        scaled_stream=specialization.scaled_stream,
        overflow_exact=specialization.overflow_exact,
        overflow_only=specialization.overflow_only,
        state_f8=specialization.state_f8,
        state_scales=specialization.state_scales,
        state_i8=specialization.state_i8,
        quant_output=specialization.quant_output,
        output_gate=specialization.output_gate,
        gate_scale=specialization.gate_scale,
    )


def launch_egt_edge_list(  # noqa: PLR0913
    cells: torch.Tensor,
    channels: torch.Tensor,
    prefix: torch.Tensor,
    counts: torch.Tensor,
    row_totals: torch.Tensor,
    row_starts: torch.Tensor,
    edges: torch.Tensor,
    capacity: int,
) -> tuple[object, object, object]:
    """Launch the three edge-list kernels on torch tensors."""
    _check_capacity(capacity)
    batch = edges.shape[0]
    count = _egt_edge_count_kernel[_row_grid](row_totals, edges, batch)
    torch.cuda.synchronize()
    offset = _egt_edge_offset_kernel[_sample_grid](row_starts, counts, row_totals, batch)
    torch.cuda.synchronize()
    listing = _egt_edge_list_kernel[_row_grid](cells, channels, prefix, edges, row_starts, batch, capacity)
    return count, offset, listing


def compile_attention_egt(specialization: AttentionEgtSpecialization) -> KernelArtifact:
    """Autotune and compile one EGT2 attention specialization."""
    _check(specialization)
    heads, depth = specialization.heads, specialization.head_dim
    samples = specialization.batch_count // heads
    width = heads * depth
    capacity = specialization.capacity
    norm_type = torch.float32 if specialization.norm_f32 else torch.float16
    state_type = torch.float32 if specialization.state_f32 else torch.float16
    if specialization.state_f8:  # r23 F8: e4m3 copy
        state_type = torch.float8_e4m3fn
    if specialization.state_i8:  # r23b I8: int8 copy
        state_type = torch.int8

    def zeros(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype, device="cuda")

    output = torch.empty((samples, TOKENS, width), dtype=torch.int8 if specialization.quant_output else torch.float16,
                         device="cuda")
    logits = torch.empty((specialization.batch_count, TOKENS, TOKENS), dtype=torch.float32, device="cuda")
    edge_list = (zeros(samples, capacity, dtype=torch.int16), zeros(samples, capacity, dtype=torch.int8),
                 zeros(samples, 2, CELLS, dtype=torch.int16), zeros(samples, dtype=torch.int32))
    tables = {
        "qk_scale": zeros(heads), "attack": zeros(heads, EDGE_CHANNELS), "key": zeros(heads, EDGE_CHANNELS, depth),
        "query": zeros(heads, EDGE_CHANNELS, depth), "scaled_coefficients": zeros(heads, EDGE_CHANNELS),
        "constant_bias": zeros(heads, TOKENS, TOKENS), "edge_read/w": zeros(heads, STATE_CHANNELS),
        "door/w": zeros(heads, STATE_CHANNELS), "gate/w": zeros(heads, STATE_CHANNELS), "gate/b": zeros(heads),
    }
    compiled = launch_attention_egt(
        output, zeros(samples, TOKENS, (4 if specialization.output_gate else 3) * width, dtype=torch.float16),
        edge_list,
        torch.zeros((samples, TOKENS, TOKENS), dtype=torch.uint64, device="cuda"),
        torch.ones((samples, TOKENS, TOKENS), dtype=norm_type, device="cuda"),
        torch.zeros((samples, round_heads_of(specialization), 3, TOKENS, TOKENS), dtype=state_type, device="cuda")
        if specialization.state_tiles
        else torch.zeros((samples, STATE_CHANNELS, TOKENS, TOKENS), dtype=state_type, device="cuda"),
        tables, specialization, logits=logits, weights=logits,
        quant_prescale=torch.ones(width, dtype=torch.float32, device="cuda") if specialization.quant_output else None,
    )
    return artifact_from_triton(compiled, grid=(grid_size(specialization), 1, 1), parameters=(_POINTER,) * 22,
                                autotuner=_attention_egt_kernel)


def _edge_list_tensors(specialization: EgtEdgeListSpecialization) -> tuple[torch.Tensor, ...]:
    _check_capacity(specialization.capacity)
    batch, capacity = specialization.batch_size, specialization.capacity
    return (torch.empty((batch, capacity), dtype=torch.int16, device="cuda"),
            torch.empty((batch, capacity), dtype=torch.int8, device="cuda"),
            torch.empty((batch, 2, CELLS), dtype=torch.int16, device="cuda"),
            torch.empty((batch,), dtype=torch.int32, device="cuda"),
            torch.zeros((batch, TOKENS), dtype=torch.int32, device="cuda"),
            torch.zeros((batch, TOKENS), dtype=torch.int32, device="cuda"),
            torch.zeros((batch, TOKENS, TOKENS), dtype=torch.uint64, device="cuda"))


def compile_egt_edge_count(specialization: EgtEdgeListSpecialization) -> KernelArtifact:
    """Autotune and compile the per-row set-bit count (one artifact per `KernelCache.get`, as `prologue_egt`)."""
    _, _, _, _, row_totals, _, edges = _edge_list_tensors(specialization)
    batch = specialization.batch_size
    compiled = _egt_edge_count_kernel[_row_grid](row_totals, edges, batch)
    return artifact_from_triton(compiled, grid=(batch * TOKENS, 1, 1), parameters=(_POINTER,) * 2,
                                autotuner=_egt_edge_count_kernel)


def compile_egt_edge_offset(specialization: EgtEdgeListSpecialization) -> KernelArtifact:
    """Autotune and compile the per-position row starts and count."""
    _, _, _, counts, row_totals, row_starts, _ = _edge_list_tensors(specialization)
    batch = specialization.batch_size
    compiled = _egt_edge_offset_kernel[_sample_grid](row_starts, counts, row_totals, batch)
    return artifact_from_triton(compiled, grid=(batch, 1, 1), parameters=(_POINTER,) * 3,
                                autotuner=_egt_edge_offset_kernel)


def compile_egt_edge_list(specialization: EgtEdgeListSpecialization) -> KernelArtifact:
    """Autotune and compile the per-row list writer."""
    cells, channels, prefix, _, _, row_starts, edges = _edge_list_tensors(specialization)
    batch = specialization.batch_size
    compiled = _egt_edge_list_kernel[_row_grid](cells, channels, prefix, edges, row_starts, batch,
                                                specialization.capacity)
    return artifact_from_triton(compiled, grid=(batch * TOKENS, 1, 1), parameters=(_POINTER,) * 5,
                                autotuner=_egt_edge_list_kernel)


def egt_edge_list(
    builder: ProgramBuilder,
    kernels: KernelCache,
    edge_list: EgtEdgeList,
    edges: Buffer,
    specialization: EgtEdgeListSpecialization,
) -> None:
    """Append the per-batch edge list: row totals, row starts and counts, then the list and the cell spans."""
    builder.set_target(lc0ex_pb2.Target.VENDOR_NVIDIA, f"sm_{specialization.architecture}")
    count = kernels.get(compile_egt_edge_count, specialization)
    offset = kernels.get(compile_egt_edge_offset, specialization)
    listing = kernels.get(compile_egt_edge_list, specialization)
    builder.call(count, edge_list.row_totals, edges, readonly=(edges,))
    builder.call(offset, edge_list.row_starts, edge_list.counts, edge_list.row_totals,
                 readonly=(edge_list.row_totals,))
    builder.call(listing, edge_list.cells, edge_list.channels, edge_list.prefix, edges, edge_list.row_starts,
                 readonly=(edges, edge_list.row_starts))


def attention_egt(  # noqa: PLR0913
    builder: ProgramBuilder,
    kernels: KernelCache,
    output: Buffer,
    qkv: Buffer,
    edge_list: EgtEdgeList,
    edges: Buffer,
    edge_norm: Buffer,
    edge_state: Buffer,
    tables: Mapping[str, Buffer],
    specialization: AttentionEgtSpecialization,
    logits: Buffer | None = None,
    quant_prescale: Buffer | None = None,
) -> None:
    """Append one EGT2 attention block. `tables` maps BLOCK_TABLES' short names to the block's persistent plans.

    `edges` is K1's packed E: the list path never reads it, the dense fallback does. `edge_state` is the state (or
    its FP16 copy) or, with `specialization.state_tiles`, the round's read tiles from `egt_state_tiles`.
    """
    builder.set_target(lc0ex_pb2.Target.VENDOR_NVIDIA, f"sm_{specialization.architecture}")
    if specialization.export_h != (logits is not None):
        message = "export_h and a logits buffer must come together"
        raise ValueError(message)
    if specialization.export_weights:
        message = "export_weights is a test switch; the served kernel never writes A"
        raise ValueError(message)
    if specialization.quant_output != (quant_prescale is not None):
        message = "quant_output and a quant_prescale buffer must come together"
        raise ValueError(message)
    kernel = kernels.get(compile_attention_egt, specialization)
    placeholder = tables["qk_scale"]  # never dereferenced without export_h / export_weights / quant_output
    reads = (qkv, edge_list.cells, edge_list.channels, edge_list.prefix, edge_list.counts, edges, edge_norm,
             edge_state, *(tables[name] for name in BLOCK_TABLES))
    arguments = (output, logits if logits is not None else placeholder, placeholder, *reads,
                 quant_prescale if quant_prescale is not None else placeholder)
    reads = (*reads, quant_prescale) if quant_prescale is not None else reads
    builder.call(kernel, *arguments, readonly=reads)
    if not specialization.overflow_exact and not specialization.overflow_only:
        # K2e: the list-only main launch is followed by the correction launch for overflow positions.
        correction = kernels.get(compile_attention_egt, replace(specialization, overflow_only=True))
        builder.call(correction, *arguments, readonly=reads)
