"""Route A': rewrite a lab export into a carrier an lc0ex artifact can load.

The lab exports a research net as an **ONNX-carrier** `Net`: an empty `weights`
stub and a 4,300-node fp32 graph whose initializers are called `const_0`,
`const_1`, and so on. `onnx-trt` builds neither of the two lab families -- egt2
refuses at 28.4 GB and static never finishes -- so this is not one route among
several; it is the only way these nets are served fast.

Round 16 proved the runtime half on BT4: an lc0ex artifact runs bit-identically
from a net whose weights arrive as ONNX initializers. What was missing is the
other half, and it is what this module is: take the export's tensors, convert
them to the FP16 the kernels read, splice the two halves of every gated FFN into
the single wide matrix the fused GLU kernel wants, and re-emit them under the
names the artifact's buffers will carry.

The output keeps everything else about the source byte for byte -- magic, format
enums, the `weights` stub -- because a carrier is a descendant of the export, not
a new file that happens to resemble it.
"""

import ctypes
import gzip
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from lczero_triton.lab._mapping import read_network
from lczero_triton.lab._names import (  # K1: EGT_*
    EGT_CODE_GROUPS,
    EGT_EDGE_CHANNELS,
    BufferPlan,
    plan_network,
)
from lczero_triton.lab._onnx import (
    FLOAT16,
    FLOAT32,
    INT32,
    UINT8,
    Graph,
    Tensor,
    encode_initializer_model,
    load_carrier,
    replace_onnx_model,
)

_LOGGER = logging.getLogger(__name__)
_F16_SIZE_BYTES = 2
_ELEMENT_BYTES = {FLOAT16: 2, FLOAT32: 4, INT32: 4, UINT8: 1}


class CarrierError(ValueError):
    """The export cannot be rewritten into an lc0ex carrier."""


@dataclass(frozen=True, slots=True)
class CarrierReport:
    """What one conversion produced, for the caller to check or print."""

    tensors: int
    parameters: int
    bytes_written: int
    maximum_relative_error: float
    architecture: object


def _as_float32(graph: Graph, name: str) -> torch.Tensor:
    """Return one source initializer as a flat FP32 tensor."""
    tensor = graph.initializers.get(name)
    if tensor is None:
        message = f"the export has no initializer called {name}"
        raise CarrierError(message)
    if tensor.data_type != FLOAT32:
        message = (
            f"{name}: expected a float32 initializer, found data type "
            f"{tensor.data_type}"
        )
        raise CarrierError(message)
    return torch.frombuffer(bytearray(tensor.raw_data), dtype=torch.float32)


def _bytes_of(values: torch.Tensor) -> bytes:
    """Return one contiguous CPU tensor's payload as bytes.

    `Tensor.numpy()` is the usual way and is unavailable: this environment has
    no NumPy, and installing one would modify a shared virtual environment.
    Reading the tensor's own storage is exact and costs one copy.
    """
    contiguous = values.contiguous()
    return ctypes.string_at(
        contiguous.data_ptr(), contiguous.numel() * contiguous.element_size()
    )


def _as_int32(graph: Graph, name: str) -> torch.Tensor:
    """Return one source integer initializer as a flat INT32 tensor."""
    tensor = graph.initializers.get(name)
    if tensor is None or tensor.data_type != INT32:
        message = f"{name}: expected an int32 initializer"
        raise CarrierError(message)
    return torch.frombuffer(bytearray(tensor.raw_data), dtype=torch.int32)


def _pair_offsets(graph: Graph, table: str, index: str) -> torch.Tensor:
    """Rebuild the export's runtime gather: `T = offset_table.T[:, relative_index]`."""
    bins, channels = graph.initializers[table].dims
    offsets = _as_float32(graph, table).reshape(bins, channels).double()
    positions = _as_int32(graph, index).to(torch.int64)
    tokens = round(positions.numel() ** 0.5)
    return offsets.T[:, positions.reshape(tokens, tokens)]


def _build_payload(
    graph: Graph, plan: BufferPlan
) -> tuple[bytes, float]:
    """Return the buffer's FP16 bytes and the worst relative rounding error."""
    if plan.source == "integer":
        tensor = graph.initializers[plan.tensors[0]]
        if tensor.data_type != INT32:
            message = f"{plan.name}: expected an int32 source"
            raise CarrierError(message)
        return tensor.raw_data, 0.0

    if plan.source == "i8_weight":
        codes, _ = _int8_site(graph, plan)
        return _bytes_of(codes), 0.0
    if plan.source == "i8_scale":
        _, scale = _int8_site(graph, plan)
        return _bytes_of(scale.to(torch.float32)), 0.0
    if plan.source == "i8_bias":
        return _bytes_of(_int8_bias(graph, plan).to(torch.float32)), 0.0

    if plan.source == "gate_pair":
        gate, up = (_as_float32(graph, name) for name in plan.tensors)
        rows = plan.shape[0]
        half = plan.shape[1] // 2
        values = torch.cat(
            (gate.reshape(rows, half), up.reshape(rows, half)), dim=1
        )
    elif plan.source == "concat_columns":
        # [in, out_i] matrices side by side along the output axis.
        rows = plan.shape[0]
        values = torch.cat([_as_float32(graph, name).reshape(rows, -1) for name in plan.tensors], dim=1)
    elif plan.source == "concat":
        values = torch.cat([_as_float32(graph, name).reshape(-1) for name in plan.tensors])
    elif plan.source == "gate_pair_padded":
        # [gate | pad | up | pad] per row: the GLU kernel reads the up half n columns
        # to the right of the gate half, so each half is padded on its own right.
        gate, up = (_as_float32(graph, name) for name in plan.tensors[:2])
        rows, half, hidden = plan.shape[0], plan.shape[1] // 2, int(plan.tensors[2])
        pad = torch.zeros(rows, half - hidden)
        values = torch.cat((gate.reshape(rows, hidden), pad, up.reshape(rows, hidden), pad), dim=1)
    elif plan.source == "gate_bias_padded":
        up = _as_float32(graph, plan.tensors[0])
        half, hidden = plan.shape[0] // 2, int(plan.tensors[1])
        values = torch.cat((torch.zeros(half), up, torch.zeros(half - hidden)))
    elif plan.source == "rows_padded":
        rows, width = plan.shape
        hidden = int(plan.tensors[1])
        down = _as_float32(graph, plan.tensors[0]).reshape(hidden, width)
        values = torch.cat((down, torch.zeros(rows - hidden, width)))
    elif plan.source == "gate_bias":
        # The gate projection has no bias in this network; a zero half is the
        # same function and, unlike an absent one, satisfies the shape check.
        up = _as_float32(graph, plan.tensors[0])
        values = torch.cat((torch.zeros_like(up), up))
    elif plan.source == "fold_scaled":
        # G = C2 @ D^T, folded in float64 so the only rounding is the final FP16.
        rows, edges = plan.shape
        head_mix = _as_float32(graph, plan.tensors[0]).double().reshape(rows, -1)
        channel_mix = _as_float32(graph, plan.tensors[1]).double().reshape(edges, -1)
        values = head_mix @ channel_mix.T
        if len(plan.tensors) > 2:
            values = values / float(plan.tensors[2])
    elif plan.source == "fold_bias":
        # K = einsum(C2, T): batch-independent, one table per head.
        rows = plan.shape[0]
        head_mix = _as_float32(graph, plan.tensors[0]).double().reshape(rows, -1)
        offsets = _pair_offsets(graph, plan.tensors[1], plan.tensors[2])
        values = torch.einsum("ht,trc->hrc", head_mix, offsets)
        if len(plan.tensors) > 3:
            values = values / float(plan.tensors[3])
    elif plan.source in ("code_table", "code_sums", "fold_scaled_codes"):
        # P6: sums over each edge code's bits of the FP16 values the per-channel kernel read.
        codes = plan.shape[-1]
        edges = codes.bit_length() - 1
        bits = ((torch.arange(codes)[:, None] >> torch.arange(edges)[None, :]) & 1).double()
        if plan.source == "fold_scaled_codes":
            heads = plan.shape[0]
            head_mix = _as_float32(graph, plan.tensors[0]).double().reshape(heads, -1)
            channel_mix = _as_float32(graph, plan.tensors[1]).double().reshape(edges, -1)
            values = (head_mix @ channel_mix.T).to(torch.float16).double() @ bits.T
        elif plan.source == "code_sums":
            per_channel = _as_float32(graph, plan.tensors[0]).to(torch.float16).double()
            values = per_channel.reshape(plan.shape[0], edges) @ bits.T
        else:
            heads, depth = plan.shape[0], plan.shape[1]
            pair = _as_float32(graph, plan.tensors[0]).to(torch.float16).double().reshape(heads, edges, depth)
            values = torch.einsum("hed,ce->hdc", pair, bits)
    elif plan.source == "egt_mix_codes":
        # K1: sums of D (first 16 columns) and p_in (last 16) over each code group's bits, in float64, the same
        # expression as `prologue_egt.fold_mix_codes` so the two agree bit for bit.
        mix = torch.cat(
            [_as_float32(graph, name).reshape(EGT_EDGE_CHANNELS, -1).double() for name in plan.tensors], dim=1
        )
        rows = []
        for first, width in EGT_CODE_GROUPS:
            codes = torch.arange(1 << width)
            bits = ((codes[:, None] >> torch.arange(width)[None, :]) & 1).double()
            rows.append(bits @ mix[first:first + width])
        values = torch.cat(rows)
    elif plan.source in ("head_scaled", "head_scaled_constant"):
        # R0: a per-head temperature [1, heads, 1, 1] times a table (or the score scale), in float64.
        heads = plan.shape[0]
        temperature = _as_float32(graph, plan.tensors[0]).double().reshape(heads, 1)
        if plan.source == "head_scaled_constant":
            values = temperature.reshape(heads) * float(plan.tensors[1])
        else:
            values = temperature * _as_float32(graph, plan.tensors[1]).double().reshape(heads, -1)
    elif plan.source == "egt_fold_scaled":
        # R0: G' = t_node (C2 @ D^T), the static fold of the pair stream tempered per head.
        heads, edges = plan.shape
        head_mix = _as_float32(graph, plan.tensors[0]).double().reshape(heads, -1)
        channel_mix = _as_float32(graph, plan.tensors[1]).double().reshape(edges, -1)
        temperature = _as_float32(graph, plan.tensors[2]).double().reshape(heads, 1)
        values = temperature * (head_mix @ channel_mix.T)
    elif plan.source == "egt_fold_bias":
        # R0: K' = t_node einsum(C2, T[relidx]).
        heads = plan.shape[0]
        head_mix = _as_float32(graph, plan.tensors[0]).double().reshape(heads, -1)
        offsets = _pair_offsets(graph, plan.tensors[1], plan.tensors[2])
        temperature = _as_float32(graph, plan.tensors[3]).double().reshape(heads, 1, 1)
        values = temperature * torch.einsum("ht,trc->hrc", head_mix, offsets)
    elif plan.source == "transpose":
        # R0: an exported [1, in, out] channel map applied transposed, served as [out, in].
        rows, columns = plan.shape
        values = _as_float32(graph, plan.tensors[0]).reshape(columns, rows).T
    elif plan.source == "zeros":
        values = torch.zeros(plan.element_count, dtype=torch.float32)
    elif plan.source == "filled":
        # Q1 probe: one constant repeated over the buffer (r = 1, m = 0 prices the conversion's shape).
        values = torch.full((plan.element_count,), float(plan.tensors[0]), dtype=torch.float32)
    elif plan.source == "quant_vector":
        values = _quant_vector(plan)
    elif plan.source == "d1_refit":
        values = _d1_refit(graph, plan)
    elif plan.source == "constant":
        values = torch.tensor([float(plan.tensors[0])], dtype=torch.float32)
    elif plan.source == "relu":
        values = torch.relu(_as_float32(graph, plan.tensors[0]))
    elif plan.source == "offsets":
        values = _pair_offsets(graph, plan.tensors[0], plan.tensors[1])
    else:
        values = _as_float32(graph, plan.tensors[0])

    values = values.reshape(-1)
    if values.numel() != plan.element_count:
        message = (
            f"{plan.name}: source holds {values.numel()} elements, the buffer "
            f"declares {plan.element_count}"
        )
        raise CarrierError(message)
    if plan.data_type == FLOAT32:
        return _bytes_of(values.to(torch.float32)), 0.0
    converted = values.to(torch.float16)
    if not bool(torch.isfinite(converted).all()):
        message = (
            f"{plan.name}: a weight overflowed FP16 -- this network cannot be "
            "served at this precision without rescaling"
        )
        raise CarrierError(message)
    return _bytes_of(converted), _relative_error(values, converted)


_SIGNIFICANT_MAGNITUDE = 1e-3


def _quant_vector(plan: BufferPlan) -> torch.Tensor:
    """Q1: one of the analyser's per-channel vectors, from the file `plan_network` already validated.

    `m` is optional and refused by measurement on this family (the 09-21 SmoothQuant read), so a file without it
    fills zeros -- which is the identity, not a guess.
    """
    from lczero_triton.lab._quant import loaded_prescale  # noqa: PLC0415

    scope, site, kind = plan.tensors
    kind, _, padded = kind.partition(":")  # round 26b: `r:<padded>` -- the dff padded to CUTLASS's alignment
    loaded = loaded_prescale()
    if loaded is None:
        message = f"{plan.name}: no quantiser vectors were loaded; set LC0EX_QUANT_PRESCALE before planning"
        raise CarrierError(message)
    vectors = loaded.get(scope, site)
    if vectors is None:
        message = f"{plan.name}: {scope}/{site} is not in {loaded.path.name}"
        raise CarrierError(message)
    if kind == "m":
        return torch.tensor(vectors.offset or (0.0,) * vectors.channels, dtype=torch.float32)
    values = torch.tensor(vectors.prescale, dtype=torch.float32)
    if padded:
        # The padded hidden channels are always 0 (all-zero GLU pairs); any positive r keeps them 0.
        values = torch.cat((values, torch.ones(int(padded) - values.numel(), dtype=torch.float32)))
    return values


# One site's folded int8 weights and scale, computed once for its two plans (`/w` and `/scale`).
_INT8_SITES: dict[tuple[str, ...], tuple[torch.Tensor, torch.Tensor]] = {}


def _site_matrix(graph: Graph, layout: str, weights: tuple[str, ...]) -> torch.Tensor:
    """The site's float64 `[k, n]` weight in the GEMM's column order: concatenated, or gate/up interleaved pairs.

    Round 26b: `<layout>:<padded>` zero-pads the FFN hidden axis to `padded` -- the pairs' axis of `glu_pairs`
    (all-zero pairs), the input rows of FFN2's `concat`.
    """
    layout, _, padded = layout.partition(":")
    matrices = [_as_float32(graph, name).double() for name in weights]
    rows = graph.initializers[weights[0]].dims[0]
    matrices = [matrix.reshape(rows, -1) for matrix in matrices]
    if layout == "concat":
        matrix = torch.cat(matrices, dim=1)
        if padded:
            matrix = torch.cat((matrix, torch.zeros(int(padded) - rows, matrix.shape[1], dtype=matrix.dtype)))
        return matrix
    if layout == "glu_pairs":
        gate, up = matrices
        pairs = torch.stack((gate, up), dim=2)
        if padded:
            pairs = torch.cat((pairs, torch.zeros(rows, int(padded) - pairs.shape[1], 2, dtype=pairs.dtype)), dim=1)
        return pairs.reshape(rows, -1)
    message = f"unknown int8 site layout {layout!r}"
    raise CarrierError(message)


def _residual_alpha(graph: Graph, alpha: str) -> float:
    """The block's residual scale (a one-element initializer), or 1 for a site without one."""
    if not alpha:
        return 1.0
    values = _as_float32(graph, alpha)
    if values.numel() != 1:
        message = f"{alpha}: a residual alpha has one element, found {values.numel()}"
        raise CarrierError(message)
    return float(values[0])


def _int8_site(graph: Graph, plan: BufferPlan) -> tuple[torch.Tensor, torch.Tensor]:
    """Q1: `W'[j, n] = s_j W[j, n]` quantised per output column -> (int8 `[n, k]` codes, float64 `[n]` scale).

    `w[n] = max_j |W'[j, n]| / 127` and the codes round half to even (the analyser's `round`); the served scale is
    `D * w[n] * alpha`, the activation step D coming from the same file as `s` (`r_j s_j = 1 / D`).
    """
    key = plan.tensors
    cached = _INT8_SITES.get(key)
    if cached is not None:
        return cached
    from lczero_triton.lab._quant import loaded_prescale  # noqa: PLC0415

    scope, site, layout, alpha, *weights = plan.tensors
    loaded = loaded_prescale()
    vectors = None if loaded is None else loaded.get(scope, site)
    if vectors is None:
        message = f"{plan.name}: no vectors for {scope}/{site}; set LC0EX_QUANT_PRESCALE to the analyser's .npz"
        raise CarrierError(message)
    matrix = _site_matrix(graph, layout, tuple(weights))
    smoothing = torch.tensor(vectors.smoothing or (), dtype=torch.float64)
    if 0 < smoothing.numel() < matrix.shape[0] and ":" in layout:
        # Round 26b: FFN2's zero-padded input rows take s = 1 (their weights are zero, their codes stay zero).
        smoothing = torch.cat((smoothing, torch.ones(matrix.shape[0] - smoothing.numel(), dtype=torch.float64)))
    if smoothing.numel() != matrix.shape[0]:
        message = f"{plan.name}: 's' has {smoothing.numel()} channels, the weight's input axis {matrix.shape[0]}"
        raise CarrierError(message)
    folded = matrix * smoothing[:, None]
    step = folded.abs().amax(dim=0) / 127.0
    step = torch.where(step > 0.0, step, torch.ones_like(step))  # an all-zero column quantises to zeros at any step
    codes = torch.clamp(torch.round(folded / step[None, :]), -127.0, 127.0).to(torch.int8).T.contiguous()
    scale = step * vectors.step * _residual_alpha(graph, alpha)
    _INT8_SITES[key] = (codes, scale)
    return codes, scale


def _int8_bias(graph: Graph, plan: BufferPlan) -> torch.Tensor:
    """Q1: a site's bias in its GEMM's column order, times the residual alpha where the site has one."""
    layout, alpha, *biases = plan.tensors
    layout, _, padded = layout.partition(":")  # round 26b: only the GLU pairs' bias is padded (FFN2's is over d_model)
    vectors = [_as_float32(graph, name).double() for name in biases]
    if layout == "concat":
        bias = torch.cat(vectors)
    elif layout == "glu_pairs":
        up = vectors[0]  # the gate projection has no bias: its half of every pair is zero
        pairs = torch.stack((torch.zeros_like(up), up), dim=1)
        if padded:
            pairs = torch.cat((pairs, torch.zeros(int(padded) - pairs.shape[0], 2, dtype=pairs.dtype)))
        bias = pairs.reshape(-1)
    else:
        message = f"unknown int8 site layout {layout!r}"
        raise CarrierError(message)
    return bias * _residual_alpha(graph, alpha)


def _read_npz_matrix(path: Path, key: str) -> torch.Tensor:
    """One little-endian float32/64 C-order member of an `.npz` (1-D or 2-D) -- no numpy on the serving fleet."""
    import ast  # noqa: PLC0415
    import struct  # noqa: PLC0415
    import zipfile  # noqa: PLC0415

    with zipfile.ZipFile(path) as archive:
        payload = archive.read(f"{key}.npy")
    major = payload[6]
    width = {1: 2, 2: 4}[major]
    length = int.from_bytes(payload[8:8 + width], "little")
    header = ast.literal_eval(payload[8 + width:8 + width + length].decode("latin-1").strip())
    descr, shape = header["descr"], tuple(header["shape"])
    if header.get("fortran_order") or descr not in ("<f4", "<f8"):
        message = f"{path.name}:{key}: {descr} fortran={header.get('fortran_order')}; want C-order float32/64"
        raise CarrierError(message)
    count = 1
    for dimension in shape:
        count *= dimension
    size, code = (4, "f") if descr == "<f4" else (8, "d")
    data = payload[8 + width + length:][: count * size]
    return torch.tensor(struct.unpack(f"<{count}{code}", data), dtype=torch.float64).reshape(shape)


def _d1_refit(graph: Graph, plan: BufferPlan) -> torch.Tensor:
    """Q1: the analyser's D1 value layer (`W` [128, 3] or `b` [3]), after checking its `W0`/`b0` is this export's.

    The refit is fitted against ONE net's int8 trunk; its trained starting point must be the layer this export
    carries, or the substitution corrects some other net.
    """
    from lczero_triton.lab._names import quant_d1_source  # noqa: PLC0415

    key, trained = plan.tensors
    path = Path(quant_d1_source())
    values = _read_npz_matrix(path, key)
    start = _read_npz_matrix(path, f"{key}0").reshape(-1)
    carried = _as_float32(graph, trained).double()
    if start.numel() != carried.numel() or float((start - carried).abs().max()) > 1e-5:
        message = (f"{plan.name}: {path.name}'s {key}0 is not this export's {trained} "
                   f"(max |diff| {float((start - carried).abs().max()) if start.numel() == carried.numel() else 'shape'}); "
                   "the refit belongs to another net or head")
        raise CarrierError(message)
    _LOGGER.info("D1: %s from %s, max |%s - %s0| = %.3g", plan.name, path.name, key, key,
                 float((values.reshape(-1) - start).abs().max()))
    return values


def _relative_error(source: torch.Tensor, converted: torch.Tensor) -> float:
    """Return the worst relative error over weights large enough to have one.

    FP16 keeps ~11 bits of mantissa, so ~5e-4 is the floor for a normal number
    and a result near it is the conversion working. The measurement is taken
    only over weights above `_SIGNIFICANT_MAGNITUDE`: below that the format runs
    out of exponent and a weight of 2e-5 lands on the nearest subnormal, giving
    a relative error of a few percent that says nothing about the conversion and
    everything about the size of the weight. Those weights contribute nothing to
    any dot product either.

    Overflow is the failure that would matter, and it is checked separately by
    the caller, on every element rather than on this subset.
    """
    significant = source.abs() >= _SIGNIFICANT_MAGNITUDE
    if not bool(significant.any()):
        return 0.0
    difference = (converted.to(torch.float32) - source).abs()[significant]
    return float((difference / source.abs()[significant]).max())


def prologue_egt_tables(graph: Graph, plans: list[BufferPlan]) -> dict[str, torch.Tensor]:
    """K1: build the EGT2 prologue's FP32 tables from an export graph, one tensor per plan.

    `plans` come from `_names.plan_prologue_egt`. Since R0, `convert` writes the same tables for an EGT2 export
    (`plan_network` plans them); this function stays the tests' and a builder's direct route to them.
    """
    tables = {}
    for plan in plans:
        if plan.data_type != FLOAT32:
            message = f"{plan.name}: the EGT2 prologue tables are FP32"
            raise CarrierError(message)
        payload, _ = _build_payload(graph, plan)
        tables[plan.name] = torch.frombuffer(bytearray(payload), dtype=torch.float32).view(plan.shape)
    return tables


def convert(source: Path, destination: Path) -> CarrierReport:
    """Rewrite one lab export as an lc0ex carrier and return what it wrote."""
    _LOGGER.info("reading lab export %s", source)
    encoded, graph = load_carrier(source)
    network = read_network(graph)
    _LOGGER.info(
        "recovered %d blocks, d_model %d, %d heads, %d edge channels",
        network.architecture.blocks,
        network.architecture.d_model,
        network.architecture.heads,
        network.architecture.edge_channels,
    )

    plans = plan_network(network)
    seen: set[str] = set()
    tensors: list[Tensor] = []
    worst = 0.0
    for plan in plans:
        if plan.name in seen:
            message = f"two buffers are both called {plan.name}"
            raise CarrierError(message)
        seen.add(plan.name)
        payload, error = _build_payload(graph, plan)
        expected = plan.element_count * _ELEMENT_BYTES[plan.data_type]
        if len(payload) != expected:
            message = (
                f"{plan.name}: produced {len(payload)} bytes, the buffer needs "
                f"{expected}"
            )
            raise CarrierError(message)
        worst = max(worst, error)
        tensors.append(
            Tensor(
                name=plan.name,
                data_type=plan.data_type,
                dims=plan.shape,
                raw_data=payload,
            )
        )

    model = encode_initializer_model(tensors, graph_name="lab")
    _LOGGER.info("writing carrier %s", destination)
    with gzip.open(destination, "wb", compresslevel=1) as target:
        target.write(replace_onnx_model(encoded, model))
    return CarrierReport(
        tensors=len(tensors),
        parameters=sum(plan.element_count for plan in plans),
        bytes_written=destination.stat().st_size,
        maximum_relative_error=worst,
        architecture=network.architecture,
    )
