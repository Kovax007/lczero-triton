#!/usr/bin/env python3
"""B3 (2026-09-28): write the carrier an int8 BT4 artifact (and its FP16 twin) load.

A classic BT4 `.pb.gz` is served by lc0ex through the ONNX initializers lc0 converts it to in process
(`network_lc0ex_cuda.cc` `UploadWeights` / `MakeConverterOptions`: opset 17, FP16, the winner value head). A net that
already carries an `onnx_model` is not converted, so the carrier is:

* the classic net, byte for byte (the artifact's fingerprint reads its format and its weights' structure), plus
* `onnx_model` = an initializer-only model holding every initializer of lc0's own conversion, unchanged (the FP16
  twin's buffers), plus the int8 sites' `w` (UINT8 `[n, k]`), `scale`, `bias` (FP32 `[n]`) and `r` (FP32 `[k]`)
  under `bt4/_q1.site_names`, plus -- with `--d1` -- the analyser's refit of the value layer under `_q1.D1_TARGETS`
  + `_q1.D1_SUFFIX` (the int8 artifact built with LC0EX_QUANT_D1 reads it there; the twin keeps the trained layer).

The int8 math is `lab/carrier.py`'s, unchanged: `W'[j, n] = s_j W[j, n]` quantised per output column
(`w[n] = max_j |W'[j, n]| / 127`, codes rounded half to even), served scale `D * w[n] * alpha`, bias `b * alpha`,
alpha the site's residual scale as the FP16 epilogue applies it (`act(acc + b) * alpha + skip`).

The ONNX comes from lc0 itself, e.g.
    lc0 leela2onnx --input=NET.pb.gz --output=NET_f16.onnx --onnx-data-type=f16 --onnx-opset=17 \\
        --value-head=winner --policy-head=vanilla
and, for the int8 fold, optionally the same at `--onnx-data-type=f32` (the weights before FP16 rounding; the analyser
fitted its vectors against lc0's FP32 export). Gate G0 proves the first half: the FP16 artifact must read
bit-identically from NET.pb.gz and from the carrier.

usage: make_carrier_bt4_q1.py NET.pb.gz NET_f16.onnx VECTORS.npz OUT.pb.gz [--f32 NET_f32.onnx] [--d1 D1.npz]
"""

import argparse
import gzip
import hashlib
import sys
from pathlib import Path

import torch

from lczero_triton.bt4 import _q1
from lczero_triton.bt4._format import load_network
from lczero_triton.lab._onnx import FLOAT16, FLOAT32, UINT8, Tensor, encode_initializer_model, parse_model
from lczero_triton.lab._quant import load_prescale

_F16 = 2  # bytes
_NET_ONNX_MODEL = 11
_ONNX_MODEL_MODEL, _ONNX_MODEL_DATA_TYPE = 1, 2
_ONNX_FLOAT16 = 10


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _field(number: int, payload: bytes) -> bytes:
    return _varint((number << 3) | 2) + _varint(len(payload)) + payload


def _values(tensor: Tensor) -> torch.Tensor:
    """One FP16 or FP32 initializer as float64."""
    dtype = {FLOAT16: torch.float16, FLOAT32: torch.float32}.get(tensor.data_type)
    if dtype is None:
        message = f"{tensor.name}: data type {tensor.data_type}, expected FP16 or FP32"
        raise SystemExit(message)
    return torch.frombuffer(bytearray(tensor.raw_data), dtype=dtype).double()


def _raw(values: torch.Tensor) -> bytes:
    """A contiguous CPU tensor's payload (no numpy on the serving fleet; this is one copy)."""
    import ctypes  # noqa: PLC0415

    contiguous = values.contiguous()
    return ctypes.string_at(contiguous.data_ptr(), contiguous.numel() * contiguous.element_size())


def _matrix(initializers: dict[str, Tensor], name: str) -> torch.Tensor:
    tensor = initializers.get(name)
    if tensor is None:
        message = f"the ONNX has no initializer {name}"
        raise SystemExit(message)
    if len(tensor.dims) != 2:  # noqa: PLR2004
        message = f"{name}: dims {tensor.dims}, expected a [k, n] matrix"
        raise SystemExit(message)
    return _values(tensor).reshape(tensor.dims)


def _vector(initializers: dict[str, Tensor], name: str) -> torch.Tensor:
    tensor = initializers.get(name)
    if tensor is None:
        message = f"the ONNX has no initializer {name}"
        raise SystemExit(message)
    return _values(tensor).reshape(-1)


def _site_tensors(source: dict[str, Tensor], prefix: str, site: str, vectors) -> list[Tensor]:  # noqa: ANN001
    """`w`, `scale`, `bias`, `r` of one int8 site (`lab/carrier.py`'s `_int8_site` / `_int8_bias` arithmetic)."""
    spec = _q1.site_sources(prefix)[site]
    matrix = torch.cat([_matrix(source, name) for name in spec.weights], dim=1)
    bias = torch.cat([_vector(source, name) for name in spec.biases])
    alpha = 1.0
    if spec.alpha:
        values = _vector(source, spec.alpha)
        if values.numel() != 1:
            message = f"{spec.alpha}: a residual alpha has one element, found {values.numel()}"
            raise SystemExit(message)
        alpha = float(values[0])
    smoothing = torch.tensor(vectors.smoothing, dtype=torch.float64)
    if smoothing.numel() != matrix.shape[0]:
        message = f"{prefix}/{site}: 's' has {smoothing.numel()} channels, the weight's input axis {matrix.shape[0]}"
        raise SystemExit(message)
    folded = matrix * smoothing[:, None]
    step = folded.abs().amax(dim=0) / 127.0
    step = torch.where(step > 0.0, step, torch.ones_like(step))
    codes = torch.clamp(torch.round(folded / step[None, :]), -127.0, 127.0).to(torch.int8).T.contiguous()
    scale = (step * vectors.step * alpha).to(torch.float32)
    bias = (bias * alpha).to(torch.float32)
    prescale = torch.tensor(vectors.prescale, dtype=torch.float32)
    names = _q1.site_names(prefix, site)
    n, k = codes.shape
    return [
        Tensor(names["w"], UINT8, (n, k), _raw(codes)),
        Tensor(names["scale"], FLOAT32, (n,), _raw(scale)),
        Tensor(names["bias"], FLOAT32, (n,), _raw(bias)),
        Tensor(names["r"], FLOAT32, (k,), _raw(prescale)),
    ]


def _read_npz(path: Path, key: str) -> torch.Tensor:
    from lczero_triton.lab.carrier import _read_npz_matrix  # noqa: PLC0415

    return _read_npz_matrix(path, key)


def _d1_tensors(served: dict[str, Tensor], exact: dict[str, Tensor], path: Path) -> list[Tensor]:
    """The analyser's value layer under the D1 names, FP16 like the layer it replaces, after checking its W0 / b0."""
    out = []
    for key, target in _q1.D1_TARGETS.items():
        values = _read_npz(path, key)
        start = _read_npz(path, f"{key}0").reshape(-1)
        carried = _vector(exact, target)
        tolerance = 1e-5 if exact[target].data_type == FLOAT32 else 2e-3 * float(carried.abs().max()) + 1e-6
        if start.numel() != carried.numel() or float((start - carried).abs().max()) > tolerance:
            message = f"{path.name}: {key}0 is not this net's {target}; the refit belongs to another net or head"
            raise SystemExit(message)
        reference = served[target]
        if values.numel() != reference.element_count:
            message = f"{path.name}: {key} has {values.numel()} values, {target} {reference.element_count}"
            raise SystemExit(message)
        print(f"# D1 {key}: max |{key} - {key}0| = {float((values.reshape(-1) - start).abs().max()):.4g}")
        out.append(Tensor(target + _q1.D1_SUFFIX, FLOAT16, reference.dims, _raw(values.to(torch.float16))))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("net", type=Path)
    parser.add_argument("onnx_f16", type=Path)
    parser.add_argument("vectors", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--f32", type=Path, help="lc0's FP32 conversion of the same net, for the int8 fold")
    parser.add_argument("--d1", type=Path, help="the analyser's d1full_W_*.npz for the served (winner) head")
    args = parser.parse_args()
    if args.output.exists():
        sys.exit(f"{args.output} exists; refusing to overwrite")

    network = load_network(args.net)
    if network.HasField("onnx_model"):
        sys.exit(f"{args.net.name} already carries an onnx_model; a BT4 carrier starts from the classic net")
    blocks = len(network.weights.encoder)
    d_model = len(network.weights.encoder[0].ln1_gammas.params) // _F16
    hidden = len(network.weights.encoder[0].ffn.dense1_b.params) // _F16
    vectors = load_prescale(args.vectors, blocks=blocks, widths=_q1.site_widths(d_model, hidden))
    plan = _q1.check_blocks(vectors, blocks)

    served = parse_model(args.onnx_f16.read_bytes()).initializers
    for name, tensor in served.items():
        if tensor.data_type == FLOAT32 and len(tensor.raw_data) > 64:  # noqa: PLR2004
            print(f"# note: {name} is FP32 in the FP16 conversion")
    exact = parse_model(args.f32.read_bytes()).initializers if args.f32 else served

    tensors = list(served.values())
    count = 0
    for index, sites in plan.items():
        for site in sorted(sites):
            tensors += _site_tensors(exact, f"/encoder{index}", site, vectors.get(f"encoder{index}", site))
            count += 1
    if args.d1:
        tensors += _d1_tensors(served, exact, args.d1)
    names = [tensor.name for tensor in tensors]
    if len(names) != len(set(names)):
        sys.exit("duplicate initializer names in the carrier")

    model = encode_initializer_model(tensors, graph_name="bt4_q1_carrier")
    with gzip.open(args.net, "rb") as source:
        classic = source.read()
    onnx_model = _field(_ONNX_MODEL_MODEL, model) + _varint((_ONNX_MODEL_DATA_TYPE << 3) | 0) + _varint(_ONNX_FLOAT16)
    carrier = classic + _field(_NET_ONNX_MODEL, onnx_model)
    with gzip.open(args.output, "wb", compresslevel=6) as sink:
        sink.write(carrier)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()[:8]  # noqa: E731
    print(f"# carrier {args.output.name} {digest(args.output)}: net {digest(args.net)}, vectors {vectors.digest} "
          f"({count} int8 sites of {4 * blocks}), onnx {digest(args.onnx_f16)}"
          + (f", fold from {digest(args.f32)}" if args.f32 else ", fold from the FP16 conversion")
          + (f", D1 {digest(args.d1)}" if args.d1 else ", no D1")
          + f"; {len(served)} converter initializers + {len(tensors) - len(served)} added")


if __name__ == "__main__":
    main()
