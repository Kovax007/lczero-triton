#!/usr/bin/env python3
"""Write the lc0ex weights file of a LAB net (the BT6 flagship) WITH the weights of an ERROR value head (backend 10-13).

    weights file = a lab weights file as it is served (`make_carrier.py` / the int8 files' maker on the ORDINARY export;
                   the lab reader never sees the error head)
                 + the six `/value_error/` initializers a program built with LC0EX_LAB_ERROR_HEAD=1 declares
                   (`lab/network.py: error_head_plans`), taken from an export that carries `/output/err`
                   (rig2 `export512_with_error_head_output_option_1013.py --error-head NAME`)

    /value_error/embedding/w [trunk, 128]   /value_error/embedding/b [128]      per-square projection (Mish)
    /value_error/dense1/w    [64*128, 128]  /value_error/dense1/b    [128]      dense on the flattened squares (Mish)
    /value_error/dense2/w    [128, 1]       /value_error/dense2/b    [1]        the error layer (sigmoid in the program)

All six are FLOAT16 (as the `/value` head's weights are). WHICH head's error output the program serves is decided by
this file alone: the tensors are those of the head the export was made with (`st` for TCEC season-30's search).
Every byte of the source weights file is kept: its Net fields, its onnx_model fields, its model's fields and its
initializers; the six tensors are appended to the graph. NO onnx_model field is added (field 8 means `output_err` in
lc0's TCEC season-30 tree and `output_childq_mean` in the lab's fleet-line tree; the lc0ex backend finds `/output/err`
as a program buffer). A program with the output refuses a file without the six tensors ("The lc0ex buffer
'/value_error/...' has no corresponding ONNX initializer"); a program without the output loads this file and prints
one warning per tensor.

The head's weights are found STRUCTURALLY in the export's graph, never by initializer name (the exporter numbers
constants in first-use order). Premises, each asserted (anything else aborts):
  * `/output/err` is produced by a Sigmoid (through Identity / Cast / Reshape only);
  * the nodes only `/output/err` depends on (its cone minus the cone of every other graph output) hold exactly three
    Gemms with alpha = beta = 1, no transposes: embedding [trunk, 128], dense1 [64 * 128, 128], error [128, 1], chained
    in that order, each next one reading a Mish (a Mul of the previous Gemm's output, through Reshape / Identity, with a
    Tanh) of the previous one;
  * the head's frontier is ONE trunk tensor (plus Shape nodes of that same tensor), and a [trunk, 128] Gemm of
    `/output/wdl`'s cone reads the very same tensor (the error head reads what the served value head reads) with
    DIFFERENT weights (the error head's body is its own, as in lc0's converter.cc when error_head != value_head);
  * the source weights file's `/value/embedding/w` has that trunk width and holds no `/value_error/` tensor yet.

usage: make_lab_error_head_carrier.py --plain WEIGHTS_FILE.pb.gz --error-net EXPORT_WITH_ERR.pb.gz --out NEW.pb.gz
       make_lab_error_head_carrier.py --error-net EXPORT_WITH_ERR.pb.gz --digest     (the six tensors' sha256 only)
       --zero-error-layer   CONTROL file: the error layer's weight and bias are written as zeros (the program then
                            serves e = 0.5 for every position); never serve such a file
Run with PYTHONPATH on this tree's packages (lczero_triton.bt4._onnx_wire). CPU only, no numpy needed.
"""

import argparse
import array
import gzip
import hashlib
import math
import os
import struct
import sys
from collections import Counter, defaultdict

from lczero_triton.bt4 import _onnx_wire as wire

_NET_ONNX_MODEL = 11
_ONNX_MODEL_MODEL = 1
_MODEL_GRAPH = 7
_GRAPH_INITIALIZER = 5
ONNX_FLOAT16 = 10
PREFIX = "/value_error"
OUTPUT = "/output/err"
WIDTH = 128
SQUARES = 64
_THROUGH = ("Identity", "Cast", "Reshape", "Flatten")


def sha8_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()[:8]


def abort(message):
    raise SystemExit(f"ABORT: {message}")


def only(items, what):
    items = list(items)
    if len(items) != 1:
        abort(f"{what}: {len(items)} candidates, expected exactly one")
    return items[0]


class Graph:
    def __init__(self, nodes, initializers):
        self.nodes, self.inits = nodes, initializers
        self.producer = {output: node for node in nodes for output in node.outputs}
        self.consumers = defaultdict(list)
        for node in nodes:
            for name in node.inputs:
                self.consumers[name].append(node)

    def cone(self, names):
        seen, stack, tensors = set(), list(names), set()
        while stack:
            tensor = stack.pop()
            if tensor in tensors:
                continue
            tensors.add(tensor)
            node = self.producer.get(tensor)
            if node is None or node.index in seen:
                continue
            seen.add(node.index)
            stack.extend(node.inputs)
        return seen

    def up(self, tensor, through=_THROUGH):
        node = self.producer.get(tensor)
        while node is not None and node.op in through:
            tensor = node.inputs[0]
            node = self.producer.get(tensor)
        return tensor


def plain_gemm(g, node):
    attrs = {k: node.attrs.get(k, d) for k, d in (("alpha", 1.0), ("beta", 1.0), ("transA", 0), ("transB", 0))}
    if attrs != {"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0}:
        abort(f"{node.name}: Gemm attributes {attrs}")
    if len(node.inputs) != 3 or node.inputs[1] not in g.inits or node.inputs[2] not in g.inits:
        abort(f"{node.name}: Gemm weight or bias is not an initializer ({node.inputs})")


def mish_source(g, tensor, ids):
    """`tensor` (through Reshape / Identity) must be Mul(x, tanh(.)) of the head; -> x (through Reshape / Identity)."""
    mul = g.producer.get(g.up(tensor))
    if mul is None or mul.op != "Mul" or mul.index not in ids:
        abort(f"{tensor}: expected a Mish (Mul with a Tanh) of the head, found {mul.op if mul else None}")
    tanh = [i for i in mul.inputs if g.producer.get(i) is not None and g.producer[i].op == "Tanh"]
    rest = [i for i in mul.inputs if i not in tanh]
    if len(tanh) != 1 or len(rest) != 1:
        abort(f"{mul.name}: not Mul(x, Tanh(..)) ({[g.producer[i].op if i in g.producer else 'init' for i in mul.inputs]})")
    return g.up(rest[0])


def extract_head(g, outputs, report):
    """-> (trunk width, {buffer name: source initializer}); aborts on any premise that does not hold."""
    if OUTPUT not in g.producer:
        abort(f"the graph does not produce {OUTPUT}: this is not an error-head export (outputs {outputs})")
    others = set()
    for output in outputs:
        if output != OUTPUT and output in g.producer:
            others |= g.cone([output])
    in_cone = g.cone([OUTPUT])
    head = [n for n in g.nodes if n.index in in_cone and n.index not in others]
    ids = {n.index for n in head}
    frontier = sorted({i for n in head for i in n.inputs if i in g.producer and g.producer[i].index not in ids})
    shapes = [t for t in frontier if g.producer[t].op == "Shape"]
    trunk = only([t for t in frontier if t not in shapes], "trunk tensors the error head reads")
    for t in shapes:
        if list(g.producer[t].inputs) != [trunk]:
            abort(f"frontier tensor {t} is a Shape of {g.producer[t].inputs}, not of the trunk tensor {trunk}")
    ops = Counter(n.op for n in head)
    report(f"error head: {len(head)} exclusive nodes {dict(sorted(ops.items()))}; reads trunk tensor {trunk} "
           f"({g.producer[trunk].op} of {g.producer[trunk].inputs[0]}) and {len(shapes)} Shape(s) of it")

    sigmoid = g.producer[g.up(OUTPUT)] if g.up(OUTPUT) in g.producer else None
    if sigmoid is None or sigmoid.op != "Sigmoid" or sigmoid.index not in ids:
        abort(f"{OUTPUT} is produced by {sigmoid.op if sigmoid else None}, not by a Sigmoid of the head")
    gemms = [n for n in head if n.op == "Gemm"]
    if len(gemms) != 3 or ops.get("Sigmoid") != 1 or ops.get("Tanh") != 2:
        abort(f"the head has {len(gemms)} Gemms, {ops.get('Sigmoid')} Sigmoid, {ops.get('Tanh')} Tanh; expected 3 / 1 / 2")
    for node in gemms:
        plain_gemm(g, node)
    error = g.producer.get(g.up(sigmoid.inputs[0]))
    if error not in gemms:
        abort("the Sigmoid's input is not a Gemm of the head")
    dense1 = g.producer.get(mish_source(g, error.inputs[0], ids))
    if dense1 not in gemms or dense1 is error:
        abort("the error layer does not read Mish(dense1)")
    embedding = g.producer.get(mish_source(g, dense1.inputs[0], ids))
    if embedding not in gemms or embedding in (error, dense1):
        abort("dense1 does not read Mish(embedding)")
    if g.up(embedding.inputs[0]) != g.up(trunk):
        abort(f"the embedding reads {g.up(embedding.inputs[0])}, not the trunk tensor {trunk} ({g.up(trunk)})")
    trunk_width = g.inits[embedding.inputs[1]].dims[0]
    expected = {"embedding": (embedding, [trunk_width, WIDTH], [WIDTH]),
                "dense1": (dense1, [SQUARES * WIDTH, WIDTH], [WIDTH]),
                "dense2": (error, [WIDTH, 1], [1])}
    sources = {}
    for layer, (node, w_dims, b_dims) in expected.items():
        w, b = g.inits[node.inputs[1]], g.inits[node.inputs[2]]
        if w.dtype != 1 or list(w.dims) != w_dims or b.dtype != 1 or list(b.dims) != b_dims:
            abort(f"{layer}: weight {w.describe()}, bias {b.describe()}; expected FLOAT {w_dims} and {b_dims}")
        sources[f"{PREFIX}/{layer}/w"], sources[f"{PREFIX}/{layer}/b"] = w, b

    # The served value head reads the same tensor, with weights of its own.
    wdl_ids = g.cone(["/output/wdl"]) if "/output/wdl" in g.producer else set()
    served = [n for n in g.nodes if n.index in wdl_ids and n.index not in ids and n.op == "Gemm"
              and g.up(n.inputs[0]) == g.up(trunk) and n.inputs[1] in g.inits
              and list(g.inits[n.inputs[1]].dims) == [trunk_width, WIDTH]]
    if not served:
        abort(f"no [{trunk_width}, {WIDTH}] Gemm of /output/wdl's cone reads the error head's trunk tensor {trunk}")
    if any(n.inputs[1] == embedding.inputs[1] for n in served):
        abort("the error head's embedding IS the served value head's: this export's error head is the served head")
    report(f"attach point: the trunk tensor {trunk} is also read by the served value head's embedding "
           f"({', '.join(n.inputs[1] for n in served)}); the error head's embedding is its own ({embedding.inputs[1]})")
    return trunk_width, sources


def encode(sources, report, zero_error_layer):
    """-> ([(name, TensorProto bytes)], {name: float values as carried})."""
    out, carried = [], {}
    for name, tensor in sources.items():
        values = list(tensor.values())
        dims = list(tensor.dims)
        if len(values) != math.prod(dims):
            abort(f"{name}: {len(values)} values for dims {dims}")
        zeroed = zero_error_layer and name.startswith(f"{PREFIX}/dense2/")
        if zeroed:
            values = [0.0] * len(values)
        try:
            raw = struct.pack(f"<{len(values)}e", *values)
        except OverflowError:
            abort(f"{name}: a value overflows float16")
        back = struct.unpack(f"<{len(values)}e", raw)
        worst = max(abs(a - b) for a, b in zip(values, back))
        rel = max((abs(a - b) / abs(a) for a, b in zip(values, back) if abs(a) >= 1e-3), default=0.0)
        peak = max(abs(v) for v in values)
        carried[name] = back
        report(f"  {name}: FLOAT16 {dims} from {tensor.name}{' ZEROED (control file)' if zeroed else ''}; max |w| {peak:.4g}; "
               f"|fp16 - fp32| max {worst:.3e}, relative over |w| >= 1e-3 max {rel:.3e}; sha256 of the carried bytes "
               f"{hashlib.sha256(raw).hexdigest()[:16]}")
        out.append((name, wire.enc_tensor(name, ONNX_FLOAT16, dims, raw)))
    return out, carried


def spans(buffer):
    """(field, wire type, start, payload start, end) of every field of one message."""
    i, n, out = 0, len(buffer), []
    while i < n:
        start = i
        key, i = wire.varint(buffer, i)
        fno, wt = key >> 3, key & 7
        if wt == 0:
            _, i = wire.varint(buffer, i)
            out.append((fno, wt, start, start, i))
        elif wt == 1:
            out.append((fno, wt, start, i, i + 8))
            i += 8
        elif wt == 5:
            out.append((fno, wt, start, i, i + 4))
            i += 4
        elif wt == 2:
            length, payload = wire.varint(buffer, i)
            i = payload + length
            out.append((fno, wt, start, payload, i))
        else:
            abort(f"wire type {wt} at byte {start}")
    return out


def one(buffer, number, what):
    hits = [s for s in spans(buffer) if s[0] == number and s[1] == 2]
    if len(hits) != 1:
        abort(f"{what}: {len(hits)} fields {number}, expected exactly one")
    return hits[0]


def replace(buffer, span, payload):
    _, _, start, _, end = span
    return bytes(buffer[:start]) + wire.enc_bytes_field(span[0], payload) + bytes(buffer[end:])


def raw_bytes(value):
    return bytes(value) if hasattr(value, "tobytes") else value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", help="the lab weights file to extend (FP16 or int8; nothing of it is changed)")
    ap.add_argument("--error-net", required=True, help="the export WITH /output/err (.pb.gz)")
    ap.add_argument("--out")
    ap.add_argument("--digest", action="store_true", help="print the six tensors' sha256 and stop")
    ap.add_argument("--zero-error-layer", action="store_true", help="CONTROL: write the error layer as zeros")
    args = ap.parse_args()
    report = print
    if not args.digest:
        if not args.plain or not args.out:
            abort("--plain and --out are required (or --digest)")
        if os.path.exists(args.out):
            abort(f"{args.out} exists")

    _, top = wire.read_net(args.error_net)
    test_model = wire.onnx_model_fields(top)[1][0][1]
    nodes, inits, _, outputs = wire.parse_graph(test_model)
    report(f"error-head export graph: {len(nodes)} nodes, {len(inits)} initializers, outputs {outputs}")
    trunk_width, sources = extract_head(Graph(nodes, inits), outputs, report)
    extra, carried = encode(sources, report, args.zero_error_layer)
    if args.digest:
        return 0
    del nodes, inits, test_model, top

    report(f"inputs: weights file {sha8_file(args.plain)}  error-head export {sha8_file(args.error_net)}")
    with gzip.open(args.plain, "rb") as handle:
        raw = handle.read()
    om_span = one(raw, _NET_ONNX_MODEL, "source weights file: Net.onnx_model")
    om = raw[om_span[3]:om_span[4]]
    model_span = one(om, _ONNX_MODEL_MODEL, "source weights file: onnx_model.model")
    model = om[model_span[3]:model_span[4]]
    graph_span = one(model, _MODEL_GRAPH, "source weights file: ModelProto.graph")
    graph = model[graph_span[3]:graph_span[4]]
    _, plain_inits, _, _ = wire.parse_graph(memoryview(model), with_nodes=False)
    clash = sorted(n for n in plain_inits if n.startswith(PREFIX + "/"))
    if clash:
        abort(f"the source weights file already has {PREFIX}/ initializers {clash}")
    embedding_w = plain_inits.get("/value/embedding/w")
    if embedding_w is None or list(embedding_w.dims) != [trunk_width, WIDTH]:
        abort(f"the source weights file's /value/embedding/w is {embedding_w.describe() if embedding_w else None}; "
              f"the error head reads width {trunk_width}")
    if embedding_w.dtype != ONNX_FLOAT16:
        abort(f"the source weights file's /value/embedding/w is {embedding_w.describe()}, not FLOAT16")
    if tuple(embedding_w.values()) == tuple(carried[f"{PREFIX}/embedding/w"]):
        abort("the error head's embedding equals the source file's /value/embedding/w: not a head of its own")

    new_graph = graph + b"".join(wire.enc_bytes_field(_GRAPH_INITIALIZER, tensor) for _, tensor in extra)
    new_model = replace(model, graph_span, new_graph)
    new_om = replace(om, model_span, new_model)
    out = replace(raw, om_span, new_om)
    with gzip.open(args.out, "wb", compresslevel=1) as handle:
        handle.write(out)
    report(f"wrote {args.out}: {os.path.getsize(args.out):,} bytes, sha {sha8_file(args.out)}")

    # ---- verify what was written, from the file --------------------------------------------------------------------
    _, written_top = wire.read_net(args.out)
    _, plain_top = wire.read_net(args.plain)
    if set(written_top) != set(plain_top):
        abort(f"written Net fields {sorted(written_top)}, source {sorted(plain_top)}")
    for fno, entries in plain_top.items():
        if fno != _NET_ONNX_MODEL and [raw_bytes(v) for _, v in written_top[fno]] != [raw_bytes(v) for _, v in entries]:
            abort(f"Net field {fno} differs from the source weights file")
    om_fields = wire.onnx_model_fields(written_top)
    plain_om_fields = wire.onnx_model_fields(plain_top)
    if set(om_fields) != set(plain_om_fields):
        abort(f"written onnx_model fields {sorted(om_fields)}, source {sorted(plain_om_fields)}")
    for fno, entries in plain_om_fields.items():
        if fno != _ONNX_MODEL_MODEL and [raw_bytes(v) for _, v in om_fields[fno]] != [raw_bytes(v) for _, v in entries]:
            abort(f"onnx_model field {fno} differs from the source weights file")
    written_model = bytes(om_fields[1][0][1])
    written_graph_span = one(written_model, _MODEL_GRAPH, "written ModelProto.graph")
    written_graph = written_model[written_graph_span[3]:written_graph_span[4]]
    if (written_model[:written_graph_span[2]] != model[:graph_span[2]]
            or written_model[written_graph_span[4]:] != model[graph_span[4]:]
            or written_graph[:len(graph)] != graph
            or len(written_graph) != len(graph) + sum(len(wire.enc_bytes_field(_GRAPH_INITIALIZER, t)) for _, t in extra)):
        abort("the written model is not the source model with initializers appended to its graph")
    _, written, _, _ = wire.parse_graph(memoryview(written_model), with_nodes=False)
    for name, tensor in plain_inits.items():
        if bytes(written[name].wire) != bytes(tensor.wire):
            abort(f"source initializer {name} changed")
    added = {n: t for n, t in written.items() if n.startswith(PREFIX + "/")}
    if len(written) != len(plain_inits) + len(added) or set(added) != set(sources):
        abort(f"written initializers: {len(written)} = {len(plain_inits)} source + {sorted(added)}")
    for name, source in sources.items():
        t = added[name]
        if t.dtype != ONNX_FLOAT16 or list(t.dims) != list(source.dims) or tuple(t.values()) != tuple(carried[name]):
            abort(f"written {name} differs from what was encoded")
    report(f"VERIFY: every Net field but onnx_model byte-identical to the source file; every onnx_model field but the model "
           f"byte-identical (none added); the model = the source model with tensors appended to its graph, nothing else "
           f"changed; {len(plain_inits)} source initializers byte-identical; {len(added)} {PREFIX}/ initializers as encoded")
    report("CARRIER OK" + (" (CONTROL FILE: zeroed error layer, never serve it)" if args.zero_error_layer else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
