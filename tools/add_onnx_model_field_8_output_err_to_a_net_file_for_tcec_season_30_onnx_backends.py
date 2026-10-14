#!/usr/bin/env python3
"""Copy an lc0 net file (.pb.gz with an ONNX payload) and name its error output in `onnx_model` field 8 (backend 10-13).

    usage: ...py SOURCE.pb.gz OUT.pb.gz [OUTPUT_NAME]            (OUTPUT_NAME default: /output/err)

lc0's TCEC season-30 tree (`Kovax007/lc0 tcec-season-30-swiss-lc0ex`, proto/net.proto) reads the error head's output
of an ONNX net through `OnnxModel.output_err = 8`; its ONNX backends return it as EvalResult::e. The lab's export tools
do not set the field, because in the lab's FLEET-LINE tree the same field number is `output_childq_mean` (and 9 is
`output_childq_var`): a file with field 8 = "/output/err" must NEVER be given to a fleet-line engine's ONNX backends,
and a child-Q export must never be given to the season-30 tree's. This tool writes the season-30 form as a separate
file, used as the ONNX fp32 REFERENCE of the error output in the gates. Every other byte of the net is kept; the
graph must produce OUTPUT_NAME and the source must not carry a field 8 or 9. Pure Python (the tree's `_onnx_wire`).
"""
import gzip
import hashlib
import os
import sys

from lczero_triton.bt4 import _onnx_wire as wire

source, out = sys.argv[1], sys.argv[2]
name = sys.argv[3] if len(sys.argv) > 3 else "/output/err"
if os.path.exists(out):
    raise SystemExit(f"{out} exists")
with gzip.open(source, "rb") as handle:
    raw = handle.read()
top_spans, i = [], 0
while i < len(raw):
    start = i
    key, i = wire.varint(raw, i)
    fno, wt = key >> 3, key & 7
    if wt == 0:
        _, i = wire.varint(raw, i)
        payload = start
    elif wt == 1:
        payload, i = i, i + 8
    elif wt == 5:
        payload, i = i, i + 4
    elif wt == 2:
        length, payload = wire.varint(raw, i)
        i = payload + length
    else:
        raise SystemExit(f"wire type {wt} at byte {start}")
    top_spans.append((fno, wt, start, payload, i))
hits = [s for s in top_spans if s[0] == 11 and s[1] == 2]
if len(hits) != 1:
    raise SystemExit(f"{len(hits)} onnx_model fields in the net")
_, _, start, payload, end = hits[0]
om = raw[payload:end]
fields = {}
for fno, wt, value in wire.fields(memoryview(om)):
    fields.setdefault(fno, []).append(value)
if 8 in fields or 9 in fields:
    raise SystemExit("the source already carries onnx_model field 8 or 9")
_, _, _, outputs = wire.parse_graph(fields[1][0], with_nodes=False)
if name not in outputs:
    raise SystemExit(f"the graph's outputs are {outputs}: no {name}")
new_om = bytes(om) + wire.enc_bytes_field(8, name.encode())
written = bytes(raw[:start]) + wire.enc_bytes_field(11, new_om) + bytes(raw[end:])
with gzip.open(out, "wb", compresslevel=1) as handle:
    handle.write(written)
# verify from the file
with gzip.open(out, "rb") as handle:
    back = handle.read()
check = {}
for fno, wt, value in wire.fields(memoryview(back)):
    check.setdefault(fno, []).append(bytes(value) if hasattr(value, "tobytes") else value)
om_back = {}
for fno, wt, value in wire.fields(memoryview(check[11][0])):
    om_back.setdefault(fno, []).append(bytes(value) if hasattr(value, "tobytes") else value)
same = all(om_back[f] == [bytes(v) if hasattr(v, "tobytes") else v for v in fields[f]] for f in fields)
if not same or set(om_back) != set(fields) | {8} or om_back[8] != [name.encode()]:
    raise SystemExit("verification failed: the written onnx_model is not the source's plus field 8")
digest = hashlib.sha256(open(out, "rb").read()).hexdigest()[:8]
print(f"wrote {out}: onnx_model fields {sorted(om_back)} (8 = {name!r}), graph outputs {outputs}, every other field byte-identical; sha256 {digest}")
