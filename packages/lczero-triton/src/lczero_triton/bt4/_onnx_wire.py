"""Dependency-free protobuf wire helpers for ONNX models and lc0 Net files.

Neither server15's system python nor its lczero-triton venvs have `onnx`, so the lane's route-A' tools read the wire
format directly (`briefs_2026-09-08/r16_raw/wrap_carrier.py`). This module generalises that reader to nodes and
attributes, and adds the few encoders the child-Q carrier needs.

Field numbers:
  Net: magic 1 (fixed32), format 4, weights 10, onnx_model 11
  OnnxModel: model 1 (the serialized ModelProto); the rest are printed generically
  ModelProto: graph 7
  GraphProto: node 1, initializer 5, input 11, output 12
  NodeProto: input 1, output 2, name 3, op_type 4, attribute 5
  AttributeProto: name 1, f 2, i 3, s 4, t 5, floats 7, ints 8, strings 9
  TensorProto: dims 1, data_type 2, float_data 4, int32_data 5, int64_data 7, name 8, raw_data 9
  ValueInfoProto: name 1
"""

import array
import gzip
import struct

DTYPE_NAMES = {1: "FLOAT", 6: "INT32", 7: "INT64", 9: "BOOL", 10: "FLOAT16", 11: "DOUBLE"}
_ARRAY_CODES = {1: "f", 6: "i", 7: "q", 11: "d"}


def varint(b, i):
    result = shift = 0
    while True:
        c = b[i]
        i += 1
        result |= (c & 0x7F) << shift
        if c < 0x80:
            return result, i
        shift += 7


def signed64(v):
    return v - (1 << 64) if v >= (1 << 63) else v


def fields(b):
    """Yield (field number, wire type, value) over one message; length-delimited values are memoryviews."""
    i, n = 0, len(b)
    while i < n:
        key, i = varint(b, i)
        fno, wt = key >> 3, key & 7
        if wt == 0:
            v, i = varint(b, i)
        elif wt == 1:
            v = b[i:i + 8]
            i += 8
        elif wt == 2:
            ln, i = varint(b, i)
            v = b[i:i + ln]
            i += ln
        elif wt == 5:
            v = b[i:i + 4]
            i += 4
        else:
            raise ValueError(f"unsupported wire type {wt} before byte {i}")
        yield fno, wt, v


def packed_varints(v):
    out, j = [], 0
    while j < len(v):
        x, j = varint(v, j)
        out.append(x)
    return out


class Tensor:
    __slots__ = ("name", "dims", "dtype", "raw", "float_data", "int32_data", "int64_data", "wire")

    def __init__(self):
        self.name, self.dims, self.dtype, self.raw = None, [], None, None
        self.float_data, self.int32_data, self.int64_data = [], [], []
        self.wire = None

    def numel(self):
        n = 1
        for d in self.dims:
            n *= d
        return n

    def values(self):
        """All values as a Python array (FLOAT16 decoded to float), from raw_data or the typed fields."""
        if self.raw is not None and len(self.raw):
            if self.dtype == 10:
                return array.array("f", struct.unpack(f"<{len(self.raw) // 2}e", self.raw))
            code = _ARRAY_CODES.get(self.dtype)
            if code is None:
                raise ValueError(f"tensor {self.name}: no decoder for dtype {self.dtype}")
            a = array.array(code)
            a.frombytes(bytes(self.raw))
            return a
        if self.dtype == 1:
            return array.array("f", self.float_data)
        if self.dtype == 7:
            return array.array("q", self.int64_data)
        if self.dtype in (6, 9, 10):
            return array.array("i", self.int32_data)
        raise ValueError(f"tensor {self.name}: empty or unsupported storage (dtype {self.dtype})")

    def describe(self):
        return f"{DTYPE_NAMES.get(self.dtype, self.dtype)}{list(self.dims)}"


def parse_tensor(b):
    t = Tensor()
    t.wire = b
    for fno, wt, v in fields(b):
        if fno == 1:
            if wt == 0:
                t.dims.append(signed64(v))
            else:
                t.dims.extend(signed64(x) for x in packed_varints(v))
        elif fno == 2:
            t.dtype = v
        elif fno == 4:
            if wt == 5:
                t.float_data.append(struct.unpack("<f", v)[0])
            else:
                t.float_data.extend(struct.unpack(f"<{len(v) // 4}f", v))
        elif fno == 5:
            if wt == 0:
                t.int32_data.append(signed64(v))
            else:
                t.int32_data.extend(signed64(x) for x in packed_varints(v))
        elif fno == 7:
            if wt == 0:
                t.int64_data.append(signed64(v))
            else:
                t.int64_data.extend(signed64(x) for x in packed_varints(v))
        elif fno == 8:
            t.name = bytes(v).decode()
        elif fno == 9:
            t.raw = v
    return t


class Node:
    __slots__ = ("inputs", "outputs", "name", "op", "attrs", "index")

    def __init__(self):
        self.inputs, self.outputs, self.name, self.op, self.attrs, self.index = [], [], "", "", {}, -1


def parse_attribute(b):
    name, value = None, None
    ints, floats, strings = [], [], []
    for fno, wt, v in fields(b):
        if fno == 1:
            name = bytes(v).decode()
        elif fno == 2:
            value = struct.unpack("<f", v)[0]
        elif fno == 3:
            value = signed64(v)
        elif fno == 4:
            value = bytes(v).decode(errors="replace")
        elif fno == 5:
            value = parse_tensor(v)
        elif fno == 7:
            if wt == 5:
                floats.append(struct.unpack("<f", v)[0])
            else:
                floats.extend(struct.unpack(f"<{len(v) // 4}f", v))
        elif fno == 8:
            if wt == 0:
                ints.append(signed64(v))
            else:
                ints.extend(signed64(x) for x in packed_varints(v))
        elif fno == 9:
            strings.append(bytes(v).decode(errors="replace"))
    if value is None:
        value = ints or floats or strings
    return name, value


def parse_node(b):
    n = Node()
    for fno, wt, v in fields(b):
        if fno == 1:
            n.inputs.append(bytes(v).decode())
        elif fno == 2:
            n.outputs.append(bytes(v).decode())
        elif fno == 3:
            n.name = bytes(v).decode()
        elif fno == 4:
            n.op = bytes(v).decode()
        elif fno == 5:
            k, val = parse_attribute(v)
            n.attrs[k] = val
    return n


def _value_info_name(b):
    for fno, wt, v in fields(b):
        if fno == 1:
            return bytes(v).decode()
    return None


def graph_bytes(model):
    graph = None
    for fno, wt, v in fields(model):
        if fno == 7 and wt == 2:
            graph = v
    if graph is None:
        raise ValueError("no graph in the ONNX model")
    return graph


def parse_graph(model, with_nodes=True):
    """-> (nodes, initializers by name, graph input names, graph output names)."""
    nodes, inits, gin, gout = [], {}, [], []
    for fno, wt, v in fields(graph_bytes(model)):
        if fno == 1 and with_nodes:
            node = parse_node(v)
            node.index = len(nodes)
            nodes.append(node)
        elif fno == 5:
            t = parse_tensor(v)
            if t.name in inits:
                raise ValueError(f"duplicate initializer {t.name}")
            inits[t.name] = t
        elif fno == 11:
            gin.append(_value_info_name(v))
        elif fno == 12:
            gout.append(_value_info_name(v))
    return nodes, inits, gin, gout


def read_net(path):
    """-> (the decompressed Net bytes, {field number: [(wire type, value), ...]} of the top-level Net)."""
    with gzip.open(path, "rb") as f:
        raw = memoryview(f.read())
    top = {}
    for fno, wt, v in fields(raw):
        top.setdefault(fno, []).append((wt, v))
    return raw, top


def onnx_model_fields(top):
    if 11 not in top:
        raise ValueError("the net has no onnx_model (field 11)")
    out = {}
    for wt, v in top[11]:
        for fno, wt2, v2 in fields(v):
            out.setdefault(fno, []).append((wt2, v2))
    return out


# ---- encoders ---------------------------------------------------------------------------------------------------------
def enc_varint(x):
    out = bytearray()
    while True:
        b = x & 0x7F
        x >>= 7
        if x:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def enc_key(fno, wt):
    return enc_varint((fno << 3) | wt)


def enc_bytes_field(fno, payload):
    payload = bytes(payload)
    return enc_key(fno, 2) + enc_varint(len(payload)) + payload


def enc_varint_field(fno, value):
    return enc_key(fno, 0) + enc_varint(value & ((1 << 64) - 1))


def enc_tensor(name, dtype, dims, raw):
    """TensorProto with dims (non-packed, as onnx writes them), data_type, name and raw_data."""
    out = b"".join(enc_varint_field(1, d) for d in dims)
    out += enc_varint_field(2, dtype)
    out += enc_bytes_field(8, name.encode())
    out += enc_bytes_field(9, raw)
    return out
