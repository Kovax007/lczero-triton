"""The child-Q distribution head in a net file: what the BT4 builder reads, and what enters the fingerprint.

The head (`childq_dist_head.py` in the lab tree, served by `export512.py --childq-head`) is not an lc0 `Weights` layer,
so a net carries it the way the lab nets carry their weights: as ONNX initializers in `onnx_model`, beside the BT4
`weights` stanza this builder reads. Two OnnxModel fields name its outputs (8 `output_childq_mean` and 9
`output_childq_var`, from the child-Q consumer patch's `net.proto`). This package's `net_pb2` predates them, so they are
read and written as unknown fields.

Three checks keep the head from being dropped or mismatched:
  * the executable's fingerprint takes the two output names, as lc0's `BuildNetworkFingerprint` does, so an executable
    built with the head accepts only a net that declares it, and the reverse;
  * the builder requires exactly the initializers of `ChildQHead.initializers()` under `/childq/`, no more, no fewer;
  * the runtime refuses a `/childq/` initializer that has no buffer, so no head weight is skipped with a warning.
"""

from dataclasses import dataclass

from google.protobuf import unknown_fields
from lc0ex.proto import lc0ex_pb2, net_pb2

from lczero_triton.bt4 import _onnx_wire
from lczero_triton.bt4._format import NetworkFormatError

PREFIX = "/childq/"
OUTPUT_MEAN = "/output/childq_mean"
OUTPUT_VAR = "/output/childq_var"
_OUTPUT_MEAN_FIELD = 8
_OUTPUT_VAR_FIELD = 9
_LENGTH_DELIMITED = 2

# ONNX TensorProto data types, and the lc0ex buffer types that carry them.
ONNX_FLOAT = 1
ONNX_FLOAT16 = 10
_BUFFER_TYPES = {
    ONNX_FLOAT: lc0ex_pb2.Buffer.DATA_TYPE_F32,
    ONNX_FLOAT16: lc0ex_pb2.Buffer.DATA_TYPE_F16,
}

TOKENS_W = PREFIX + "tokens/w"
TOKENS_B = PREFIX + "tokens/b"
QUERY_W = PREFIX + "q/w"
QUERY_B = PREFIX + "q/b"
KEY_W = PREFIX + "k/w"
KEY_B = PREFIX + "k/b"
BINS_W = PREFIX + "bins/w"
PROMOTION_W = PREFIX + "promotion/w"
POSITION_W = PREFIX + "position/w"
POSITION_B = PREFIX + "position/b"
ATOMS = PREFIX + "atoms"


@dataclass(frozen=True, slots=True)
class ChildQHead:
    """The head's widths, read from the initializers a net carries."""

    source_width: int
    embedding_width: int
    model_width: int
    bin_count: int

    def initializers(self) -> dict[str, tuple[int, tuple[int, ...]]]:
        """Every initializer the head needs: name -> (ONNX data type, dims).

        Matrices are in ONNX Gemm layout [input width, output width], as the BT4
        builder's own FP16 matrices. The GEMM weights are FP16 like BT4's; the bin,
        promotion and position weights and the atom grid are FLOAT, read by the head's
        float32 kernels.
        """
        source, embedding = self.source_width, self.embedding_width
        depth, bins = self.model_width, self.bin_count
        return {
            TOKENS_W: (ONNX_FLOAT16, (source, embedding)),
            TOKENS_B: (ONNX_FLOAT16, (embedding,)),
            QUERY_W: (ONNX_FLOAT16, (embedding, depth)),
            QUERY_B: (ONNX_FLOAT16, (depth,)),
            KEY_W: (ONNX_FLOAT16, (embedding, depth)),
            KEY_B: (ONNX_FLOAT16, (depth,)),
            BINS_W: (ONNX_FLOAT, (depth, bins)),
            PROMOTION_W: (ONNX_FLOAT, (depth, 4, bins)),
            POSITION_W: (ONNX_FLOAT, (embedding, bins)),
            POSITION_B: (ONNX_FLOAT, (bins,)),
            ATOMS: (ONNX_FLOAT, (bins,)),
        }

    def buffers(self) -> dict[str, tuple[int, tuple[int, ...]]]:
        """The same initializers with the lc0ex buffer data types."""
        return {
            name: (_BUFFER_TYPES[data_type], dims)
            for name, (data_type, dims) in self.initializers().items()
        }


def declared_outputs(network: net_pb2.Net) -> dict[int, str]:
    """Return the output names in onnx_model fields 8 and 9, by field number."""
    if not network.HasField("onnx_model"):
        return {}
    known = net_pb2.OnnxModel.DESCRIPTOR.fields_by_number
    if _OUTPUT_MEAN_FIELD in known or _OUTPUT_VAR_FIELD in known:
        message = "net_pb2.OnnxModel defines field 8 or 9: read them as known fields"
        raise NetworkFormatError(message)
    names: dict[int, str] = {}
    for field in unknown_fields.UnknownFieldSet(network.onnx_model):
        if field.field_number not in (_OUTPUT_MEAN_FIELD, _OUTPUT_VAR_FIELD):
            continue
        if field.wire_type != _LENGTH_DELIMITED:
            message = f"onnx_model field {field.field_number} is not a string"
            raise NetworkFormatError(message)
        names[field.field_number] = bytes(field.data).decode()
    return names


def declare_outputs(onnx_model: net_pb2.OnnxModel) -> None:
    """Name the head's two outputs in onnx_model fields 8 and 9."""
    onnx_model.SetInParent()
    onnx_model.MergeFromString(
        _onnx_wire.enc_bytes_field(_OUTPUT_MEAN_FIELD, OUTPUT_MEAN.encode())
        + _onnx_wire.enc_bytes_field(_OUTPUT_VAR_FIELD, OUTPUT_VAR.encode())
    )


def add_to_fingerprint(fingerprint: net_pb2.Net) -> None:
    """Put the head's output names in the sparse fingerprint, as lc0 does."""
    declare_outputs(fingerprint.onnx_model)


def read_childq_head(network: net_pb2.Net) -> ChildQHead | None:
    """Return the head a net carries, or None; refuse a partial or inconsistent one."""
    names = declared_outputs(network)
    carried: dict[str, tuple[int, tuple[int, ...]]] = {}
    if network.HasField("onnx_model") and network.onnx_model.model:
        _, initializers, _, _ = _onnx_wire.parse_graph(
            memoryview(network.onnx_model.model), with_nodes=False
        )
        carried = {
            name: (tensor.dtype, tuple(tensor.dims))
            for name, tensor in initializers.items()
            if name.startswith(PREFIX)
        }
    if not names and not carried:
        return None
    expected_names = {_OUTPUT_MEAN_FIELD: OUTPUT_MEAN, _OUTPUT_VAR_FIELD: OUTPUT_VAR}
    if names != expected_names:
        message = (
            f"child-Q head: onnx_model fields 8/9 must name {OUTPUT_MEAN} and "
            f"{OUTPUT_VAR}; got {names} beside {len(carried)} {PREFIX} initializers"
        )
        raise NetworkFormatError(message)
    try:
        _, (source_width, embedding_width) = carried[TOKENS_W]
        _, (_, model_width) = carried[QUERY_W]
        _, (_, bin_count) = carried[BINS_W]
    except (KeyError, ValueError) as error:
        message = f"child-Q head: cannot read its widths ({error!r})"
        raise NetworkFormatError(message) from error
    head = ChildQHead(source_width, embedding_width, model_width, bin_count)
    expected = head.initializers()
    if carried != expected:
        missing = sorted(set(expected) - set(carried))
        extra = sorted(set(carried) - set(expected))
        wrong = [
            (name, carried[name], expected[name])
            for name in sorted(set(expected) & set(carried))
            if carried[name] != expected[name]
        ]
        message = (
            f"child-Q head: missing {missing}, unexpected {extra}, "
            f"wrong type or shape (carried, expected) {wrong}"
        )
        raise NetworkFormatError(message)
    return head
