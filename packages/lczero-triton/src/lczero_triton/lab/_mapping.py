"""Recover the lab net's architecture and weight names from its own graph.

The lab's exporter names every initializer `const_<n>` in first-use order, so
the names carry no meaning and cannot be mapped by a table -- a table would also
silently rot the first time an arm reorders an operation. What the graph does
carry is structure, and the structure is rigid: fifteen identical encoder blocks
around fifteen `Softmax` nodes, each with the same operator sequence.

So this module matches on the structure and reads the names off the matched
nodes. Every block must produce the same shapes as every other block, and that
agreement is asserted rather than assumed: a mapping that is wrong for one block
is wrong for all fifteen and will not survive the check.

⚠ The FFN gate is the reason the checking matters (round 17, AC-2). The lab's
`ACTIVATION_SWIGLU` enum selects Dauphin's **sigmoid** GLU, not SiLU, and the
only place that is unambiguous is the artifact: this recovery reads the gate
from the `Sigmoid` node it actually finds, and refuses a graph whose gate is
some other activation rather than guessing.
"""

import math
import struct
from dataclasses import dataclass
from typing import Literal

from lczero_triton.lab._onnx import FLOAT32, Graph, INT32, INT64, Node, Tensor

_SQUARE_COUNT = 64
_INPUT_CHANNELS = 112
_POSITION_CHANNELS = 12
_EXPECTED_LAYER_NORMS_PER_BLOCK = 2


class ArchitectureError(ValueError):
    """The graph does not match the encoder structure this package builds."""


@dataclass(frozen=True, slots=True)
class Architecture:
    """The dimensions the builder needs, all recovered from the graph."""

    blocks: int
    d_model: int
    heads: int
    head_dim: int
    ffn_hidden: int
    embedding_ffn_hidden: int
    edge_channels: int
    mix_channels: int
    tokens: int = _SQUARE_COUNT
    # Zero on a net without smolgen; mix_channels is zero without the pair stream.
    smolgen_channels: int = 0
    smolgen_hidden: int = 0
    smolgen_gen: int = 0
    # O (round 22): read off block 0's structure; every block must agree (`_assert_blocks_agree`).
    # "postnorm": x = ln1(x + a attn(x)); x = ln2(x + a ffn(x)).  "prenorm": x += a attn(ln1(x)); x += a ffn(ln2(x)),
    # then one final norm before the heads (`LabNetwork.final_norm`).
    block_style: Literal["postnorm", "prenorm"] = "postnorm"
    # `gate_scale * sigmoid(x W_g + b_g)` on the merged attention output before the out-projection, read from the
    # same tensor as Q, K and V (`EncoderBlock.gate_weight`).
    output_gate: bool = False
    # BT6-test (09-21). `ffn_softcap`: both GLU branches pass through c*tanh(./c) before their product, in every
    # encoder block AND in the embedding FFN (one `DefaultsConfig` value); 0 = the uncapped sigmoid GLU.
    ffn_softcap: float = 0.0
    # Per-square width of the preprocess dense (`embedding.dense_size`) and width of the policy head
    # (`shared_policy_embedding_size` / the served head's `d_model`). Both equal d_model on every 512 lab net, which
    # is why they were once assumed; the sponsor net has 512 and 512 on a 1024 trunk. 0 = "d_model" (old pickles).
    embedding_dense: int = 0
    policy_width: int = 0


@dataclass(frozen=True, slots=True)
class EncoderBlock:
    """Initializer names for one encoder block, in the order they are used."""

    query_weight: str
    query_bias: str
    key_weight: str
    key_bias: str
    value_weight: str
    value_bias: str
    output_weight: str
    output_bias: str
    attention_alpha: str
    ln1_scale: str
    ln1_bias: str
    ffn_gate_weight: str
    ffn_up_weight: str
    ffn_up_bias: str
    ffn_down_weight: str
    ffn_down_bias: str
    ffn_alpha: str
    ln2_scale: str
    ln2_bias: str
    # The five pre-softmax logit terms beyond QK^T, in the order the graph adds
    # them: C1 [1,H,E], Qk [1,H,E,D] against the keys, Qq [1,H,E,D] against the
    # queries, C2 [1,H,D] against the mix stream.
    edge_coefficients: str
    pair_key: str
    pair_query: str
    # None on a net without the pair stream (the smolgen twin).
    mix_coefficients: str | None
    attention_scale: float
    smolgen: "SmolgenBlock | None" = None
    # O: the attention output gate, `gate_scale * sigmoid(x W_g + b_g)` per (square, channel) on the merged heads
    # before the out-projection; None without `mha_output_gate`. `gate_node` is the product with the attention.
    gate_weight: str | None = None
    gate_bias: str | None = None
    gate_scale: float = 0.0
    gate_node: int = -1
    # BT6-test: c of the capped sigmoid GLU (`Architecture.ffn_softcap`); 0 = uncapped.
    ffn_softcap: float = 0.0


@dataclass(frozen=True, slots=True)
class SmolgenBlock:
    """One block's smolgen chain, BT4's family: compress, then two dense + swish + norm.

    Every block names the same `weight_gen` initializer; `_assert_blocks_agree`
    holds the export to that.
    """

    compress_weight: str
    dense1_weight: str
    dense1_bias: str
    ln1_scale: str
    ln1_bias: str
    dense2_weight: str
    dense2_bias: str
    ln2_scale: str
    ln2_bias: str
    weight_gen: str


@dataclass(frozen=True, slots=True)
class Embedding:
    """Initializer names for the input embedding and its gated FFN."""

    preprocess_weight: str
    preprocess_bias: str
    embedding_weight: str
    embedding_bias: str
    ln0_scale: str
    ln0_bias: str
    input_gate: str
    positional: str
    ffn_gate_weight: str
    ffn_up_weight: str
    ffn_up_bias: str
    ffn_down_weight: str
    ffn_down_bias: str
    ffn_alpha: str
    ln1_scale: str
    ln1_bias: str
    activation: Literal["mish"]
    # BT6-test: c of the capped sigmoid GLU in the embedding FFN; 0 = uncapped. Must equal the blocks'.
    ffn_softcap: float = 0.0


@dataclass(frozen=True, slots=True)
class Heads:
    """Initializer names for the three output heads the graph actually wires.

    The export carries tensors for policy and value heads the selected graph
    never reaches -- the lab trains several and exports the pair a run selected.
    Recovering from the nodes rather than from the initializer list is what
    keeps those unreachable weights out of the artifact instead of quietly
    sizing buffers for them.
    """

    value_square_weight: str
    value_square_bias: str
    value_hidden_weight: str
    value_hidden_bias: str
    value_output_weight: str
    value_output_bias: str
    policy_embedding_weight: str
    policy_embedding_bias: str
    policy_query_weight: str
    policy_query_bias: str
    policy_key_weight: str
    policy_key_bias: str
    policy_promotion_weight: str
    policy_mix_coefficients: str | None
    policy_map: str
    moves_square_weight: str
    moves_square_bias: str
    moves_hidden_weight: str
    moves_hidden_bias: str
    moves_output_weight: str
    moves_output_bias: str
    activation: Literal["mish"]


@dataclass(frozen=True, slots=True)
class PairTables:
    """The pair stream's shared constants, read once for the whole network.

    `Z = einsum("cij,cd->dij", E, channel_mix) + T` with
    `T = offset_table.T[:, relative_index]`, and the stream is RMS-normed with
    `epsilon`. Every block's mix term and the policy head's contract against the
    same `Z`, so these feed the build-time folds of all sixteen consumers.
    """

    channel_mix: str
    offset_table: str
    relative_index: str
    epsilon: float


@dataclass(frozen=True, slots=True)
class EgtBlock:
    """One EGT2 block's edge-stream terms, read around its attention Softmax (item E, R0).

    `H = (t_node F + t_edge sum_c edge_e[h,c] e[c]) (1 + sum_c edge_m[h,c] e[c])` with `F` the static logits
    (`EncoderBlock`'s tables), then `A = softmax(H) g`, `g = 2 sigmoid(sum_c edge_g[h,c] e[c] + g_b)`, clipped
    at 1 when `cap`. The names are the export's constants: temperatures and `g_b` are `[1, heads, 1, 1]`, the
    three channel maps `[1, heads, state_channels]`, applied as `[heads, state_channels]` (out, in).
    """

    node_temperature: str
    edge_temperature: str
    edge_read: str
    door: str
    gate_weight: str
    gate_bias: str
    cap: bool
    state: str  # the edge state this block reads: e_0, or the last update site's output
    edges: str  # E, the 34-channel attack graph
    pair_stream: str  # Z0
    softmax_node: int
    logits_node: int  # H, the readback source
    weights_node: int  # A, the gated attention weights


# The triplet's two contraction orders (item E map section 4), keyed by the equations of its (in, out) Einsums.
TRIPLET_CONTRACTIONS = {
    ("zyabc,zydb->zyacd", "zyabc,zycd->zyabd"): "path",
    ("zyabc,zydc->zyabd", "zyabc,zybd->zyacd"): "ag",
}


@dataclass(frozen=True, slots=True)
class EgtTriplet:
    """The optional triplet branch of one update site, between the readback and the edge FFN.

    Constants as exported: `value_weight` `[1, S, 2S]`, `gate_weight` `[1, S, S]`, `gate_bias` `[1, S, 1, 1]`,
    `output_weight` `[1, 2S, S]`; all are applied transposed, so they serve as `[2S, S]`, `[S, S]`, `[S]` and
    `[S, 2S]` (out, in).
    """

    value_weight: str
    gate_weight: str
    gate_bias: str
    output_weight: str
    contraction: Literal["path", "ag"]
    split_node: int
    add_node: int  # e_hat + triplet


@dataclass(frozen=True, slots=True)
class EgtSite:
    """One edge update site: `e_hat = e + O_e H`, the optional triplet, then `e' = rms(e_hat + W_out relu(W_in rms(e_hat) + b))`.

    Its nodes run from `first_node` (right after the block's second layer norm) to `last_node` (the closing rms),
    so they belong to neither neighbouring block. Constants as exported: `readback_weight` `[1, heads, S]`,
    `ffn_in_weight` `[1, S, hidden]`, `ffn_in_bias` `[1, hidden, 1, 1]`, `ffn_out_weight` `[1, hidden, S]`, all
    applied transposed (served as `[S, heads]`, `[hidden, S]`, `[hidden]`, `[S, hidden]`).
    """

    after_block: int
    readback_weight: str
    ffn_in_weight: str
    ffn_in_bias: str
    ffn_out_weight: str
    epsilon: float
    triplet: EgtTriplet | None
    first_node: int
    last_node: int  # e'
    readback_node: int
    state_node: int  # e_hat
    ffn_add_node: int
    state: str  # the tensor e' names, read by the following blocks
    # The lab's round-32 `rev_edge`: a second in-projection `[1, S, hidden]` reading rms(e_hat) with the two board
    # axes swapped, added to the hidden layer before the Relu. None on a net without it.
    ffn_rev_weight: str | None = None


@dataclass(frozen=True, slots=True)
class EgtPrologue:
    """The edge-stream seed: `e_0 = rms(sum_c p_in[c,d] E[c] + t_off[relidx])` (K1's `/prologue/edge_*`)."""

    edge_mix: str  # p_in, [1, edge_channels, S]
    edge_offset_table: str  # t_off, [bins, S]
    relative_index: str  # shared with the pair stream
    epsilon: float
    edges_node: int  # E
    pair_node: int  # Z0's rms
    state_node: int  # e_0's rms


@dataclass(frozen=True, slots=True)
class EgtNetwork:
    """The EGT2 edge stream of one export, absent (None) on the static family."""

    blocks: tuple[EgtBlock, ...]
    sites: tuple[EgtSite, ...]
    prologue: EgtPrologue
    cap: bool
    state_channels: int
    site_hidden: int


@dataclass(frozen=True, slots=True)
class FinalNorm:
    """The pre-norm tower's one closing norm, between the last residual add and every head (O)."""

    scale: str
    bias: str
    node: int


@dataclass(frozen=True, slots=True)
class LabNetwork:
    """Everything recovered from one lab carrier graph."""

    architecture: Architecture
    embedding: Embedding
    blocks: tuple[EncoderBlock, ...]
    heads: Heads
    pair: PairTables | None
    policy_divisor: float
    egt: EgtNetwork | None = None  # R0: set on an EGT2 export
    final_norm: FinalNorm | None = None  # O: set on a pre-norm export


def _initializer(graph: Graph, name: str) -> Tensor | None:
    """Return the initializer called `name`, or None if it is an activation."""
    return graph.initializers.get(name)


def _weight_of(graph: Graph, node: Node) -> Tensor:
    """Return the initializer a Gemm multiplies by, or fail loudly."""
    tensor = _initializer(graph, node.inputs[1])
    if tensor is None:
        message = f"node {node.index} ({node.op_type}) has no constant weight"
        raise ArchitectureError(message)
    return tensor


def _scalar(graph: Graph, name: str) -> float | None:
    """Return a float32 scalar initializer's value, or None if it is not one."""
    tensor = _initializer(graph, name)
    if tensor is None or tensor.dims != () or tensor.data_type != FLOAT32:
        return None
    return struct.unpack("<f", tensor.raw_data[:4])[0]


def _trace_to_gemm(graph: Graph, tensor: str, *, depth: int = 6) -> Node | None:
    """Walk back through shape-only nodes to the Gemm that produced a value.

    The exporter interleaves `Reshape`/`Transpose` with every projection, so a
    consumer never names the Gemm it depends on directly. Only nodes that move
    data without changing it are stepped through, which keeps the walk from
    wandering into a different branch.
    """
    passthrough = {"Reshape", "Transpose", "Squeeze", "Unsqueeze", "Identity"}
    current = tensor
    for _ in range(depth):
        node = graph.produced_by(current)
        if node is None:
            return None
        if node.op_type == "Gemm":
            return node
        if node.op_type not in passthrough:
            return None
        current = node.inputs[0]
    return None


def _is_swish_norm(graph: Graph, node: Node) -> bool:
    """Tell a smolgen norm apart: its input is `x * sigmoid(x)`, which no body norm has.

    The embedding norm follows Mish (a Tanh product), the block norms follow a
    residual Add, and the GLU gate multiplies a sigmoid by a *different* tensor.
    """
    if node.op_type != "LayerNormalization":
        return False
    product = graph.produced_by(node.inputs[0])
    if product is None or product.op_type != "Mul" or len(product.inputs) != 2:  # noqa: PLR2004
        return False
    for index, name in enumerate(product.inputs):
        source = graph.produced_by(name)
        if source is not None and source.op_type == "Sigmoid" and source.inputs[0] == product.inputs[1 - index]:
            return True
    return False


def _reaches(graph: Graph, tensor: str, target: Node, *, depth: int = 6) -> bool:
    """Report whether `tensor` is `target`'s output behind shape-only nodes."""
    passthrough = {"Reshape", "Transpose", "Squeeze", "Unsqueeze", "Identity"}
    current = tensor
    for _ in range(depth):
        node = graph.produced_by(current)
        if node is None:
            return False
        if node.index == target.index:
            return True
        if node.op_type not in passthrough:
            return False
        current = node.inputs[0]
    return False


def _read_smolgen(graph: Graph, nodes: list[Node], *, start: int, end: int) -> tuple[SmolgenBlock | None, set[int]]:
    """Recover one block's smolgen chain and the node indices it claims, or (None, {}).

    Anchored on the two swish norms. dense1/dense2 are the Gemms under their swishes,
    compress the Gemm behind dense1's input, weight_gen the Gemm reading ln2's output.
    """
    norms = [node for node in nodes if _is_swish_norm(graph, node)]
    if not norms:
        return None, set()
    if len(norms) != 2:  # noqa: PLR2004
        message = f"block at nodes {start}-{end}: expected two smolgen norms, found {len(norms)}"
        raise ArchitectureError(message)
    claimed: set[int] = set()
    denses: list[Node] = []
    for norm in norms:
        product = graph.produced_by(norm.inputs[0])
        sigmoid = next(
            graph.produced_by(name) for name in product.inputs
            if graph.produced_by(name) is not None and graph.produced_by(name).op_type == "Sigmoid"
        )
        dense = graph.produced_by(sigmoid.inputs[0])
        if dense is None or dense.op_type != "Gemm" or len(dense.inputs) < 3 or not dense.inputs[2]:  # noqa: PLR2004
            message = f"block at nodes {start}-{end}: a smolgen swish is not fed by a biased Gemm"
            raise ArchitectureError(message)
        denses.append(dense)
        claimed |= {norm.index, sigmoid.index}
    ln1, ln2 = norms
    dense1, dense2 = denses
    compress = _trace_to_gemm(graph, dense1.inputs[0])
    generators = [
        node for node in nodes if node.op_type == "Gemm" and _reaches(graph, node.inputs[0], ln2)
    ]
    if compress is None or len(generators) != 1:
        message = f"block at nodes {start}-{end}: smolgen compress or weight_gen projection not found"
        raise ArchitectureError(message)
    weight_gen = generators[0]
    for node in (compress, weight_gen):
        if len(node.inputs) > 2 and node.inputs[2]:  # noqa: PLR2004
            message = f"block at nodes {start}-{end}: smolgen {node.name} carries a bias; BT4's does not"
            raise ArchitectureError(message)
    claimed |= {compress.index, dense1.index, dense2.index, weight_gen.index}
    return SmolgenBlock(
        compress_weight=compress.inputs[1],
        dense1_weight=dense1.inputs[1],
        dense1_bias=dense1.inputs[2],
        ln1_scale=ln1.inputs[1],
        ln1_bias=ln1.inputs[2],
        dense2_weight=dense2.inputs[1],
        dense2_bias=dense2.inputs[2],
        ln2_scale=ln2.inputs[1],
        ln2_bias=ln2.inputs[2],
        weight_gen=weight_gen.inputs[1],
    ), claimed


_BLOCK_PASSTHROUGH = {"Reshape", "Transpose", "Squeeze", "Unsqueeze", "Identity"}
BlockStyle = Literal["postnorm", "prenorm"]


@dataclass(frozen=True, slots=True)
class _BlockWiring:
    """One block's residual wiring, read forward from its attention Softmax (O).

    Post-norm: `attention_add -> ln1 -> ffn -> ffn_add -> ln2`, the block ends at ln2. Pre-norm: `ln1 -> q/k/v ...
    attention_add -> ln2 -> ffn -> ffn_add`, the block ends at ffn_add and the stream is never normed inside it.
    """

    style: BlockStyle
    output_gemm: int
    attention_add: int
    ffn_add: int
    skip: str  # the block's input: what the attention residual adds to
    ln1: int
    ln2: int
    end: int


def _source_behind(graph: Graph, tensor: str, *, depth: int = 6) -> str:
    """Return the tensor behind `tensor`'s chain of shape-only nodes: the value a projection really reads."""
    current = tensor
    for _ in range(depth):
        node = graph.produced_by(current)
        if node is None or node.op_type not in _BLOCK_PASSTHROUGH:
            return current
        current = node.inputs[0]
    return current


def _residual_add(graph: Graph, consumers: dict[str, list[Node]], gemm: Node) -> tuple[Node, str] | None:
    """Return (the two-input Add, the branch tensor) that `gemm`'s output reaches through reshapes and a scalar Mul."""
    current = gemm.outputs[0]
    for _ in range(6):
        nodes = consumers.get(current, [])
        if len(nodes) != 1:
            return None
        node = nodes[0]
        if node.op_type == "Add" and len(node.inputs) == 2:  # noqa: PLR2004
            return node, current
        scaled = node.op_type == "Mul" and any(_scalar(graph, name) is not None for name in node.inputs)
        if node.op_type not in _BLOCK_PASSTHROUGH and not scaled:
            return None
        current = node.outputs[0]
    return None


def _block_wiring(graph: Graph, consumers: dict[str, list[Node]], anchor: int) -> _BlockWiring:  # noqa: C901
    """Recover one block's residual wiring from its Softmax, or say what was seen instead.

    The out-projection is the first square Gemm after the Softmax whose output lands in a residual Add (a gate
    projection ends in a Sigmoid, the FFN's projections are not square, the next block's Q/K/V end in reshapes).
    What consumes that Add, together with whether the last norm ahead of the Softmax produces or reads the Add's
    skip, tells the two block styles apart.
    """
    where = f"block anchored at Softmax {anchor}"
    found = None
    for node in graph.nodes[anchor + 1:]:
        if node.op_type != "Gemm":
            continue
        rows, columns = _weight_of(graph, node).dims
        if rows != columns:
            continue
        landed = _residual_add(graph, consumers, node)
        if landed is not None:
            found = (node, *landed)
            break
    _require(found is not None, f"{where}: no square projection after the Softmax lands in a residual Add")
    output_gemm, attention_add, branch = found
    skip = next(name for name in attention_add.inputs if name != branch)
    after = consumers.get(attention_add.outputs[0], [])
    norms = [node for node in after if node.op_type == "LayerNormalization"]
    adds = [node for node in after if node.op_type == "Add" and len(node.inputs) == 2]  # noqa: PLR2004
    before = [node for node in graph.nodes[:anchor]
              if node.op_type == "LayerNormalization" and not _is_swish_norm(graph, node)]
    _require(bool(before), f"{where}: no layer norm ahead of the Softmax")
    last = before[-1]
    if len(after) == 1 and len(norms) == 1 and last.outputs[0] == skip:
        # Post-norm: the residual is normed at once (LN1), LN1's output is the FFN's input and skip, LN2 closes.
        ln1 = norms[0]
        ffn_adds = [node for node in consumers.get(ln1.outputs[0], [])
                    if node.op_type == "Add" and len(node.inputs) == 2]  # noqa: PLR2004
        _require(len(ffn_adds) == 1, f"{where}: post-norm block, but LN1 {ln1.index} feeds {len(ffn_adds)} residual Adds")
        ffn_add = ffn_adds[0]
        closing = consumers.get(ffn_add.outputs[0], [])
        _require(len(closing) == 1 and closing[0].op_type == "LayerNormalization",
                 f"{where}: post-norm block, but the FFN residual Add {ffn_add.index} is consumed by "
                 f"{[node.op_type for node in closing]}, expected one LayerNormalization")
        return _BlockWiring("postnorm", output_gemm.index, attention_add.index, ffn_add.index, skip, ln1.index,
                            closing[0].index, closing[0].index)
    if len(after) == 2 and len(norms) == 1 and len(adds) == 1 and last.inputs[0] == skip:  # noqa: PLR2004
        # Pre-norm: LN1 read the skip; the residual feeds LN2 (the FFN's input) and the FFN residual Add directly.
        ln2, ffn_add = norms[0], adds[0]
        fed = [node for node in graph.nodes[attention_add.index:ffn_add.index] if node.op_type == "Gemm"
               and (landed := _residual_add(graph, consumers, node)) is not None and landed[0] is ffn_add]
        _require(len(fed) == 1, f"{where}: pre-norm block, but {len(fed)} projections between the two residual Adds "
                 f"feed Add {ffn_add.index}, expected the FFN's down projection alone")
        return _BlockWiring("prenorm", output_gemm.index, attention_add.index, ffn_add.index, skip, last.index,
                            ln2.index, ffn_add.index)
    relation = ("reads" if last.inputs[0] == skip else "produces" if last.outputs[0] == skip
                else "neither reads nor produces")
    message = (
        f"{where}: the attention residual Add {attention_add.index} is consumed by "
        f"{sorted(node.op_type for node in after)} and the last norm ahead of the Softmax (node {last.index}) "
        f"{relation} the residual's skip; expected post-norm (Add -> LayerNormalization, that norm producing the "
        "skip) or pre-norm (Add -> LayerNormalization + Add, that norm reading the skip)"
    )
    raise ArchitectureError(message)


def _block_ranges(graph: Graph) -> tuple[int, list[tuple[int, int]], BlockStyle, FinalNorm | None]:
    """Return the prologue's last node, each block's [start, end] range, the block style and the final norm.

    A block is anchored on its `Softmax`. Post-norm (the static family): it ends at the second
    `LayerNormalization` after the Softmax -- the norm that closes the FFN residual -- and every layer norm ahead
    of the first Softmax belongs to the embedding, the last of them being the boundary. Pre-norm (O): the norms
    sit inside the branches, so a block ends at its FFN residual Add, the last norm ahead of the first Softmax is
    block 0's own LN1 (the prologue ends just before it), and one final norm after the last block closes the
    tower. `_block_wiring` reads the style off every block and refuses a graph that is neither; the blocks must
    all agree. Either way the ~2,300 nodes of board-geometry construction sit before the boundary in the export,
    out of block 0, even though nothing consumes them until the first attention.

    R0: on an EGT2 export only the attention Softmaxes anchor (`_attention_softmaxes`), and `read_network` then
    moves each block's start past the update site that precedes it.
    """
    softmaxes = _attention_softmaxes(graph)
    if not softmaxes:
        message = "the graph has no Softmax; it is not an attention network"
        raise ArchitectureError(message)
    consumers = _consumers(graph)
    wirings = [_block_wiring(graph, consumers, anchor) for anchor in softmaxes]
    styles = sorted({wiring.style for wiring in wirings})
    if len(styles) != 1:
        message = "the blocks mix residual styles: " + ", ".join(
            f"Softmax {anchor} {wiring.style}" for anchor, wiring in zip(softmaxes, wirings, strict=True)
        )
        raise ArchitectureError(message)
    style = styles[0]
    if style == "postnorm":
        prologue_norms = [
            node.index
            for node in graph.nodes[: softmaxes[0]]
            if node.op_type == "LayerNormalization" and not _is_swish_norm(graph, node)
        ]
        if not prologue_norms:
            message = "the embedding has no layer norm to close it"
            raise ArchitectureError(message)
        prologue_end = prologue_norms[-1]
    else:
        prologue_end = wirings[0].ln1 - 1
    ranges: list[tuple[int, int]] = []
    start = prologue_end + 1
    for anchor, wiring in zip(softmaxes, wirings, strict=True):
        if style == "postnorm":
            # The rule the static family was read with: the second norm after the Softmax closes the block.
            seen = 0
            end = None
            for node in graph.nodes[anchor:]:
                if node.op_type == "LayerNormalization" and not _is_swish_norm(graph, node):
                    seen += 1
                    if seen == _EXPECTED_LAYER_NORMS_PER_BLOCK:
                        end = node.index
                        break
            if end is None:
                message = f"block anchored at node {anchor} has no closing norm"
                raise ArchitectureError(message)
            if end != wiring.end:
                message = f"block anchored at node {anchor}: closes at norm {end} by count, at {wiring.end} by wiring"
                raise ArchitectureError(message)
        ranges.append((start, wiring.end))
        start = wiring.end + 1
    final = None
    if style == "prenorm":
        closing = consumers.get(graph.nodes[wirings[-1].ffn_add].outputs[0], [])
        if len(closing) != 1 or closing[0].op_type != "LayerNormalization":
            message = (
                f"pre-norm tower: the last residual Add {wirings[-1].ffn_add} is consumed by "
                f"{[node.op_type for node in closing]}, expected one final LayerNormalization"
            )
            raise ArchitectureError(message)
        final = FinalNorm(scale=closing[0].inputs[1], bias=closing[0].inputs[2], node=closing[0].index)
    return prologue_end, ranges, style, final


@dataclass(frozen=True, slots=True)
class _OutputGate:
    """The attention output gate as exported: `Gemm -> Sigmoid -> Mul(scale) -> Mul(attention output)` (O)."""

    gemm: Node
    sigmoid: int
    doubling: int
    product: int
    scale: float


def _read_output_gate(
    graph: Graph, nodes: list[Node], sigmoids: list[Node], square_gemms: list[Node], *, start: int, end: int
) -> tuple[_OutputGate | None, list[Node]]:
    """Split a block's Sigmoids into the output gate (behind a square Gemm) and the rest (the GLU's)."""
    where = f"block at nodes {start}-{end}"
    gates: list[tuple[Node, Node]] = []
    others: list[Node] = []
    for sigmoid in sigmoids:
        gemm = _trace_to_gemm(graph, sigmoid.inputs[0])
        if gemm is not None and gemm in square_gemms:
            gates.append((sigmoid, gemm))
        else:
            others.append(sigmoid)
    if not gates:
        return None, others
    _require(len(gates) == 1, f"{where}: {len(gates)} Sigmoids read square projections; at most one output gate expected")
    sigmoid, gemm = gates[0]
    _require(len(gemm.inputs) == 3 and bool(gemm.inputs[2]), f"{where}: the output gate projection carries no bias")  # noqa: PLR2004
    scaling = [node for node in nodes if sigmoid.outputs[0] in node.inputs]
    _require(len(scaling) == 1 and scaling[0].op_type == "Mul" and len(scaling[0].inputs) == 2,  # noqa: PLR2004
             f"{where}: the output gate's Sigmoid is consumed by {[node.op_type for node in scaling]}, "
             "expected one scaling Mul")
    scales = [_scalar(graph, name) for name in scaling[0].inputs if _scalar(graph, name) is not None]
    _require(len(scales) == 1, f"{where}: the output gate's scaling Mul carries {len(scales)} scalar factors, expected one")
    products = [node for node in nodes if scaling[0].outputs[0] in node.inputs]
    _require(len(products) == 1 and products[0].op_type == "Mul" and len(products[0].inputs) == 2,  # noqa: PLR2004
             f"{where}: the scaled output gate is consumed by {[node.op_type for node in products]}, "
             "expected one product with the attention output")
    return _OutputGate(gemm, sigmoid.index, scaling[0].index, products[0].index, scales[0]), others


def _check_output_gate(graph: Graph, gate: _OutputGate, query: Node, output: Node, *, start: int, end: int) -> None:
    """Hold the gate to the form the builder serves: it reads what Q reads and gates what the out-projection reads."""
    where = f"block at nodes {start}-{end}"
    gate_source, query_source = _source_behind(graph, gate.gemm.inputs[0]), _source_behind(graph, query.inputs[0])
    _require(gate_source == query_source, f"{where}: the output gate reads {gate_source} but Q reads {query_source}; "
             "the packed QKVG projection needs one input")
    product = graph.nodes[gate.product]
    read = _source_behind(graph, output.inputs[0])
    _require(read == product.outputs[0], f"{where}: the out-projection reads {read}, not the gated attention "
             f"{product.outputs[0]}")
    doubled = graph.nodes[gate.doubling].outputs[0]
    attention = _source_behind(graph, next(name for name in product.inputs if name != doubled))
    producer = graph.produced_by(attention)
    weights = graph.produced_by(producer.inputs[0]) if producer is not None and producer.op_type == "MatMul" else None
    if weights is not None and weights.op_type == "Mul":
        # Round 26b: an EGT2 block's weights are its Softmax times the edge door's gate, `Mul(Softmax, 2 sigmoid(.))`.
        softmaxes = [node for name in weights.inputs
                     if (node := graph.produced_by(name)) is not None and node.op_type == "Softmax"]
        weights = softmaxes[0] if len(softmaxes) == 1 else weights
    _require(weights is not None and weights.op_type == "Softmax",
             f"{where}: the gate multiplies {attention}, which is not the attention weights' product with V")


def _assert_gate_agrees(graph: Graph, index: int, block: EncoderBlock, first: EncoderBlock, architecture: Architecture) -> None:
    """Hold every block's output gate to block 0's presence, scale and shapes."""
    if (block.gate_weight is None) != (first.gate_weight is None):
        message = f"block {index}: the attention output gate is present in some blocks only"
        raise ArchitectureError(message)
    if block.gate_weight is None:
        return
    if block.gate_scale != first.gate_scale:
        message = f"block {index}: output gate scale {block.gate_scale} differs from block 0's {first.gate_scale}"
        raise ArchitectureError(message)
    width = architecture.d_model
    for attribute, shape in (("gate_weight", (width, width)), ("gate_bias", (width,))):
        name = getattr(block, attribute)
        actual = tuple(graph.initializers[name].dims)
        if actual != shape:
            message = f"block {index}: {attribute} ({name}) has shape {actual}, expected {shape}"
            raise ArchitectureError(message)


def _source_tensor(graph: Graph, tensor: str, *, depth: int = 6) -> str:
    """r25: the tensor a value was reshaped from -- walk back through data-moving nodes only."""
    current = tensor
    for _ in range(depth):
        node = graph.produced_by(current)
        if node is None or node.op_type not in {"Reshape", "Transpose", "Squeeze", "Unsqueeze", "Identity"}:
            return current
        current = node.inputs[0]
    return current


def _softcap_behind(graph: Graph, tensor: str) -> tuple[str, float, int] | None:
    """Return (x, c, Mul index) if `tensor` is `c * tanh(x / c)` as exported: `Mul(c, Tanh(Div(x, c)))`."""
    product = graph.produced_by(tensor)
    if product is None or product.op_type != "Mul" or len(product.inputs) != 2:  # noqa: PLR2004
        return None
    for index, name in enumerate(product.inputs):
        cap = _scalar(graph, name)
        tangent = graph.produced_by(product.inputs[1 - index])
        if cap is None or tangent is None or tangent.op_type != "Tanh":
            continue
        quotient = graph.produced_by(tangent.inputs[0])
        if quotient is None or quotient.op_type != "Div" or len(quotient.inputs) != 2:  # noqa: PLR2004
            continue
        if _scalar(graph, quotient.inputs[1]) == cap and cap > 0.0:
            return quotient.inputs[0], cap, product.index
    return None


def _softcap_products(graph: Graph, nodes: list[Node]) -> set[int]:
    """The `Mul(c, .)` nodes that close a softcap: scalar factors that are NOT residual scales."""
    return {node.index for node in nodes if node.op_type == "Mul" and _softcap_behind(graph, node.outputs[0])}


def _glu_by_dataflow(graph: Graph, nodes: list[Node]) -> tuple[Node, Node, Node, float] | None:
    """r25: a block's (gate, up, down, softcap) found by dataflow, or None if not exactly one GLU is seen.

    The shape rule in `_read_block` needs dff > d_model; at dff == d_model every projection is square and below it
    the wide / narrow roles swap. gate = the Gemm behind a Sigmoid; up = another Gemm reading the same tensor whose
    output meets the gated value in a Mul; down = the Gemm consuming that product.

    BT6-test `ffn_softcap`: both branches pass through `c * tanh(. / c)` before they meet. The gate's cap sits
    between the Sigmoid and the product, the up branch's between its Gemm and the product; both are stepped through
    and must carry one c. softcap = 0.0 on an uncapped GLU.
    """
    consumers: dict[str, list[Node]] = {}
    for node in nodes:
        for name in node.inputs:
            consumers.setdefault(name, []).append(node)
    passthrough = {"Reshape", "Transpose", "Squeeze", "Unsqueeze", "Identity"}
    found: list[tuple[Node, Node, Node, float]] = []
    for sigmoid in nodes:
        if sigmoid.op_type != "Sigmoid":
            continue
        gate = _trace_to_gemm(graph, sigmoid.inputs[0])
        if gate is None:
            continue
        gate_source = _source_tensor(graph, gate.inputs[0])
        up: Node | None = None
        down: Node | None = None
        caps: set[float] = set()
        gated = sigmoid.outputs[0]
        # A capped gate: Sigmoid -> Div(c) -> Tanh -> Mul(c). Step to the capped value and remember c.
        for quotient in consumers.get(gated, []):
            tangents = consumers.get(quotient.outputs[0], []) if quotient.op_type == "Div" else []
            closing = consumers.get(tangents[0].outputs[0], []) if len(tangents) == 1 else []
            capped = _softcap_behind(graph, closing[0].outputs[0]) if len(closing) == 1 else None
            if capped is not None and capped[0] == gated:
                gated = closing[0].outputs[0]
                caps.add(capped[1])
        frontier = [gated]
        for _ in range(5):
            following: list[str] = []
            for tensor in frontier:
                for consumer in consumers.get(tensor, []):
                    if consumer.op_type == "Mul":
                        for other in consumer.inputs:
                            if other == tensor:
                                continue
                            capped = _softcap_behind(graph, other)
                            candidate = _trace_to_gemm(graph, capped[0] if capped is not None else other)
                            if (candidate is not None and candidate is not gate
                                    and _source_tensor(graph, candidate.inputs[0]) == gate_source):
                                up = candidate
                                # One cap or none: a capped gate needs a capped up branch with the same c.
                                caps.add(capped[1] if capped is not None else 0.0)
                                if gated == sigmoid.outputs[0]:
                                    caps.add(0.0)
                        following.append(consumer.outputs[0])
                    elif consumer.op_type in passthrough:
                        following.append(consumer.outputs[0])
                    elif consumer.op_type == "Gemm" and up is not None:
                        down = consumer
            if down is not None or not following:
                break
            frontier = following
        if up is not None and down is not None:
            if len(caps) != 1:
                message = (f"the GLU behind Sigmoid {sigmoid.index} caps its branches unevenly ({sorted(caps)}); "
                           "`ffn_softcap` puts one c on both")
                raise ArchitectureError(message)
            found.append((gate, up, down, caps.pop()))
    return found[0] if len(found) == 1 else None


def _read_block(graph: Graph, start: int, end: int, exclude: frozenset[int] = frozenset()) -> EncoderBlock:
    """Recover one encoder block's weight names from its node range.

    `exclude` (R0) holds the node indices of an EGT2 block's edge-stream terms -- the gate's Sigmoid and the
    expanded edge_e / edge_m / edge_g tables -- which `EgtBlock` owns; empty on the static family.
    """
    nodes = [node for node in graph.nodes[start : end + 1] if node.index not in exclude]
    smolgen, smolgen_nodes = _read_smolgen(graph, nodes, start=start, end=end)
    gemms = [node for node in nodes if node.op_type == "Gemm" and node.index not in smolgen_nodes]
    square_gemms = []
    wide_gemms = []
    narrow_gemms = []
    for node in gemms:
        weight = _weight_of(graph, node)
        rows, columns = weight.dims
        if rows == columns:
            square_gemms.append(node)
        elif columns > rows:
            wide_gemms.append(node)
        else:
            narrow_gemms.append(node)

    # r25: take the FFN's three projections by DATAFLOW first. The shape rule above needs dff > d_model; the ruled
    # BT6 sponsor shape has dff == d_model (all seven projections square) and thinner FFNs swap wide and narrow.
    glu = _glu_by_dataflow(graph, [node for node in nodes if node.index not in smolgen_nodes])
    if glu is not None:
        square_gemms = [node for node in square_gemms if all(node is not member for member in glu[:3])]
        wide_gemms = [glu[0], glu[1]]
        narrow_gemms = [glu[2]]
    softcap = glu[3] if glu is not None else 0.0
    # A softcap the dataflow walk did not account for would be served as an uncapped GLU, silently. Every Tanh in a
    # block belongs to the cap (two per capped GLU); smolgen and the attention have none.
    tangents = sum(1 for node in nodes if node.op_type == "Tanh" and node.index not in smolgen_nodes)
    if tangents != (2 if softcap > 0.0 else 0):
        message = (f"block at nodes {start}-{end}: {tangents} Tanh nodes but the FFN's softcap reads as {softcap}; "
                   "an unrecognised cap would be dropped silently, so the block is refused")
        raise ArchitectureError(message)
    sigmoids = [node for node in nodes if node.op_type == "Sigmoid" and node.index not in smolgen_nodes]
    # O: a Sigmoid behind a square projection is the attention output gate; the FFN's sits behind a wide one.
    output_gate, glu_sigmoids = _read_output_gate(graph, nodes, sigmoids, square_gemms, start=start, end=end)
    if output_gate is not None:
        square_gemms = [node for node in square_gemms if node is not output_gate.gemm]
    expected_square = 4
    if len(square_gemms) != expected_square:
        message = (
            f"block at nodes {start}-{end}: expected {expected_square} square "
            f"projections{' besides the output gate' if output_gate is not None else ''}, "
            f"found {len(square_gemms)}"
        )
        raise ArchitectureError(message)
    query, key, value, output = square_gemms
    if output_gate is not None:
        _check_output_gate(graph, output_gate, query, output, start=start, end=end)

    if len(glu_sigmoids) != 1:
        message = (
            f"block at nodes {start}-{end}: expected exactly one Sigmoid gate in the FFN, "
            f"found {len(glu_sigmoids)} -- read the gate from the graph, never from "
            "the activation enum's name"
        )
        raise ArchitectureError(message)
    gate_gemm = _trace_to_gemm(graph, glu_sigmoids[0].inputs[0])
    if gate_gemm is None or gate_gemm not in wide_gemms:
        message = f"block at nodes {start}-{end}: the gate projection is not a Gemm"
        raise ArchitectureError(message)
    up_candidates = [node for node in wide_gemms if node is not gate_gemm]
    if len(up_candidates) != 1 or len(narrow_gemms) != 1:
        message = (
            f"block at nodes {start}-{end}: expected one gate, one up and one "
            "down projection in the FFN"
        )
        raise ArchitectureError(message)
    up_gemm = up_candidates[0]
    down_gemm = narrow_gemms[0]

    norms = [
        node for node in nodes if node.op_type == "LayerNormalization" and node.index not in smolgen_nodes
    ]
    if len(norms) != _EXPECTED_LAYER_NORMS_PER_BLOCK:
        message = f"block at nodes {start}-{end}: expected two layer norms"
        raise ArchitectureError(message)
    ln1, ln2 = norms

    cap_products = _softcap_products(graph, nodes)
    alphas = [
        node.inputs[1]
        for node in nodes
        if node.op_type == "Mul" and _scalar(graph, node.inputs[1]) is not None
        and (output_gate is None or node.index != output_gate.doubling)
        and node.index not in cap_products
    ]
    expected_alphas = 2
    if len(alphas) != expected_alphas:
        message = (
            f"block at nodes {start}-{end}: expected two residual scale factors, "
            f"found {len(alphas)}"
        )
        raise ArchitectureError(message)

    edge, pair_key, pair_query, mix = _read_logit_terms(
        graph, nodes, key_gemm=key, query_gemm=query, start=start, end=end
    )
    return EncoderBlock(
        query_weight=query.inputs[1],
        query_bias=query.inputs[2],
        key_weight=key.inputs[1],
        key_bias=key.inputs[2],
        value_weight=value.inputs[1],
        value_bias=value.inputs[2],
        output_weight=output.inputs[1],
        output_bias=output.inputs[2],
        attention_alpha=alphas[0],
        ln1_scale=ln1.inputs[1],
        ln1_bias=ln1.inputs[2],
        ffn_gate_weight=gate_gemm.inputs[1],
        ffn_up_weight=up_gemm.inputs[1],
        ffn_up_bias=up_gemm.inputs[2],
        ffn_down_weight=down_gemm.inputs[1],
        ffn_down_bias=down_gemm.inputs[2],
        ffn_alpha=alphas[1],
        ln2_scale=ln2.inputs[1],
        ln2_bias=ln2.inputs[2],
        edge_coefficients=edge,
        pair_key=pair_key,
        pair_query=pair_query,
        mix_coefficients=mix,
        attention_scale=_read_attention_scale(graph, nodes, start=start, end=end),
        smolgen=smolgen,
        gate_weight=output_gate.gemm.inputs[1] if output_gate is not None else None,
        gate_bias=output_gate.gemm.inputs[2] if output_gate is not None else None,
        gate_scale=output_gate.scale if output_gate is not None else 0.0,
        gate_node=output_gate.product if output_gate is not None else -1,
        ffn_softcap=softcap,
    )


def _read_logit_terms(  # noqa: C901, PLR0912, PLR0913
    graph: Graph,
    nodes: list[Node],
    *,
    key_gemm: Node,
    query_gemm: Node,
    start: int,
    end: int,
) -> tuple[str, str, str, str | None]:
    """Return the four learned logit tables, told apart by what they contract.

    Two of them have the identical shape `[1, heads, edge_channels, head_dim]`
    and differ only in whether they meet the keys or the queries -- the
    column-indexed and row-indexed halves of the pair term. Nothing in the name
    or the shape says which, so each is followed to the projection it multiplies
    and identified by that.

    The two rank-3 tables are found only after the head count is known from the
    rank-4 pair tables. Filtering on `dims[0] == 1` alone would sweep in the
    board-geometry constants, which are `[1, 64, 64]` and would pass every shape
    test a naive rule could apply.
    """
    pairs: list[tuple[Tensor, Node]] = []
    for node in nodes:
        if node.op_type != "Expand":
            continue
        tensor = _initializer(graph, node.inputs[0])
        rank_of_pair_table = 4
        if (
            tensor is not None
            and len(tensor.dims) == rank_of_pair_table
            and tensor.dims[0] == 1
        ):
            pairs.append((tensor, node))

    expected_pairs = 2
    if len(pairs) != expected_pairs:
        message = (
            f"block at nodes {start}-{end}: expected two pair tables, found "
            f"{len(pairs)}"
        )
        raise ArchitectureError(message)
    heads = pairs[0][0].dims[1]
    edge_channels = pairs[0][0].dims[2]

    vectors: list[Tensor] = []
    for node in nodes:
        if node.op_type != "Expand":
            continue
        tensor = _initializer(graph, node.inputs[0])
        rank_of_vector_table = 3
        if (
            tensor is not None
            and len(tensor.dims) == rank_of_vector_table
            and tensor.dims[0] == 1
            and tensor.dims[1] == heads
        ):
            vectors.append(tensor)
    if len(vectors) == 1 and vectors[0].dims[2] == edge_channels:
        # No pair stream (the smolgen twin): the edge table stands alone.
        pass
    elif len(vectors) != expected_pairs:
        message = (
            f"block at nodes {start}-{end}: expected two per-head coefficient "
            f"tables, found {len(vectors)}"
        )
        raise ArchitectureError(message)
    # The edge table is indexed by edge channel and the mix table by the wider
    # mix stream. Where the two widths coincide, graph order decides -- the edge
    # term is added to the scores first.
    if len(vectors) == 1:
        edge_table, mix_table = vectors[0], None
    elif vectors[1].dims[2] == edge_channels and vectors[0].dims[2] != edge_channels:
        mix_table, edge_table = vectors
    else:
        edge_table, mix_table = vectors

    resolved: dict[str, str] = {}
    for tensor, node in pairs:
        projection = _projection_behind(graph, node, nodes)
        if projection is key_gemm:
            resolved["pair_key"] = tensor.name
        elif projection is query_gemm:
            resolved["pair_query"] = tensor.name
        else:
            message = (
                f"block at nodes {start}-{end}: pair table {tensor.name} "
                "contracts with neither the key nor the query projection"
            )
            raise ArchitectureError(message)
    if len(resolved) != expected_pairs:
        message = (
            f"block at nodes {start}-{end}: both pair tables resolved to the "
            "same projection"
        )
        raise ArchitectureError(message)
    return (
        edge_table.name,
        resolved["pair_key"],
        resolved["pair_query"],
        mix_table.name if mix_table is not None else None,
    )


def _projection_behind(graph: Graph, expand: Node, nodes: list[Node]) -> Node | None:
    """Return the Q/K Gemm the expanded pair table is contracted against."""
    produced = expand.outputs[0]
    frontier = {produced}
    passthrough = {"Transpose", "Reshape", "Squeeze", "Unsqueeze", "Identity"}
    for node in nodes:
        if node.op_type in passthrough and node.inputs[0] in frontier:
            frontier.add(node.outputs[0])
            continue
        if node.op_type != "MatMul":
            continue
        others = [name for name in node.inputs if name not in frontier]
        if len(others) == len(node.inputs):
            continue
        for other in others:
            projection = _trace_to_gemm(graph, other)
            if projection is not None:
                return projection
    return None


def _read_attention_scale(
    graph: Graph, nodes: list[Node], *, start: int, end: int
) -> float:
    """Return the reciprocal square root the graph divides the scores by."""
    for node in nodes:
        if node.op_type != "Div":
            continue
        producer = graph.produced_by(node.inputs[1])
        if producer is None or producer.op_type != "Sqrt":
            continue
        source = graph.produced_by(producer.inputs[0])
        name = source.inputs[0] if source is not None else producer.inputs[0]
        tensor = _initializer(graph, name)
        if tensor is None:
            continue
        value = int.from_bytes(tensor.raw_data[:4], "little")
        if value <= 0:
            message = f"block at nodes {start}-{end}: non-positive attention scale"
            raise ArchitectureError(message)
        return 1.0 / math.sqrt(value)
    message = f"block at nodes {start}-{end}: no scaled dot product found"
    raise ArchitectureError(message)


def _read_embedding(graph: Graph, limit: int) -> Embedding:
    """Recover the input embedding and its gated FFN from the prologue."""
    nodes = graph.nodes[:limit]
    gemms = [node for node in nodes if node.op_type == "Gemm"]
    expected_gemms = 5
    if len(gemms) != expected_gemms:
        message = (
            f"the prologue has {len(gemms)} dense projections, expected "
            f"{expected_gemms} (preprocess, embedding, gate, up, down)"
        )
        raise ArchitectureError(message)
    preprocess, embedding, first_wide, second_wide, down = gemms

    sigmoids = [node for node in nodes if node.op_type == "Sigmoid"]
    if len(sigmoids) != 1:
        message = "the embedding FFN does not have exactly one Sigmoid gate"
        raise ArchitectureError(message)
    gate = _trace_to_gemm(graph, sigmoids[0].inputs[0])
    if gate is first_wide:
        up = second_wide
    elif gate is second_wide:
        up = first_wide
    else:
        message = "the embedding FFN gate is not one of its wide projections"
        raise ArchitectureError(message)

    # BT6-test `ffn_softcap` also caps the embedding FFN (`embedding.py` passes the same default). Mish accounts
    # for one Tanh, a capped GLU for two more; any other count is a graph this reader does not understand.
    glu = _glu_by_dataflow(graph, nodes)
    softcap = glu[3] if glu is not None else 0.0
    if glu is not None and (glu[0] is not gate or glu[1] is not up or glu[2] is not down):
        message = "the embedding FFN's dataflow disagrees with its projection order"
        raise ArchitectureError(message)
    tangents = sum(1 for node in nodes if node.op_type == "Tanh")
    if tangents != 1 + (2 if softcap > 0.0 else 0):
        message = (f"the prologue has {tangents} Tanh nodes: expected Mish's one"
                   f"{' plus the two of the capped GLU' if softcap > 0.0 else ''}")
        raise ArchitectureError(message)

    norms = [node for node in nodes if node.op_type == "LayerNormalization"]
    if len(norms) != _EXPECTED_LAYER_NORMS_PER_BLOCK:
        message = f"the prologue has {len(norms)} layer norms, expected two"
        raise ArchitectureError(message)
    ln0, ln1 = norms

    gates = [
        node.inputs[0]
        for node in nodes
        if node.op_type == "Relu" and _initializer(graph, node.inputs[0]) is not None
    ]
    if len(gates) != 1:
        message = "the input multiplicative gate is missing from the prologue"
        raise ArchitectureError(message)

    positional = _find_positional(graph, nodes)
    cap_products = _softcap_products(graph, nodes)
    alphas = [
        node.inputs[1]
        for node in nodes
        if node.op_type == "Mul" and _scalar(graph, node.inputs[1]) is not None and node.index not in cap_products
    ]
    if len(alphas) != 1:
        message = (
            f"the prologue has {len(alphas)} residual scale factors, expected one"
        )
        raise ArchitectureError(message)

    return Embedding(
        preprocess_weight=preprocess.inputs[1],
        preprocess_bias=preprocess.inputs[2],
        embedding_weight=embedding.inputs[1],
        embedding_bias=embedding.inputs[2],
        ln0_scale=ln0.inputs[1],
        ln0_bias=ln0.inputs[2],
        input_gate=gates[0],
        positional=positional,
        ffn_gate_weight=gate.inputs[1],
        ffn_up_weight=up.inputs[1],
        ffn_up_bias=up.inputs[2],
        ffn_down_weight=down.inputs[1],
        ffn_down_bias=down.inputs[2],
        ffn_alpha=alphas[0],
        ln1_scale=ln1.inputs[1],
        ln1_bias=ln1.inputs[2],
        activation="mish",
        ffn_softcap=softcap,
    )


def _find_positional(graph: Graph, nodes: list[Node]) -> str:
    """Return the `[1, tokens, d_model]` table added to the gated embedding."""
    for node in nodes:
        if node.op_type != "Add":
            continue
        for name in node.inputs:
            tensor = _initializer(graph, name)
            rank = 3
            if tensor is not None and len(tensor.dims) == rank and tensor.dims[0] == 1:
                return name
    message = "the positional embedding table is missing from the prologue"
    raise ArchitectureError(message)


def read_network(graph: Graph) -> LabNetwork:
    """Recover a lab network's architecture and weight names from its graph."""
    prologue_end, ranges, style, final_norm = _block_ranges(graph)
    embedding = _read_embedding(graph, prologue_end + 1)
    exclusions: list[frozenset[int]] = [frozenset()] * len(ranges)
    egt = egt_pair = None
    agreed: list[tuple[str, str]] = []
    if any(_is_door_softmax(graph, node) for node in graph.nodes):
        # Round 26b: pre-norm EGT2 (P1 / P1g). The site walk needs only each block's end -- the FFN residual Add
        # under pre-norm, LN2 under post-norm -- and its own range must stay clear of Gemms and norms either way.
        egt, egt_pair, ranges, exclusions, agreed = _read_egt(graph, ranges)
    blocks = tuple(
        _read_block(graph, start, end, exclude) for (start, end), exclude in zip(ranges, exclusions, strict=True)
    )
    # O: a pre-norm tower's heads read the final norm, so the heads region starts after it.
    heads_start = final_norm.node + 1 if final_norm is not None else ranges[-1][1] + 1
    heads = _read_heads(graph, heads_start)
    architecture = _derive_architecture(graph, embedding, blocks, style=style, heads=heads)
    _assert_blocks_agree(graph, blocks, architecture)
    if egt is not None:
        _assert_egt_agrees(graph, blocks, egt, architecture, agreed)
    if egt is not None:
        pair = egt_pair
    else:
        pair = _read_pair_tables(graph, architecture) if architecture.mix_channels else None
    policy_divisor = _read_policy_divisor(graph, heads_start)
    return LabNetwork(
        architecture=architecture,
        embedding=embedding,
        blocks=blocks,
        heads=heads,
        pair=pair,
        policy_divisor=policy_divisor,
        egt=egt,
        final_norm=final_norm,
    )


def _derive_architecture(
    graph: Graph, embedding: Embedding, blocks: tuple[EncoderBlock, ...], *, style: BlockStyle = "postnorm",
    heads: "Heads | None" = None,
) -> Architecture:
    """Derive every dimension from the shapes the recovered names carry."""
    if not blocks:
        message = "the graph has no encoder blocks"
        raise ArchitectureError(message)
    first = blocks[0]
    d_model = graph.initializers[first.query_weight].dims[0]
    # The preprocess dense writes [tokens * dense] from the twelve piece planes; the embedding projection then reads
    # 112 + dense per square. Both constants must tell the same `dense` (the lab's `embedding.dense_size`).
    preprocess_out = graph.initializers[embedding.preprocess_weight].dims[1]
    embedding_in = graph.initializers[embedding.embedding_weight].dims[0]
    if preprocess_out % _SQUARE_COUNT or embedding_in != _INPUT_CHANNELS + preprocess_out // _SQUARE_COUNT:
        message = (f"the preprocess dense writes {preprocess_out} values but the embedding projection reads "
                   f"{embedding_in}; expected {_INPUT_CHANNELS} + a per-square width")
        raise ArchitectureError(message)
    if embedding.ffn_softcap != first.ffn_softcap:
        message = (f"the embedding FFN's softcap {embedding.ffn_softcap} differs from the blocks' "
                   f"{first.ffn_softcap}; the lab sets one `ffn_softcap` for both")
        raise ArchitectureError(message)
    policy_width = graph.initializers[heads.policy_embedding_weight].dims[1] if heads is not None else d_model
    ffn_hidden = graph.initializers[first.ffn_gate_weight].dims[1]
    embedding_hidden = graph.initializers[embedding.ffn_gate_weight].dims[1]
    _, heads, edge_channels, head_dim = graph.initializers[first.pair_key].dims
    mix_channels = (
        graph.initializers[first.mix_coefficients].dims[2] if first.mix_coefficients is not None else 0
    )
    smolgen_channels = smolgen_hidden = smolgen_gen = 0
    if first.smolgen is not None:
        smolgen_channels = graph.initializers[first.smolgen.compress_weight].dims[1]
        smolgen_hidden = graph.initializers[first.smolgen.dense1_weight].dims[1]
        smolgen_gen = graph.initializers[first.smolgen.weight_gen].dims[0]
    if heads * head_dim != d_model:
        message = (
            f"heads {heads} x head_dim {head_dim} does not reconstruct d_model "
            f"{d_model}"
        )
        raise ArchitectureError(message)
    return Architecture(
        blocks=len(blocks),
        d_model=d_model,
        heads=heads,
        head_dim=head_dim,
        ffn_hidden=ffn_hidden,
        embedding_ffn_hidden=embedding_hidden,
        edge_channels=edge_channels,
        mix_channels=mix_channels,
        smolgen_channels=smolgen_channels,
        smolgen_hidden=smolgen_hidden,
        smolgen_gen=smolgen_gen,
        block_style=style,
        output_gate=first.gate_weight is not None,
        ffn_softcap=first.ffn_softcap,
        embedding_dense=preprocess_out // _SQUARE_COUNT,
        policy_width=policy_width,
    )


def _assert_blocks_agree(
    graph: Graph, blocks: tuple[EncoderBlock, ...], architecture: Architecture
) -> None:
    """Require every block to carry the shapes the first one implied.

    This is the check that makes a structural recovery trustworthy: the pattern
    matched fifteen independent times, and every match produced tensors of the
    same shape. A rule that latched onto the wrong node would have to do so
    identically in all fifteen blocks to get past it.
    """
    expected = {
        "query_weight": (architecture.d_model, architecture.d_model),
        "key_weight": (architecture.d_model, architecture.d_model),
        "value_weight": (architecture.d_model, architecture.d_model),
        "output_weight": (architecture.d_model, architecture.d_model),
        "ffn_gate_weight": (architecture.d_model, architecture.ffn_hidden),
        "ffn_up_weight": (architecture.d_model, architecture.ffn_hidden),
        "ffn_down_weight": (architecture.ffn_hidden, architecture.d_model),
        "query_bias": (architecture.d_model,),
        "key_bias": (architecture.d_model,),
        "value_bias": (architecture.d_model,),
        "output_bias": (architecture.d_model,),
        "ffn_up_bias": (architecture.ffn_hidden,),
        "ffn_down_bias": (architecture.d_model,),
        "ln1_scale": (architecture.d_model,),
        "ln1_bias": (architecture.d_model,),
        "ln2_scale": (architecture.d_model,),
        "ln2_bias": (architecture.d_model,),
        "edge_coefficients": (1, architecture.heads, architecture.edge_channels),
        "pair_key": (
            1,
            architecture.heads,
            architecture.edge_channels,
            architecture.head_dim,
        ),
        "pair_query": (
            1,
            architecture.heads,
            architecture.edge_channels,
            architecture.head_dim,
        ),
        "mix_coefficients": (1, architecture.heads, architecture.mix_channels),
    }
    for index, block in enumerate(blocks):
        for attribute, shape in expected.items():
            name = getattr(block, attribute)
            if name is None and attribute == "mix_coefficients" and architecture.mix_channels == 0:
                continue
            actual = graph.initializers[name].dims
            if actual != shape:
                message = (
                    f"block {index}: {attribute} ({name}) has shape {actual}, "
                    f"expected {shape}"
                )
                raise ArchitectureError(message)
        if block.attention_scale != blocks[0].attention_scale:
            message = f"block {index}: attention scale differs from block 0"
            raise ArchitectureError(message)
        if block.ffn_softcap != architecture.ffn_softcap:
            message = f"block {index}: FFN softcap {block.ffn_softcap} differs from block 0's {architecture.ffn_softcap}"
            raise ArchitectureError(message)
        _assert_smolgen_agrees(graph, index, block, blocks[0], architecture)
        _assert_gate_agrees(graph, index, block, blocks[0], architecture)


def _assert_smolgen_agrees(
    graph: Graph, index: int, block: EncoderBlock, first: EncoderBlock, architecture: Architecture
) -> None:
    """Hold every block's smolgen chain to the shapes block 0 implied, weight_gen shared."""
    if (block.smolgen is None) != (architecture.smolgen_gen == 0):
        message = f"block {index}: smolgen is present in some blocks only"
        raise ArchitectureError(message)
    if block.smolgen is None:
        return
    s = architecture
    generated = s.heads * s.smolgen_gen
    expected = {
        "compress_weight": (s.d_model, s.smolgen_channels),
        "dense1_weight": (s.tokens * s.smolgen_channels, s.smolgen_hidden),
        "dense1_bias": (s.smolgen_hidden,),
        "ln1_scale": (s.smolgen_hidden,),
        "ln1_bias": (s.smolgen_hidden,),
        "dense2_weight": (s.smolgen_hidden, generated),
        "dense2_bias": (generated,),
        "ln2_scale": (generated,),
        "ln2_bias": (generated,),
        "weight_gen": (s.smolgen_gen, s.tokens * s.tokens),
    }
    for attribute, shape in expected.items():
        name = getattr(block.smolgen, attribute)
        actual = tuple(graph.initializers[name].dims)
        if actual != shape:
            message = f"block {index}: smolgen {attribute} ({name}) has shape {actual}, expected {shape}"
            raise ArchitectureError(message)
    if block.smolgen.weight_gen != first.smolgen.weight_gen:
        message = f"block {index}: weight_gen {block.smolgen.weight_gen} is not the shared one"
        raise ArchitectureError(message)


_MIX_TABLE_RANK = 2
_GEMM_INPUTS_WITH_BIAS = 3
_WDL_OUTPUTS = 3
_MOVES_LEFT_OUTPUTS = 1
_PROMOTION_OUTPUTS = 4


def _read_heads(graph: Graph, start: int) -> Heads:
    """Recover the output heads from the nodes after the encoder tower.

    Each projection is classified by the width it produces, not by the order it
    appears in, so an exporter that emits the heads in a different sequence
    still resolves. The two widths that could collide -- the value and
    moves-left square projections both narrow `d_model` -- are separated by the
    head each one leads to.
    """
    nodes = graph.nodes[start:]
    gemms = [node for node in nodes if node.op_type == "Gemm"]
    by_width: dict[int, list[Node]] = {}
    for node in gemms:
        by_width.setdefault(_weight_of(graph, node).dims[1], []).append(node)

    value_output = _only(by_width.get(_WDL_OUTPUTS, []), "the WDL projection")
    moves_output = _only(
        by_width.get(_MOVES_LEFT_OUTPUTS, []), "the moves-left projection"
    )
    promotion = _only(
        [
            node
            for node in by_width.get(_PROMOTION_OUTPUTS, [])
            if len(node.inputs) < _GEMM_INPUTS_WITH_BIAS
        ],
        "the promotion projection",
    )

    d_model = _weight_of(graph, gemms[0]).dims[0]

    # _FLATTENED_VALUE_BY_HEAD: the value head and the moves-left head BOTH flatten the 64 squares,
    # so both produce a projection whose input width is not d_model -- 64 x 128 for the value head and
    # 64 x num_channels (512 at num_channels 8) for the moves-left head. Those two are told apart by
    # the head each one leads to, exactly as the square projections below are; the width test alone
    # only works when the moves-left flattened width happens to equal d_model, which is true at
    # d_model 512 and false at every other trunk width.
    flattened = [
        node
        for node in gemms
        if _weight_of(graph, node).dims[0] != d_model
        and node is not value_output
        and node is not moves_output
    ]
    if len(flattened) > 1:
        flattened = [node for node in flattened if _feeds(node, value_output, nodes)]
    value_hidden = _only(flattened, "the flattened value projection")
    value_width = _weight_of(graph, value_hidden).dims[1]
    value_square = _only(
        [
            node
            for node in by_width.get(value_width, [])
            if node is not value_hidden and _feeds(node, value_hidden, nodes)
        ],
        "the per-square value projection",
    )
    moves_hidden = _only(
        [
            node
            for node in by_width.get(value_width, [])
            if node is not value_hidden and node is not value_square
        ],
        "the hidden moves-left projection",
    )
    # The policy head's three projections produce ITS width, which the promotion table names (`[policy, 4]`):
    # d_model on the 512 lab nets, `shared_policy_embedding_size` = 512 on the 1024 sponsor net. The value and
    # moves-left projections are already named, so a collision of widths cannot mislabel them.
    policy_width = _weight_of(graph, promotion).dims[0]
    taken = {value_square.index, value_hidden.index, moves_hidden.index}
    policy = [node for node in by_width.get(policy_width, []) if node.index not in taken]
    expected_policy = 3
    if len(policy) != expected_policy:
        message = (
            f"expected {expected_policy} policy projections of width {policy_width} (embedding, "
            f"query, key), found {len(policy)}"
        )
        raise ArchitectureError(message)
    policy_embedding, policy_query, policy_key = policy
    widths = [_weight_of(graph, node).dims[0] for node in policy]
    if widths != [d_model, policy_width, policy_width]:
        message = (f"the policy projections read widths {widths}; expected the embedding to read the trunk "
                   f"({d_model}) and the query and key to read the embedding ({policy_width})")
        raise ArchitectureError(message)
    claimed = {
        node.index
        for node in (
            value_square,
            value_hidden,
            value_output,
            moves_hidden,
            moves_output,
            promotion,
            policy_embedding,
            policy_query,
            policy_key,
        )
    }
    moves_square = _only(
        [node for node in gemms if node.index not in claimed],
        "the per-square moves-left projection",
    )

    mix_candidates = (
        [
            graph.initializers[node.inputs[0]].name
            for node in nodes
            if node.op_type == "Expand"
            and _initializer(graph, node.inputs[0]) is not None
            and len(graph.initializers[node.inputs[0]].dims) == _MIX_TABLE_RANK
            and graph.initializers[node.inputs[0]].dims[0] == 1
        ]
    )
    # Absent without the pair stream: that policy head is plain Q K^T.
    mix = _only(mix_candidates, "the policy static-mix table") if mix_candidates else None
    policy_map = _only(
        sorted(
            {
                name
                for node in nodes
                for name in node.inputs
                if _initializer(graph, name) is not None
                and graph.initializers[name].data_type == INT32
                and len(graph.initializers[name].dims) == 1
            }
        ),
        "the policy move map",
    )
    if not any(node.op_type == "Tanh" for node in nodes):
        message = "the head activation is not Mish; no Tanh after the tower"
        raise ArchitectureError(message)

    return Heads(
        value_square_weight=value_square.inputs[1],
        value_square_bias=value_square.inputs[2],
        value_hidden_weight=value_hidden.inputs[1],
        value_hidden_bias=value_hidden.inputs[2],
        value_output_weight=value_output.inputs[1],
        value_output_bias=value_output.inputs[2],
        policy_embedding_weight=policy_embedding.inputs[1],
        policy_embedding_bias=policy_embedding.inputs[2],
        policy_query_weight=policy_query.inputs[1],
        policy_query_bias=policy_query.inputs[2],
        policy_key_weight=policy_key.inputs[1],
        policy_key_bias=policy_key.inputs[2],
        policy_promotion_weight=promotion.inputs[1],
        policy_mix_coefficients=mix,
        policy_map=policy_map,
        moves_square_weight=moves_square.inputs[1],
        moves_square_bias=moves_square.inputs[2],
        moves_hidden_weight=moves_hidden.inputs[1],
        moves_hidden_bias=moves_hidden.inputs[2],
        moves_output_weight=moves_output.inputs[1],
        moves_output_bias=moves_output.inputs[2],
        activation="mish",
    )


def _only(candidates: list, description: str) -> object:
    """Return the single candidate, or say how many there were instead."""
    if len(candidates) != 1:
        message = (
            f"expected exactly one node for {description}, found "
            f"{len(candidates)}"
        )
        raise ArchitectureError(message)
    return candidates[0]


def _feeds(source: Node, target: Node, nodes: list[Node]) -> bool:
    """Return whether `source`'s value reaches `target` through reshapes only."""
    reachable = set(source.outputs)
    passthrough = {"Reshape", "Transpose", "Squeeze", "Unsqueeze", "Identity"}
    for node in nodes:
        if (node.op_type in passthrough and node.inputs[0] in reachable) or (
            node.op_type not in passthrough
            and node.op_type != "Gemm"
            # Mish is expanded into a dozen elementwise nodes; step through any
            # of them so long as the value keeps flowing forward.
            and any(name in reachable for name in node.inputs)
        ):
            reachable.update(node.outputs)
    return target.inputs[0] in reachable


def _read_pair_tables(graph: Graph, architecture: Architecture) -> PairTables:
    """Recover the pair stream's constants from the structure that uses them.

    They cannot be looked up by position: `_block_ranges` counts the whole
    attack-graph computation (~2,300 nodes) into block 0's range. Each constant
    is instead the unique tensor with its role's shape in its role's place:

    * `channel_mix` -- the one `[1, edge_channels, mix_channels]` table that is
      Expanded (the per-head tables are `[1, heads, ...]`);
    * `offset_table` -- the `[bins, mix_channels]` table a Gather reads, and
      `relative_index` the `tokens x tokens` integer table behind its indices;
    * `epsilon` -- the scalar added to the mean of a square before its root.
    """
    producers = {output: node for node in graph.nodes for output in node.outputs}
    tokens = architecture.tokens
    edges, mix = architecture.edge_channels, architecture.mix_channels

    channel_mixes = set()
    for node in graph.nodes:
        if node.op_type != "Expand":
            continue
        tensor = _initializer(graph, node.inputs[0])
        if tensor is not None and tuple(tensor.dims) == (1, edges, mix) and edges != architecture.heads:
            channel_mixes.add(tensor.name)
    if len(channel_mixes) != 1:
        message = f"expected one [1, {edges}, {mix}] channel-mix table, found {sorted(channel_mixes)}"
        raise ArchitectureError(message)

    offset_tables: set[str] = set()
    relative_indices: set[str] = set()
    for node in graph.nodes:
        if node.op_type != "Gather":
            continue
        tensor = _initializer(graph, node.inputs[0])
        rank_of_offset_table = 2
        if tensor is None or len(tensor.dims) != rank_of_offset_table or tensor.dims[1] != mix:
            continue
        offset_tables.add(tensor.name)
        relative_indices |= _integer_tables_behind(graph, producers, node.inputs[1], tokens * tokens)
    if len(offset_tables) != 1 or len(relative_indices) != 1:
        message = (
            f"expected one offset table and one relative index, found "
            f"{sorted(offset_tables)} and {sorted(relative_indices)}"
        )
        raise ArchitectureError(message)

    epsilons: set[float] = set()
    for node in graph.nodes:
        if node.op_type != "ReduceMean":
            continue
        square = producers.get(node.inputs[0])
        if square is None or square.op_type != "Mul" or square.inputs[0] != square.inputs[1]:
            continue
        for consumer in graph.nodes:
            if consumer.op_type == "Add" and node.outputs[0] in consumer.inputs:
                for name in consumer.inputs:
                    value = _scalar(graph, name)
                    if value is not None:
                        epsilons.add(value)
    if len(epsilons) != 1:
        message = f"expected one pair-norm epsilon, found {sorted(epsilons)}"
        raise ArchitectureError(message)

    return PairTables(
        channel_mix=channel_mixes.pop(),
        offset_table=offset_tables.pop(),
        relative_index=relative_indices.pop(),
        epsilon=epsilons.pop(),
    )


def _integer_tables_behind(
    graph: Graph, producers: dict[str, Node], name: str, element_count: int, depth: int = 8
) -> set[str]:
    """Return the integer initializers of `element_count` elements feeding `name`."""
    tensor = _initializer(graph, name)
    if tensor is not None:
        is_integer = tensor.data_type in (INT32, INT64)
        return {tensor.name} if is_integer and tensor.element_count == element_count else set()
    node = producers.get(name)
    if node is None or depth == 0:
        return set()
    found: set[str] = set()
    for source in node.inputs:
        found |= _integer_tables_behind(graph, producers, source, element_count, depth - 1)
    return found


def _read_policy_divisor(graph: Graph, start: int) -> float:
    """Return the scalar the policy logits are divided by, checked for promotions.

    In the heads region the only scalar divisions are the policy block's and the
    promotion block's, and the only scalar multiplication is the promotion
    offsets'. All three must agree: the served head relies on
    `((QK + bias) + offsets * c) / c == (QK + bias) / c + offsets`.
    """
    divisors: set[float] = set()
    multipliers: set[float] = set()
    for node in graph.nodes[start:]:
        if node.op_type == "Div":
            value = _scalar(graph, node.inputs[1])
            if value is not None:
                divisors.add(value)
        elif node.op_type == "Mul":
            for name in node.inputs:
                value = _scalar(graph, name)
                if value is not None:
                    multipliers.add(value)
    if len(divisors) != 1:
        message = f"expected one policy logit divisor in the heads, found {sorted(divisors)}"
        raise ArchitectureError(message)
    divisor = divisors.pop()
    if multipliers - {divisor}:
        message = (
            f"the promotion offsets are scaled by {sorted(multipliers)}, not by the policy "
            f"divisor {divisor}; the BT4 promotion kernel's arithmetic would not match"
        )
        raise ArchitectureError(message)
    return divisor


# ---------------------------------------------------------------- R0 (item E, round 20c): the EGT2 edge stream
# Recovered by structure, walking back from each attention Softmax as the item E map's `check_map.py` does, in
# both export forms: Einsum (gcap) and the Einsum->MatMul rewrite (the triplet exports).

_SHAPE_ONLY = {"Reshape", "Squeeze", "Unsqueeze", "Identity"}
_CHANNEL_EQUATIONS = {"zab,zbcd->zacd": False, "zab,zacd->zbcd": True}  # equation -> transposed
_TRIPLET_SPLIT_ROLES = {0: ("Softmax", 3), 1: ("Sigmoid", None), 2: ("Softmax", 2), 3: ("Sigmoid", None)}


def _require(condition: object, message: str) -> None:
    """Raise an ArchitectureError with `message` unless `condition` holds."""
    if not condition:
        raise ArchitectureError(message)


def _one_plus(graph: Graph, tensor: str) -> str | None:
    """Return `x` when `tensor` is `1 + x` (the door's `1 + edge_m.e`), else None."""
    node = graph.produced_by(tensor)
    if node is None or node.op_type != "Add" or len(node.inputs) != 2:  # noqa: PLR2004
        return None
    others = [name for name in node.inputs if _scalar(graph, name) != 1.0]
    return others[0] if len(others) == 1 else None


def _is_door_softmax(graph: Graph, node: Node) -> bool:
    """Report whether `node` is an EGT2 attention Softmax: fed by the door `H = T (1 + edge_m.e)`."""
    if node.op_type != "Softmax":
        return False
    door = graph.produced_by(node.inputs[0])
    if door is None or door.op_type != "Mul" or len(door.inputs) != 2:  # noqa: PLR2004
        return False
    return sum(_one_plus(graph, name) is not None for name in door.inputs) == 1


def _attention_softmaxes(graph: Graph) -> list[int]:
    """Return the indices of the Softmaxes that anchor encoder blocks.

    On the static family that is every Softmax (each is fed by the score sum). An EGT2 export also normalizes
    inside its triplet branches, so there only the door-fed Softmaxes count; `_read_egt` requires every other
    Softmax to be a triplet's.
    """
    consumed = {name for node in graph.nodes for name in node.inputs}
    # r25: a terminal Softmax is an output normalisation (the post-hoc WDL softmax), never a block anchor.
    softmaxes = [node for node in graph.nodes if node.op_type == "Softmax" and node.outputs[0] in consumed]
    doors = [node.index for node in softmaxes if _is_door_softmax(graph, node)]
    return doors or [node.index for node in softmaxes]


def _consumers(graph: Graph) -> dict[str, list[Node]]:
    """Return every tensor's consuming nodes, in graph order."""
    table: dict[str, list[Node]] = {}
    for node in graph.nodes:
        for name in node.inputs:
            table.setdefault(name, []).append(node)
    return table


def _column(graph: Graph, name: str) -> int | None:
    """Return `n` for a float `[1, n, 1, 1]` initializer (a per-head or per-channel scalar), else None."""
    tensor = _initializer(graph, name)
    rank = 4
    if tensor is None or tensor.data_type != FLOAT32 or len(tensor.dims) != rank:
        return None
    return tensor.dims[1] if (tensor.dims[0], tensor.dims[2], tensor.dims[3]) == (1, 1, 1) else None


@dataclass(frozen=True, slots=True)
class _ChannelRead:
    """A per-pair channel map `y[., i, j] = W x[., i, j]`, in either export form.

    Einsum form: `Einsum(Expand(W), x)`. Rewritten form: `Reshape(MatMul(Expand(W), Reshape(x)))`, with a
    `Transpose [0, 2, 1]` of the expanded weight for the transposed equation. `weight` is the `[1, a, b]`
    constant; `transposed` means it applies as `[b, a]` (out, in).
    """

    weight: str
    source: str
    transposed: bool
    node: int  # the Einsum, or the rewrite's MatMul
    expand: int


def _expanded_weight(graph: Graph, tensor: str) -> tuple[str, int, bool] | None:
    """Return (constant, Expand index, transposed) behind a channel map's weight input, or None."""
    transposed = False
    expand = None
    current = tensor
    for _ in range(6):
        if current in graph.initializers:
            break
        node = graph.produced_by(current)
        if node is None:
            return None
        if node.op_type == "Transpose":
            if node.integers.get("perm") != [0, 2, 1]:
                return None
            transposed = not transposed
        elif node.op_type == "Expand":
            if expand is not None:
                return None
            expand = node.index
        elif node.op_type not in _SHAPE_ONLY:
            return None
        current = node.inputs[0]
    else:
        return None
    constant = graph.initializers[current]
    rank = 3
    if expand is None or constant.data_type != FLOAT32 or len(constant.dims) != rank or constant.dims[0] != 1:
        return None
    return current, expand, transposed


def _channel_read(graph: Graph, tensor: str) -> _ChannelRead | None:
    """Return the channel map that produces `tensor`, or None if it is something else."""
    node = graph.produced_by(tensor)
    if node is None:
        return None
    if node.op_type == "Einsum" and len(node.inputs) == 2:  # noqa: PLR2004
        transposed = _CHANNEL_EQUATIONS.get(node.strings.get("equation", ""))
        weight = _expanded_weight(graph, node.inputs[0])
        if transposed is None or weight is None or weight[2]:
            return None
        return _ChannelRead(weight[0], node.inputs[1], transposed, node.index, weight[1])
    if node.op_type == "Reshape":
        product = graph.produced_by(node.inputs[0])
        if product is None or product.op_type != "MatMul":
            return None
        flat = graph.produced_by(product.inputs[1])
        weight = _expanded_weight(graph, product.inputs[0])
        if flat is None or flat.op_type != "Reshape" or weight is None:
            return None
        return _ChannelRead(weight[0], flat.inputs[0], weight[2], product.index, weight[1])
    return None


def _reduce_axes(graph: Graph, node: Node) -> list[int] | None:
    """Return a ReduceMean's axes, from its attribute or its (opset 18) constant input."""
    if "axes" in node.integers:
        return node.integers["axes"]
    if len(node.inputs) > 1:
        tensor = _initializer(graph, node.inputs[1])
        if tensor is not None and tensor.data_type == INT64:
            return list(struct.unpack(f"<{len(tensor.raw_data) // 8}q", tensor.raw_data))
    return None


def _rms(graph: Graph, tensor: str) -> tuple[str, float, int] | None:
    """Return (source, epsilon, node) when `tensor` is `x / sqrt(mean_1 x^2 + eps)` as exported, else None.

    The export form is `Mul(x, Div(1, Sqrt(Add(ReduceMean(Mul(x, x), axes=[1]), eps))))`.
    """
    node = graph.produced_by(tensor)
    if node is None or node.op_type != "Mul" or len(node.inputs) != 2:  # noqa: PLR2004
        return None
    for index in (0, 1):
        source, other = node.inputs[index], node.inputs[1 - index]
        divide = graph.produced_by(other)
        if divide is None or divide.op_type != "Div" or _scalar(graph, divide.inputs[0]) != 1.0:
            continue
        root = graph.produced_by(divide.inputs[1])
        total = graph.produced_by(root.inputs[0]) if root is not None and root.op_type == "Sqrt" else None
        if total is None or total.op_type != "Add":
            continue
        epsilons = [_scalar(graph, name) for name in total.inputs if _scalar(graph, name) is not None]
        means = [graph.produced_by(name) for name in total.inputs if graph.produced_by(name) is not None]
        if len(epsilons) != 1 or len(means) != 1 or means[0].op_type != "ReduceMean":
            continue
        square = graph.produced_by(means[0].inputs[0])
        if square is None or square.op_type != "Mul" or square.inputs != (source, source):
            continue
        if _reduce_axes(graph, means[0]) != [1]:
            continue
        return source, epsilons[0], node.index
    return None


def _split_add(graph: Graph, tensor: str, predicate, where: str, what: str) -> tuple[str, str]:  # noqa: ANN001
    """Return (the other input, the matching input) of the two-input Add producing `tensor`."""
    node = graph.produced_by(tensor)
    _require(node is not None and node.op_type == "Add" and len(node.inputs) == 2, f"{where}: {what} is not an Add")  # noqa: PLR2004
    matches = [name for name in node.inputs if predicate(name)]
    _require(len(matches) == 1, f"{where}: expected one {what} input, found {len(matches)}")
    return next(name for name in node.inputs if name != matches[0]), matches[0]


def _read_egt_block(  # noqa: C901, PLR0915
    graph: Graph, consumers: dict[str, list[Node]], anchor: int
) -> tuple[EgtBlock, frozenset[int], str, str]:
    """Walk one EGT2 block back from its attention Softmax; return it, its excluded nodes and (C1, C2) names."""
    softmax = graph.nodes[anchor]
    where = f"EGT2 block at Softmax {anchor}"
    door = graph.produced_by(softmax.inputs[0])
    opened = [_one_plus(graph, name) for name in door.inputs]
    door_read = _channel_read(graph, next(name for name in opened if name is not None))
    _require(door_read is not None and not door_read.transposed, f"{where}: the door is not a read of the edge state")
    state = door_read.source
    scores = next(name for name, inner in zip(door.inputs, opened, strict=True) if inner is None)
    temperature_sum = graph.produced_by(scores)
    _require(
        temperature_sum is not None and temperature_sum.op_type == "Add" and len(temperature_sum.inputs) == 2,  # noqa: PLR2004
        f"{where}: the door does not multiply a sum of two tempered terms",
    )
    node_term: tuple[str, str] | None = None
    edge_term: tuple[str, _ChannelRead] | None = None
    for name in temperature_sum.inputs:
        product = graph.produced_by(name)
        _require(product is not None and product.op_type == "Mul" and len(product.inputs) == 2,  # noqa: PLR2004
                 f"{where}: a tempered term is not a product")
        temperatures = [item for item in product.inputs if _column(graph, item) is not None]
        _require(len(temperatures) == 1, f"{where}: a tempered term has {len(temperatures)} temperatures")
        operand = next(item for item in product.inputs if item != temperatures[0])
        read = _channel_read(graph, operand)
        if read is not None and read.source == state and not read.transposed:
            _require(edge_term is None, f"{where}: two edge-state terms before the door")
            edge_term = (temperatures[0], read)
        else:
            _require(node_term is None, f"{where}: two static-logit terms before the door")
            node_term = (temperatures[0], operand)
    _require(node_term is not None and edge_term is not None, f"{where}: the tempered terms are incomplete")

    def is_read(item: str) -> bool:
        return _channel_read(graph, item) is not None

    def is_transpose(item: str) -> bool:
        producer = graph.produced_by(item)
        return producer is not None and producer.op_type == "Transpose"

    rest, pair_term = _split_add(graph, node_term[1], is_read, where, "pair-stream term")
    rest, _ = _split_add(graph, rest, is_transpose, where, "query pair term")
    rest, _ = _split_add(graph, rest, is_transpose, where, "key pair term")
    _, attack_term = _split_add(graph, rest, is_read, where, "attack term")
    pair_read, attack_read = _channel_read(graph, pair_term), _channel_read(graph, attack_term)
    _require(not pair_read.transposed and not attack_read.transposed, f"{where}: a static logit table is transposed")

    gated = [node for node in consumers.get(softmax.outputs[0], []) if node.op_type == "Mul"]
    _require(len(gated) == 1, f"{where}: expected one gate product of the Softmax, found {len(gated)}")
    gate = next(name for name in gated[0].inputs if name != softmax.outputs[0])
    gate_node = graph.produced_by(gate)
    cap = gate_node is not None and gate_node.op_type == "Sub"
    doubled = gate
    if cap:
        clip = graph.produced_by(gate_node.inputs[1])
        excess = graph.produced_by(clip.inputs[0]) if clip is not None and clip.op_type == "Relu" else None
        _require(
            excess is not None and excess.op_type == "Sub" and excess.inputs[0] == gate_node.inputs[0]
            and _scalar(graph, excess.inputs[1]) == 1.0,
            f"{where}: the gate cap is not min(g, 1) = g - relu(g - 1)",
        )
        doubled = gate_node.inputs[0]
    twice = graph.produced_by(doubled)
    _require(twice is not None and twice.op_type == "Mul" and len(twice.inputs) == 2,  # noqa: PLR2004
             f"{where}: the gate is not 2 sigmoid(.)")
    twos = [name for name in twice.inputs if _scalar(graph, name) == 2.0]  # noqa: PLR2004
    sigmoid = graph.produced_by(next((name for name in twice.inputs if _scalar(graph, name) is None), ""))
    _require(len(twos) == 1 and sigmoid is not None and sigmoid.op_type == "Sigmoid", f"{where}: the gate is not 2 sigmoid(.)")
    gate_sum = graph.produced_by(sigmoid.inputs[0])
    _require(gate_sum is not None and gate_sum.op_type == "Add" and len(gate_sum.inputs) == 2,  # noqa: PLR2004
             f"{where}: the gate's pre-activation is not a biased read")
    biases = [name for name in gate_sum.inputs if _column(graph, name) is not None]
    _require(len(biases) == 1, f"{where}: the gate has {len(biases)} biases")
    gate_read = _channel_read(graph, next(name for name in gate_sum.inputs if name != biases[0]))
    _require(gate_read is not None and gate_read.source == state and not gate_read.transposed,
             f"{where}: the gate does not read the block's edge state")
    values = [node for node in consumers.get(gated[0].outputs[0], []) if node.op_type == "MatMul"]
    _require(len(values) == 1, f"{where}: the gated weights do not meet the values")

    block = EgtBlock(
        node_temperature=node_term[0],
        edge_temperature=edge_term[0],
        edge_read=edge_term[1].weight,
        door=door_read.weight,
        gate_weight=gate_read.weight,
        gate_bias=biases[0],
        cap=cap,
        state=state,
        edges=attack_read.source,
        pair_stream=pair_read.source,
        softmax_node=anchor,
        logits_node=door.index,
        weights_node=gated[0].index,
    )
    excluded = frozenset({sigmoid.index, twice.index, door_read.expand, edge_term[1].expand, gate_read.expand})
    return block, excluded, attack_read.weight, pair_read.weight


def _read_triplet(  # noqa: C901, PLR0913
    graph: Graph, state: str, residual: Node, output: _ChannelRead, where: str
) -> tuple[EgtTriplet, float]:
    """Recover a triplet branch `e_hat + tri_o (va_in | va_out)` and its rms epsilon."""
    mixed = graph.produced_by(output.source)
    concat = mixed if mixed is not None and mixed.op_type == "Concat" else (
        graph.produced_by(mixed.inputs[0]) if mixed is not None else None
    )
    _require(concat is not None and concat.op_type == "Concat" and len(concat.inputs) == 2,  # noqa: PLR2004
             f"{where}: a residual branch is neither the edge FFN nor a triplet")
    contractions = []
    for name in concat.inputs:
        swap = graph.produced_by(name)
        contraction = graph.produced_by(swap.inputs[0]) if swap is not None and swap.op_type == "Transpose" else None
        _require(contraction is not None and contraction.op_type == "Einsum" and len(contraction.inputs) == 2,  # noqa: PLR2004
                 f"{where}: a triplet direction is not a transposed Einsum")
        contractions.append(contraction)
    equations = tuple(node.strings.get("equation", "") for node in contractions)
    _require(equations in TRIPLET_CONTRACTIONS, f"{where}: unknown triplet contraction {equations}")
    value_splits, gate_splits = set(), {}
    roles: dict[int, tuple[str, int | None]] = {}
    for direction, contraction in enumerate(contractions):
        view = graph.produced_by(contraction.inputs[0])
        split = graph.produced_by(view.inputs[0]) if view is not None else None
        product = graph.produced_by(contraction.inputs[1])
        _require(split is not None and split.op_type == "Split" and product is not None and product.op_type == "Mul",
                 f"{where}: triplet direction {direction} does not contract split values with gated weights")
        value_splits.add(split.index)
        outputs = set()
        for name in product.inputs:
            weight = graph.produced_by(name)
            source = graph.produced_by(weight.inputs[0]) if weight is not None else None
            _require(weight is not None and weight.op_type in ("Softmax", "Sigmoid") and source is not None
                     and source.op_type == "Split", f"{where}: triplet weights are not softmax x sigmoid of a split")
            gate_splits[source.index] = source
            position = source.outputs.index(weight.inputs[0])
            outputs.add(position)
            axis = weight.integers.get("axis", [None])[0] if weight.op_type == "Softmax" else None
            roles[position] = (weight.op_type, axis)
        _require(outputs == {2 * direction, 2 * direction + 1}, f"{where}: triplet direction {direction} reads split outputs {sorted(outputs)}")
    _require(len(value_splits) == 1 and len(gate_splits) == 1 and roles == _TRIPLET_SPLIT_ROLES,
             f"{where}: the triplet's split roles are {roles}")
    gate_split = next(iter(gate_splits.values()))
    gate_sum = graph.produced_by(gate_split.inputs[0])
    _require(gate_sum is not None and gate_sum.op_type == "Add" and len(gate_sum.inputs) == 2,  # noqa: PLR2004
             f"{where}: the triplet gate is not a biased read")
    biases = [name for name in gate_sum.inputs if _column(graph, name) is not None]
    _require(len(biases) == 1, f"{where}: the triplet gate has {len(biases)} biases")
    gate_read = _channel_read(graph, next(name for name in gate_sum.inputs if name != biases[0]))
    value_split = graph.nodes[next(iter(value_splits))]
    value_read = _channel_read(graph, value_split.inputs[0])
    _require(gate_read is not None and value_read is not None and gate_read.transposed and value_read.transposed
             and output.transposed and gate_read.source == value_read.source,
             f"{where}: the triplet projections are not transposed reads of one tensor")
    norm = _rms(graph, value_read.source)
    _require(norm is not None and norm[0] == state, f"{where}: the triplet does not read rms of the state after the readback")
    return EgtTriplet(
        value_weight=value_read.weight,
        gate_weight=gate_read.weight,
        gate_bias=biases[0],
        output_weight=output.weight,
        contraction=TRIPLET_CONTRACTIONS[equations],
        split_node=gate_split.index,
        add_node=residual.index,
    ), norm[1]


def _board_transposed(graph: Graph, tensor: str) -> tuple[str, int] | None:
    """Return (source, node index) if `tensor` is a `[batch, c, 64, 64]` map with its two board axes swapped."""
    node = graph.produced_by(tensor)
    if node is None or node.op_type != "Transpose" or node.integers.get("perm") != [0, 1, 3, 2]:
        return None
    return node.inputs[0], node.index


def _shape_only(consumers: dict[str, list[Node]], node: Node) -> bool:
    """True when every reader of `node` reads only its shape: a value-dead node the export kept for a `Shape`.

    Round 26b: a pre-norm tower's tap norm at a site block (`tap_norms[b](x)`, for the lab's depth-attending aux
    heads) survives the export only because its batch dimension sizes the readback's `Expand`; nothing reads its value.
    """
    readers = [reader for name in node.outputs for reader in consumers.get(name, [])]
    return bool(readers) and all(reader.op_type == "Shape" for reader in readers)


def _read_egt_site(  # noqa: C901, PLR0913
    graph: Graph, consumers: dict[str, list[Node]], add: Node, readback: _ChannelRead, after: int, block_end: int
) -> EgtSite:
    """Recover one update site from the Add that reads block `after`'s logits back into the edge state."""
    where = f"update site after block {after}"
    _require(readback.transposed, f"{where}: the readback is not a transposed channel map")
    current = add.outputs[0]
    triplet = None
    epsilons = set()
    touched = {readback.expand, readback.node, add.index}
    while True:
        branches = []
        for consumer in consumers.get(current, []):
            if consumer.op_type != "Add" or len(consumer.inputs) != 2:  # noqa: PLR2004
                continue
            other = consumer.inputs[1] if consumer.inputs[0] == current else consumer.inputs[0]
            branch = _channel_read(graph, other)
            if branch is not None:
                branches.append((consumer, branch))
        _require(len(branches) == 1, f"{where}: expected one residual branch on {current}, found {len(branches)}")
        residual, branch = branches[0]
        _require(branch.transposed, f"{where}: a branch output is not a transposed channel map")
        touched |= {residual.index, branch.node}
        hidden = graph.produced_by(branch.source)
        if hidden is None or hidden.op_type != "Relu":
            _require(triplet is None, f"{where}: two triplet branches")
            triplet, epsilon = _read_triplet(graph, current, residual, branch, where)
            epsilons.add(epsilon)
            current = residual.outputs[0]
            continue
        biased = graph.produced_by(hidden.inputs[0])
        _require(biased is not None and biased.op_type == "Add" and len(biased.inputs) == 2,  # noqa: PLR2004
                 f"{where}: the edge FFN's hidden layer is not a biased read")
        # The lab's round-32 `rev_edge` (BT6-test): relu((W_in rms(e_hat) + b) + W_rev rms(e_hat)^T). The Add under
        # the Relu then carries no column bias: one side is the reverse channel map -- read against the SAME rms
        # with its two board axes swapped -- and the other side is the biased forward read of every other net.
        reverse = None
        if not any(_column(graph, name) is not None for name in biased.inputs):
            found = []
            for name in biased.inputs:
                read = _channel_read(graph, name)
                swapped = _board_transposed(graph, read.source) if read is not None else None
                if read is not None and read.transposed and swapped is not None:
                    found.append((name, read, swapped))
            _require(len(found) == 1, f"{where}: the edge FFN's hidden layer is neither a biased read nor a biased "
                                      "read plus one reverse-edge read")
            reverse_name, reverse, (reverse_source, swap_node) = found[0]
            touched |= {biased.index, reverse.node, reverse.expand, swap_node}
            biased = graph.produced_by(next(name for name in biased.inputs if name != reverse_name))
            _require(biased is not None and biased.op_type == "Add" and len(biased.inputs) == 2,  # noqa: PLR2004
                     f"{where}: the forward half of the edge FFN's hidden layer is not a biased read")
        biases = [name for name in biased.inputs if _column(graph, name) is not None]
        inward = _channel_read(graph, next((name for name in biased.inputs if name not in biases), ""))
        _require(len(biases) == 1 and inward is not None and inward.transposed, f"{where}: the edge FFN's input map")
        norm = _rms(graph, inward.source)
        _require(norm is not None and norm[0] == current, f"{where}: the edge FFN does not read rms of the state")
        _require(reverse is None or reverse_source == inward.source,
                 f"{where}: the reverse-edge read does not take the transposed rms the forward read takes")
        closing = [
            found for found in (_rms(graph, node.outputs[0]) for node in consumers.get(residual.outputs[0], [])
                                if node.op_type == "Mul")
            if found is not None and found[0] == residual.outputs[0]
        ]
        _require(len(closing) == 1, f"{where}: the site does not close with one rms")
        epsilons |= {norm[1], closing[0][1]}
        last = closing[0][2]
        break
    _require(len(epsilons) == 1, f"{where}: rms epsilons differ: {sorted(epsilons)}")
    first = block_end + 1
    _require(min(touched) >= first and max(touched) < last, f"{where}: its nodes are not between block {after} and e'")
    intruders = [node.index for node in graph.nodes[first : last + 1]
                 if node.op_type in ("Gemm", "LayerNormalization") and not _shape_only(consumers, node)]
    _require(not intruders, f"{where}: block nodes {intruders[:4]} inside the site's range")
    return EgtSite(
        after_block=after,
        readback_weight=readback.weight,
        ffn_in_weight=inward.weight,
        ffn_in_bias=biases[0],
        ffn_out_weight=branch.weight,
        epsilon=epsilons.pop(),
        triplet=triplet,
        first_node=first,
        last_node=last,
        readback_node=readback.node,
        state_node=add.index,
        ffn_add_node=residual.index,
        state=graph.nodes[last].outputs[0],
        ffn_rev_weight=reverse.weight if reverse is not None else None,
    )


def _read_egt_seed(graph: Graph, tensor: str, where: str) -> tuple[str, str, str, float, int, str]:
    """Return (mix, offset table, relative index, epsilon, rms node, E) of `rms(mix.E + table[relidx])`."""
    norm = _rms(graph, tensor)
    _require(norm is not None, f"{where}: not an rms-normed seed")
    total = graph.produced_by(norm[0])
    _require(total is not None and total.op_type == "Add" and len(total.inputs) == 2, f"{where}: not a sum")  # noqa: PLR2004
    reads = [_channel_read(graph, name) for name in total.inputs]
    mixes = [read for read in reads if read is not None]
    _require(len(mixes) == 1 and mixes[0].transposed, f"{where}: expected one transposed channel mix")
    current = next(name for name, read in zip(total.inputs, reads, strict=True) if read is None)
    gather = None
    for _ in range(8):
        node = graph.produced_by(current)
        _require(node is not None, f"{where}: the offsets are not a gathered table")
        if node.op_type == "Gather":
            gather = node
            break
        _require(node.op_type in _SHAPE_ONLY | {"Transpose", "Expand"}, f"{where}: {node.op_type} behind the offsets")
        current = node.inputs[0]
    _require(gather is not None, f"{where}: no Gather behind the offsets")
    table = _initializer(graph, gather.inputs[0])
    rank = 2
    _require(table is not None and len(table.dims) == rank, f"{where}: the gathered offsets are not a table")
    indices = _integer_tables_behind(graph, graph.producer, gather.inputs[1], _SQUARE_COUNT * _SQUARE_COUNT)
    _require(len(indices) == 1, f"{where}: expected one relative index, found {sorted(indices)}")
    return mixes[0].weight, table.name, indices.pop(), norm[1], norm[2], mixes[0].source


def _read_egt(  # noqa: C901
    graph: Graph, ranges: list[tuple[int, int]]
) -> tuple[EgtNetwork, PairTables, list[tuple[int, int]], list[frozenset[int]], list[tuple[str, str]]]:
    """Recover the edge stream; return it, the pair tables, block ranges without the sites, exclusions, (C1, C2)."""
    consumers = _consumers(graph)
    anchors = _attention_softmaxes(graph)
    for node in graph.nodes:
        if node.op_type == "Softmax" and node.index not in anchors:
            if not consumers.get(node.outputs[0]):
                # r25: a terminal Softmax is an output normalisation (the post-hoc WDL softmax of every shipped
                # export), identified by structure: nothing consumes it. It is neither a door nor a triplet weight.
                continue
            split = graph.produced_by(node.inputs[0])
            _require(split is not None and split.op_type == "Split",
                     f"Softmax {node.index} is neither an attention door nor a triplet weight")
    _require(len(anchors) == len(ranges), "the block ranges do not follow the attention Softmaxes")
    read = [_read_egt_block(graph, consumers, anchor) for anchor in anchors]
    blocks = tuple(item[0] for item in read)
    logits = {graph.nodes[block.logits_node].outputs[0]: index for index, block in enumerate(blocks)}

    sites: list[EgtSite] = []
    for node in graph.nodes:
        if node.op_type != "Add" or len(node.inputs) != 2:  # noqa: PLR2004
            continue
        for index in (0, 1):
            producer = graph.produced_by(node.inputs[index])
            if producer is None or producer.op_type not in ("Einsum", "Reshape"):
                continue
            readback = _channel_read(graph, node.inputs[index])
            if readback is not None and readback.source in logits:
                after = logits[readback.source]
                sites.append(_read_egt_site(graph, consumers, node, readback, after, ranges[after][1]))
    sites.sort(key=lambda site: site.after_block)
    after_blocks = [site.after_block for site in sites]
    _require(len(set(after_blocks)) == len(sites), f"two update sites read one block: {after_blocks}")

    adjusted = list(ranges)
    for site in sites:
        index = site.after_block
        _require(index + 1 < len(ranges), f"an update site follows the last block ({index})")
        _require(site.last_node < anchors[index + 1], f"update site after block {index} overlaps the next block")
        adjusted[index + 1] = (site.last_node + 1, ranges[index + 1][1])

    pair_stream, edges = blocks[0].pair_stream, blocks[0].edges
    mix, offsets, relative, pair_epsilon, pair_node, pair_edges = _read_egt_seed(graph, pair_stream, "the pair stream")
    edge_mix, edge_offsets, edge_relative, state_epsilon, state_node, state_edges = _read_egt_seed(
        graph, blocks[0].state, "the edge-stream seed"
    )
    _require(pair_edges == edges == state_edges, "the seeds and the attack terms read different edge tensors")
    _require(relative == edge_relative, "the two seeds gather with different relative indices")
    expected = blocks[0].state
    by_block = {site.after_block: site for site in sites}
    for index, block in enumerate(blocks):
        _require(block.pair_stream == pair_stream and block.edges == edges, f"block {index} reads other seeds")
        _require(block.state == expected, f"block {index} reads edge state {block.state}, expected {expected}")
        if index in by_block:
            expected = by_block[index].state
    caps = {block.cap for block in blocks}
    _require(len(caps) == 1, "the gate cap is on in some blocks only")
    _require(len({site.triplet is not None for site in sites}) <= 1, "a triplet is present at some sites only")
    contractions = {site.triplet.contraction for site in sites if site.triplet is not None}
    _require(len(contractions) <= 1, f"the sites mix triplet contractions {sorted(contractions)}")
    epsilons = {pair_epsilon, state_epsilon, *(site.epsilon for site in sites)}
    _require(len(epsilons) == 1, f"the edge-stream rms epsilons differ: {sorted(epsilons)}")
    network = EgtNetwork(
        blocks=blocks,
        sites=tuple(sites),
        prologue=EgtPrologue(
            edge_mix=edge_mix,
            edge_offset_table=edge_offsets,
            relative_index=relative,
            epsilon=state_epsilon,
            edges_node=graph.produced_by(edges).index,
            pair_node=pair_node,
            state_node=state_node,
        ),
        cap=caps.pop(),
        state_channels=graph.initializers[blocks[0].edge_read].dims[2],
        site_hidden=graph.initializers[sites[0].ffn_in_weight].dims[2] if sites else 0,
    )
    pair = PairTables(channel_mix=mix, offset_table=offsets, relative_index=relative, epsilon=pair_epsilon)
    return network, pair, adjusted, [item[1] for item in read], [(item[2], item[3]) for item in read]


def _assert_egt_agrees(
    graph: Graph,
    blocks: tuple[EncoderBlock, ...],
    egt: EgtNetwork,
    architecture: Architecture,
    agreed: list[tuple[str, str]],
) -> None:
    """Hold the edge stream to one set of shapes, and the two recoveries of C1 and C2 to each other."""
    heads, states, hidden = architecture.heads, egt.state_channels, egt.site_hidden

    def shape(name: str, expected: tuple[int, ...], what: str) -> None:
        actual = tuple(graph.initializers[name].dims)
        _require(actual == expected, f"{what} ({name}) has shape {actual}, expected {expected}")

    for index, (block, edge, (attack, mix)) in enumerate(zip(blocks, egt.blocks, agreed, strict=True)):
        _require(block.edge_coefficients == attack and block.mix_coefficients == mix,
                 f"block {index}: the door walk finds C1/C2 {attack}/{mix}, the table recovery "
                 f"{block.edge_coefficients}/{block.mix_coefficients}")
        for attribute, expected in (
            ("node_temperature", (1, heads, 1, 1)), ("edge_temperature", (1, heads, 1, 1)),
            ("edge_read", (1, heads, states)), ("door", (1, heads, states)),
            ("gate_weight", (1, heads, states)), ("gate_bias", (1, heads, 1, 1)),
        ):
            shape(getattr(edge, attribute), expected, f"block {index}: {attribute}")
    for site in egt.sites:
        where = f"site after block {site.after_block}"
        shape(site.readback_weight, (1, heads, states), f"{where}: readback")
        shape(site.ffn_in_weight, (1, states, hidden), f"{where}: ffn_in")
        shape(site.ffn_in_bias, (1, hidden, 1, 1), f"{where}: ffn_in bias")
        shape(site.ffn_out_weight, (1, hidden, states), f"{where}: ffn_out")
        _require((site.ffn_rev_weight is None) == (egt.sites[0].ffn_rev_weight is None),
                 f"{where}: the sites disagree on the reverse-edge read")
        if site.ffn_rev_weight is not None:
            shape(site.ffn_rev_weight, (1, states, hidden), f"{where}: ffn_rev")
        if site.triplet is not None:
            shape(site.triplet.value_weight, (1, states, 2 * states), f"{where}: triplet value")
            shape(site.triplet.gate_weight, (1, states, states), f"{where}: triplet gate")
            shape(site.triplet.gate_bias, (1, states, 1, 1), f"{where}: triplet gate bias")
            shape(site.triplet.output_weight, (1, 2 * states, states), f"{where}: triplet output")
    shape(egt.prologue.edge_mix, (1, architecture.edge_channels, states), "the edge-stream seed mix")
    _require(graph.initializers[egt.prologue.edge_offset_table].dims[1] == states, "the edge-stream offsets' width")
