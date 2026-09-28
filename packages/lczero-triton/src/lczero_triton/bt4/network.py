"""Protobuf-driven grammar for construction of the BT4 executable graph."""

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from lc0ex import Buffer, ExecutableBuilder, ProgramBuilder
from lc0ex.proto import lc0ex_metadata_pb2, lc0ex_pb2, net_pb2

from lczero_triton.bt4._format import (
    normalize_network,
    validate_network_format,
)
from lczero_triton.bt4.kernels._autotune import active_architecture
from lczero_triton.bt4.kernels._cache import KernelCache
from lczero_triton.bt4.kernels.add_vectors import (
    AddVectorsSpecialization,
    add_vectors,
)
from lczero_triton.bt4.kernels.batched_matmul import (
    BatchedMatmulSpecialization,
    batched_matmul,
)
from lczero_triton.bt4 import _q1
from lczero_triton.bt4.kernels.cutlass_gemm_i8 import (
    CutlassGemmI8Specialization,
    cutlass_gemm_i8,
)
from lczero_triton.bt4.kernels.cutlass_matmul import (
    CutlassMatmulSpecialization,
    cutlass_matmul,
)
from lczero_triton.bt4.kernels.expand_planes import (
    ExpandPlanesSpecialization,
    expand_planes,
)
from lczero_triton.bt4.kernels.fused_attention import (
    FusedAttentionSpecialization,
    fused_attention,
)
from lczero_triton.bt4.kernels.input_gating import (
    InputGatingSpecialization,
    input_gating,
)
from lczero_triton.bt4.kernels.layer_norm import (
    LayerNormSpecialization,
    layer_norm,
)
from lczero_triton.bt4.kernels.mapping_table import compile_symbol
from lczero_triton.bt4.kernels.matmul import MatmulSpecialization, matmul
from lczero_triton.bt4.kernels.nchw_to_nhwc import (
    NchwToNhwcSpecialization,
    nchw_to_nhwc,
)
from lczero_triton.bt4.kernels.policy_map import (
    PolicyMapSpecialization,
    policy_map,
)
from lczero_triton.bt4.kernels.preprocess_attention_body import (
    PreprocessAttentionBodySpecialization,
    preprocess_attention_body,
)
from lczero_triton.bt4.kernels.promotion_logits import (
    PromotionLogitsSpecialization,
    promotion_logits,
)
from lczero_triton.bt4.kernels.quantise_operand import (
    QuantiseOperandSpecialization,
    quantise_operand,
)

_F16_SIZE_BYTES = 2
_INPUT_CHANNELS = 112
_POSITION_CHANNELS = 12

# Opt-in per GEMM: build the encoder FFN and out-projection from CUTLASS rather
# than Triton. Each is a separate arm because the arms do not add (R7), and each
# must gate bit-identical against the Triton kernel it replaces before it counts.
_CUTLASS_FFN1 = os.environ.get("LC0EX_CUTLASS_FFN1") == "1"
_CUTLASS_FFN2 = os.environ.get("LC0EX_CUTLASS_FFN2") == "1"
_CUTLASS_OUTPROJ = os.environ.get("LC0EX_CUTLASS_OUTPROJ") == "1"
_CUTLASS_QKV = os.environ.get("LC0EX_CUTLASS_QKV") == "1"

# Chunked attention segment (round 14 item A.2).  `LC0EX_SEGMENT_CHUNK` is the
# chunk size in SAMPLES; 0 or unset leaves the segment whole and this file emits
# exactly what it emitted before.  `LC0EX_CHUNK_MODE` selects how the chunks may
# overlap: `chain` (one reused buffer set, sequential), `free` (a set per chunk,
# the graph overlaps them) or `rotate2` (two sets, i overlaps i+1 but not i+2).
_SEGMENT_CHUNK = os.environ.get("LC0EX_SEGMENT_CHUNK", "")
_CHUNK_MODE = os.environ.get("LC0EX_CHUNK_MODE", "chain")
# The layer norm is a streaming kernel with no residency to win, and splitting it
# measured +45 % on its own time, so by default it stays whole and only the
# segment ahead of it is chunked.  `chunked` restores the split for an A/B.
_CHUNK_LN_WHOLE = os.environ.get("LC0EX_CHUNK_LN", "whole") != "chunked"
_MIN_SEGMENT_CHUNK = 16

_SQUARE_COUNT = 64
_LOGGER = logging.getLogger(__name__)


def _segment_chunk(batch_size: int) -> int:
    """Return the segment chunk size in samples, or 0 when it is not chunked.

    A chunk must divide the batch and be smaller than it; anything else silently
    means "no chunking" rather than an artifact that is subtly not the rung it
    claims to be.
    """
    if not _SEGMENT_CHUNK:
        return 0
    target = int(_SEGMENT_CHUNK)
    if target <= 0 or target >= batch_size:
        return 0
    # Snap DOWN to a divisor of the rung rather than giving up on it: at the
    # two-slot target of 32 the deploy ladder's 40, 48 and 56 would otherwise
    # stay whole, and 48 is where the two-slot throughput peaks.  The floor is
    # 16 -- c = 16 measured -4.2 % at one slot, so smaller chunks never pay.
    for chunk in range(target, _MIN_SEGMENT_CHUNK - 1, -1):
        if batch_size % chunk == 0:
            return chunk
    return 0


def _chunk_set_count(mode: str, chunk_count: int) -> int:
    """Return how many independent per-chunk buffer sets a mode allocates."""
    if mode == "chain":
        return 1
    if mode == "rotate2":
        return min(2, chunk_count)
    return chunk_count


@dataclass(slots=True)
class _BuildContext:
    """Construction services shared by grammar productions in one build."""

    builder: ProgramBuilder
    kernels: KernelCache
    batch_size: int
    architecture: int
    fingerprint: net_pb2.Net
    fingerprint_layers: dict[str, net_pb2.Weights.Layer]
    shared_buffers: dict[str, Buffer]
    # B3 (Q1 on BT4): the int8 sites of every encoder, by the analyser's vector file; empty sets = the FP16 artifact.
    q1_sites: dict[int, frozenset[str]]


def build(
    builder: ExecutableBuilder,
    network: net_pb2.Net,
    *,
    batch_sizes: Sequence[int],
) -> None:
    """Traverse the active network and append one or more batch programs."""
    _LOGGER.info("normalizing and validating BT4 network")
    normalize_network(network)
    validate_network_format(network)
    network_format = network.format.network_format
    default_activation = _default_activation(network_format.default_activation)
    ffn_activation = _resolve_activation(
        network_format.ffn_activation,
        default_activation,
    )
    smolgen_activation = _resolve_activation(
        network_format.smolgen_activation,
        default_activation,
    )
    fingerprint, fingerprint_layers = _fingerprint_network(
        network,
        ffn_activation=ffn_activation,
        smolgen_activation=smolgen_activation,
    )
    kernels = KernelCache(builder)
    shared_buffers: dict[str, Buffer] = {}
    q1_sites = _q1_plan(network.weights)
    _LOGGER.info(
        "building BT4 graph for batch sizes %s with %d encoder layers",
        batch_sizes,
        len(network.weights.encoder),
    )
    for size in batch_sizes:
        program_name = "main" if len(batch_sizes) == 1 else f"batch-{size}"
        _LOGGER.info("building program %s for batch size %d", program_name, size)
        program = builder.program(
            name=program_name,
            metadata=lc0ex_metadata_pb2.ProgramMetadata(
                batch_size=size,
            ).SerializeToString(deterministic=True),
        )
        context = _BuildContext(
            builder=program,
            kernels=kernels,
            batch_size=size,
            architecture=active_architecture(),
            fingerprint=fingerprint,
            fingerprint_layers=fingerprint_layers,
            shared_buffers=shared_buffers,
            q1_sites=q1_sites,
        )
        _network(context, network.weights)
        _LOGGER.info("finished program %s", program_name)
    builder.set_metadata(fingerprint.SerializeToString(deterministic=True))
    _LOGGER.info("finished BT4 graph construction")


def _network(context: _BuildContext, weights: net_pb2.Weights) -> None:
    """Build inputs, body embedding, encoders, and selected output heads."""
    _LOGGER.info("batch size %d: building input embedding", context.batch_size)
    inputs = _inputs(context)
    body, body_width, body_codes = _embedding(context, inputs, weights)
    _LOGGER.info(
        "batch size %d: building encoder tower (%d layers)",
        context.batch_size,
        len(weights.encoder),
    )
    body = _encoder_tower(context, body, body_width, weights, body_codes)
    _LOGGER.info("batch size %d: building output heads", context.batch_size)
    _policy_head(context, body, body_width, weights)
    _value_head(context, body, body_width, weights.value_heads.winner)
    _moves_left_head(context, body, body_width, weights)


def _fingerprint_network(
    network: net_pb2.Net,
    *,
    ffn_activation: net_pb2.NetworkFormat.ActivationFunction,
    smolgen_activation: net_pb2.NetworkFormat.ActivationFunction,
) -> tuple[net_pb2.Net, dict[str, net_pb2.Weights.Layer]]:
    """Create a sparse network and paths for layers consumed by the graph."""
    fingerprint = net_pb2.Net()
    source_format = network.format.network_format
    target_format = fingerprint.format.network_format
    for field in (
        "input",
        "output",
        "network",
        "policy",
        "value",
        "moves_left",
        "input_embedding",
    ):
        setattr(target_format, field, getattr(source_format, field))
    target_format.default_activation = source_format.default_activation
    target_format.ffn_activation = ffn_activation
    target_format.smolgen_activation = smolgen_activation

    source_weights = network.weights
    target_weights = fingerprint.weights
    target_weights.headcount = source_weights.headcount
    layers: dict[str, net_pb2.Weights.Layer] = {}

    def add(path: str, layer: net_pb2.Weights.Layer) -> None:
        layers[path] = layer

    for path, layer in (
        ("weights.ip_emb_preproc_w", target_weights.ip_emb_preproc_w),
        ("weights.ip_emb_preproc_b", target_weights.ip_emb_preproc_b),
        ("weights.ip_emb_w", target_weights.ip_emb_w),
        ("weights.ip_emb_b", target_weights.ip_emb_b),
        ("weights.ip_emb_ln_gammas", target_weights.ip_emb_ln_gammas),
        ("weights.ip_emb_ln_betas", target_weights.ip_emb_ln_betas),
        ("weights.ip_mult_gate", target_weights.ip_mult_gate),
        ("weights.ip_add_gate", target_weights.ip_add_gate),
        ("weights.ip_emb_ffn.dense1_w", target_weights.ip_emb_ffn.dense1_w),
        ("weights.ip_emb_ffn.dense1_b", target_weights.ip_emb_ffn.dense1_b),
        ("weights.ip_emb_ffn.dense2_w", target_weights.ip_emb_ffn.dense2_w),
        ("weights.ip_emb_ffn.dense2_b", target_weights.ip_emb_ffn.dense2_b),
        (
            "weights.ip_emb_ffn_ln_gammas",
            target_weights.ip_emb_ffn_ln_gammas,
        ),
        ("weights.ip_emb_ffn_ln_betas", target_weights.ip_emb_ffn_ln_betas),
        ("weights.smolgen_w", target_weights.smolgen_w),
        ("weights.ip_mov_w", target_weights.ip_mov_w),
        ("weights.ip_mov_b", target_weights.ip_mov_b),
        ("weights.ip1_mov_w", target_weights.ip1_mov_w),
        ("weights.ip1_mov_b", target_weights.ip1_mov_b),
        ("weights.ip2_mov_w", target_weights.ip2_mov_w),
        ("weights.ip2_mov_b", target_weights.ip2_mov_b),
    ):
        add(path, layer)

    for index in range(len(source_weights.encoder)):
        target_encoder = target_weights.encoder.add()
        prefix = f"weights.encoder{index}"
        for path, layer in (
            (
                f"{prefix}.mha.smolgen.compress",
                target_encoder.mha.smolgen.compress,
            ),
            (
                f"{prefix}.mha.smolgen.dense1_w",
                target_encoder.mha.smolgen.dense1_w,
            ),
            (
                f"{prefix}.mha.smolgen.dense1_b",
                target_encoder.mha.smolgen.dense1_b,
            ),
            (
                f"{prefix}.mha.smolgen.ln1_gammas",
                target_encoder.mha.smolgen.ln1_gammas,
            ),
            (
                f"{prefix}.mha.smolgen.ln1_betas",
                target_encoder.mha.smolgen.ln1_betas,
            ),
            (
                f"{prefix}.mha.smolgen.dense2_w",
                target_encoder.mha.smolgen.dense2_w,
            ),
            (
                f"{prefix}.mha.smolgen.dense2_b",
                target_encoder.mha.smolgen.dense2_b,
            ),
            (
                f"{prefix}.mha.smolgen.ln2_gammas",
                target_encoder.mha.smolgen.ln2_gammas,
            ),
            (
                f"{prefix}.mha.smolgen.ln2_betas",
                target_encoder.mha.smolgen.ln2_betas,
            ),
            (f"{prefix}.mha.q_w", target_encoder.mha.q_w),
            (f"{prefix}.mha.q_b", target_encoder.mha.q_b),
            (f"{prefix}.mha.k_w", target_encoder.mha.k_w),
            (f"{prefix}.mha.k_b", target_encoder.mha.k_b),
            (f"{prefix}.mha.v_w", target_encoder.mha.v_w),
            (f"{prefix}.mha.v_b", target_encoder.mha.v_b),
            (f"{prefix}.mha.dense_w", target_encoder.mha.dense_w),
            (f"{prefix}.mha.dense_b", target_encoder.mha.dense_b),
            (f"{prefix}.ln1_gammas", target_encoder.ln1_gammas),
            (f"{prefix}.ln1_betas", target_encoder.ln1_betas),
            (f"{prefix}.ffn.dense1_w", target_encoder.ffn.dense1_w),
            (f"{prefix}.ffn.dense1_b", target_encoder.ffn.dense1_b),
            (f"{prefix}.ffn.dense2_w", target_encoder.ffn.dense2_w),
            (f"{prefix}.ffn.dense2_b", target_encoder.ffn.dense2_b),
            (f"{prefix}.ln2_gammas", target_encoder.ln2_gammas),
            (f"{prefix}.ln2_betas", target_encoder.ln2_betas),
        ):
            add(path, layer)
    target_policy = target_weights.policy_heads.vanilla
    for path, layer in (
        (
            "weights.policy_heads.vanilla.ip_pol_w",
            target_policy.ip_pol_w,
        ),
        (
            "weights.policy_heads.vanilla.ip_pol_b",
            target_policy.ip_pol_b,
        ),
        (
            "weights.policy_heads.vanilla.ip2_pol_w",
            target_policy.ip2_pol_w,
        ),
        (
            "weights.policy_heads.vanilla.ip2_pol_b",
            target_policy.ip2_pol_b,
        ),
        (
            "weights.policy_heads.vanilla.ip3_pol_w",
            target_policy.ip3_pol_w,
        ),
        (
            "weights.policy_heads.vanilla.ip3_pol_b",
            target_policy.ip3_pol_b,
        ),
        (
            "weights.policy_heads.vanilla.ip4_pol_w",
            target_policy.ip4_pol_w,
        ),
    ):
        add(path, layer)

    target_winner = target_weights.value_heads.winner
    for path, layer in (
        ("weights.value_heads.winner.ip_val_w", target_winner.ip_val_w),
        ("weights.value_heads.winner.ip_val_b", target_winner.ip_val_b),
        ("weights.value_heads.winner.ip1_val_w", target_winner.ip1_val_w),
        ("weights.value_heads.winner.ip1_val_b", target_winner.ip1_val_b),
        ("weights.value_heads.winner.ip2_val_w", target_winner.ip2_val_w),
        ("weights.value_heads.winner.ip2_val_b", target_winner.ip2_val_b),
    ):
        add(path, layer)
    return fingerprint, layers


def _inputs(context: _BuildContext) -> tuple[Buffer, Buffer]:
    """Declare the packed execution inputs consumed by plane expansion."""
    masks = context.builder.buffer(
        name="/input/plane_masks",
        shape=(context.batch_size, _INPUT_CHANNELS),
        dtype=lc0ex_pb2.Buffer.DATA_TYPE_U64,
    )
    values = context.builder.buffer(
        name="/input/plane_values",
        shape=(context.batch_size, _INPUT_CHANNELS),
        dtype=lc0ex_pb2.Buffer.DATA_TYPE_F32,
    )
    return masks, values


def _embedding(
    context: _BuildContext,
    inputs: tuple[Buffer, Buffer],
    weights: net_pb2.Weights,
) -> tuple[Buffer, int, Buffer | None]:
    """Build dense positional preprocessing, gated embedding, and embedding FFN.

    B3: when encoder 0's `attn_in` is int8, the last norm also writes its int8 copy (the third return value).
    """
    position_input_width = _SQUARE_COUNT * _POSITION_CHANNELS
    position_weights, position_output_width = _matrix_f16(
        context,
        weights.ip_emb_preproc_w,
        input_width=position_input_width,
        name="/attn_body/embedding/preprocess/matmul/w",
        path="weights.ip_emb_preproc_w",
    )
    position_bias = _vector_f16(
        context,
        weights.ip_emb_preproc_b,
        name="/attn_body/embedding/preprocess/add/w",
        path="weights.ip_emb_preproc_b",
    )
    encoding_width = position_output_width // _SQUARE_COUNT

    embedding_input_width = _INPUT_CHANNELS + encoding_width
    embedding_weights, body_width = _matrix_f16(
        context,
        weights.ip_emb_w,
        input_width=embedding_input_width,
        name="/attn_body/matmul/w",
        path="weights.ip_emb_w",
    )
    embedding_bias = _vector_f16(
        context,
        weights.ip_emb_b,
        name="/attn_body/add/w",
        path="weights.ip_emb_b",
    )
    embedding_gammas = _vector_f16(
        context,
        weights.ip_emb_ln_gammas,
        name="/attn_body/ln/w/scale",
        path="weights.ip_emb_ln_gammas",
    )
    embedding_betas = _vector_f16(
        context,
        weights.ip_emb_ln_betas,
        name="/attn_body/ln/w/bias",
        path="weights.ip_emb_ln_betas",
    )

    multiplicative_gate, _ = _matrix_f16(
        context,
        weights.ip_mult_gate,
        input_width=_SQUARE_COUNT,
        name="/ip_mul_gate/w",
        path="weights.ip_mult_gate",
    )
    additive_gate, _ = _matrix_f16(
        context,
        weights.ip_add_gate,
        input_width=_SQUARE_COUNT,
        name="/ip_add_gate/w",
        path="weights.ip_add_gate",
    )
    dense1_weights, hidden_width = _matrix_f16(
        context,
        weights.ip_emb_ffn.dense1_w,
        input_width=body_width,
        name="/attn_body/ffn/dense1/w/w",
        path="weights.ip_emb_ffn.dense1_w",
    )
    dense1_bias = _vector_f16(
        context,
        weights.ip_emb_ffn.dense1_b,
        name="/attn_body/ffn/dense1/b/w",
        path="weights.ip_emb_ffn.dense1_b",
    )
    dense2_weights, _ = _matrix_f16(
        context,
        weights.ip_emb_ffn.dense2_w,
        input_width=hidden_width,
        name="/attn_body/ffn/dense2/w/w",
        path="weights.ip_emb_ffn.dense2_w",
    )
    dense2_bias = _vector_f16(
        context,
        weights.ip_emb_ffn.dense2_b,
        name="/attn_body/ffn/dense2/b/w",
        path="weights.ip_emb_ffn.dense2_b",
    )
    alpha = context.builder.persistent_buffer(
        name="/attn_body/ffn/alpha/w",
        shape=(1,),
        dtype=lc0ex_pb2.Buffer.DATA_TYPE_F16,
        alignment_bytes=256,
    )
    ffn_gammas = _vector_f16(
        context,
        weights.ip_emb_ffn_ln_gammas,
        name="/attn_body/ln2/w/scale",
        path="weights.ip_emb_ffn_ln_gammas",
    )
    ffn_betas = _vector_f16(
        context,
        weights.ip_emb_ffn_ln_betas,
        name="/attn_body/ln2/w/bias",
        path="weights.ip_emb_ffn_ln_betas",
    )

    masks, values = inputs
    plane_count = context.batch_size * _INPUT_CHANNELS
    token_rows = context.batch_size * _SQUARE_COUNT
    planes = _temporary_f16(
        context,
        element_count=plane_count * _SQUARE_COUNT,
    )
    expand_planes(
        context.builder,
        context.kernels,
        planes,
        masks,
        values,
        ExpandPlanesSpecialization(plane_count, context.architecture),
    )

    position_input = _temporary_f16(
        context,
        element_count=context.batch_size * position_input_width,
    )
    nchw_to_nhwc(
        context.builder,
        context.kernels,
        position_input,
        planes,
        NchwToNhwcSpecialization(
            context.batch_size,
            _INPUT_CHANNELS,
            _POSITION_CHANNELS,
            8,
            8,
            context.architecture,
        ),
    )

    position = _temporary_f16(
        context,
        element_count=context.batch_size * position_output_width,
    )
    matmul(
        context.builder,
        context.kernels,
        position,
        position_input,
        position_weights,
        MatmulSpecialization(
            context.batch_size,
            position_output_width,
            position_input_width,
            context.architecture,
        ),
    )
    add_vectors(
        context.builder,
        context.kernels,
        position,
        position,
        position_bias,
        AddVectorsSpecialization(
            context.batch_size * position_output_width,
            position_output_width,
            "none",
            context.architecture,
        ),
    )

    embedding_input = _temporary_f16(
        context,
        element_count=token_rows * embedding_input_width,
    )
    preprocess_attention_body(
        context.builder,
        context.kernels,
        embedding_input,
        planes,
        position,
        PreprocessAttentionBodySpecialization(
            context.batch_size,
            _INPUT_CHANNELS,
            encoding_width,
            context.architecture,
        ),
    )

    projected = _temporary_f16(context, element_count=token_rows * body_width)
    matmul(
        context.builder,
        context.kernels,
        projected,
        embedding_input,
        embedding_weights,
        MatmulSpecialization(
            token_rows,
            body_width,
            embedding_input_width,
            context.architecture,
            has_bias=True,
        ),
        bias=embedding_bias,
    )
    skip = _temporary_f16(context, element_count=token_rows * body_width)
    layer_norm(
        context.builder,
        context.kernels,
        skip,
        projected,
        None,
        embedding_gammas,
        embedding_betas,
        LayerNormSpecialization(
            row_count=token_rows,
            width=body_width,
            activation="mish",
            has_skip=False,
            has_bias=False,
            architecture=context.architecture,
        ),
    )
    input_gating(
        context.builder,
        context.kernels,
        skip,
        skip,
        multiplicative_gate,
        additive_gate,
        InputGatingSpecialization(
            context.batch_size,
            _SQUARE_COUNT,
            body_width,
            context.architecture,
        ),
    )

    hidden = _temporary_f16(context, element_count=token_rows * hidden_width)
    matmul(
        context.builder,
        context.kernels,
        hidden,
        skip,
        dense1_weights,
        MatmulSpecialization(
            token_rows,
            hidden_width,
            body_width,
            context.architecture,
            has_bias=True,
            activation="mish",
        ),
        bias=dense1_bias,
    )

    branch = _temporary_f16(context, element_count=token_rows * body_width)
    matmul(
        context.builder,
        context.kernels,
        branch,
        hidden,
        dense2_weights,
        MatmulSpecialization(
            token_rows,
            body_width,
            hidden_width,
            context.architecture,
            has_bias=True,
            has_skip=True,
        ),
        bias=dense2_bias,
        skip=skip,
        alpha=alpha,
    )
    body = _temporary_f16(context, element_count=token_rows * body_width)
    codes = _q1_norm(
        context,
        body,
        branch,
        ffn_gammas,
        ffn_betas,
        token_rows,
        body_width,
        "/encoder0" if "attn_in" in context.q1_sites.get(0, frozenset()) else None,
        "attn_in",
    )
    return body, body_width, codes


def _encoder_tower(
    context: _BuildContext,
    body: Buffer,
    body_width: int,
    weights: net_pb2.Weights,
    body_codes: Buffer | None = None,
) -> Buffer:
    """Visit each protobuf encoder in evaluation order without fixed depth.

    B3: `body_codes` is the int8 copy of `body` that the next encoder's `attn_in` reads, when that site is int8.
    """
    for index, encoder in enumerate(weights.encoder):
        _LOGGER.info(
            "batch size %d: building encoder %d/%d",
            context.batch_size,
            index + 1,
            len(weights.encoder),
        )
        body, body_codes = _encoder(
            context,
            body,
            body_width,
            encoder,
            prefix=f"/encoder{index}",
            head_count=weights.headcount,
            shared_smolgen=weights.smolgen_w if _has_smolgen(weights) else None,
            index=index,
            body_codes=body_codes,
        )
    return body


def _has_smolgen(weights: net_pb2.Weights) -> bool:
    """Report whether this network carries smolgen at all.

    BT5-rpe has none: `mha.smolgen` is absent and the shared `smolgen_w` is
    empty, and the three RPE tables take its place. The shared projection is
    what decides -- a per-layer `smolgen` block without it could not be
    projected to 64x64 logits.
    """
    return weights.HasField("smolgen_w") and bool(weights.smolgen_w.params)


def _encoder(  # noqa: PLR0913
    context: _BuildContext,
    body: Buffer,
    body_width: int,
    encoder: net_pb2.Weights.EncoderLayer,
    *,
    prefix: str,
    head_count: int,
    shared_smolgen: net_pb2.Weights.Layer | None,
    index: int = 0,
    body_codes: Buffer | None = None,
) -> tuple[Buffer, Buffer | None]:
    """Build one attention and FFN encoder residual block, smolgen optional.

    Returns the block's output and, when the next encoder's `attn_in` is int8, its ln2's int8 copy (B3).
    """
    if shared_smolgen is None:
        smolgen, generated_width = None, 0
    else:
        smolgen, generated_width = _smolgen(
            context,
            body,
            body_width,
            encoder,
            prefix=prefix,
            head_count=head_count,
        )
    sites = context.q1_sites.get(index, frozenset())
    if "attn_in" in sites and body_codes is None:
        message = f"{prefix}: attn_in is int8 but its producer (the previous norm) wrote no int8 copy"
        raise ValueError(message)
    attended, attended_codes = _attention(
        context,
        body,
        smolgen,
        body_width,
        encoder,
        prefix=prefix,
        head_count=head_count,
        shared_smolgen=shared_smolgen,
        generated_width=generated_width,
        sites=sites,
        body_codes=body_codes,
    )
    following = f"/encoder{index + 1}" if "attn_in" in context.q1_sites.get(index + 1, frozenset()) else None
    return _ffn(context, attended, body_width, encoder, prefix=prefix, sites=sites, body_codes=attended_codes,
                next_attn_in=following)


def _smolgen(  # noqa: PLR0913
    context: _BuildContext,
    body: Buffer,
    body_width: int,
    encoder: net_pb2.Weights.EncoderLayer,
    *,
    prefix: str,
    head_count: int,
) -> tuple[Buffer, int]:
    """Build local Smolgen compression and its two normalized dense layers."""
    smolgen = encoder.mha.smolgen
    path = f"weights.{prefix[1:]}.mha.smolgen"
    compress_weights, compression_width = _matrix_f16(
        context,
        smolgen.compress,
        input_width=body_width,
        name=f"{prefix}/smolgen/compress/w",
        path=f"{path}.compress",
    )
    dense1_weights, hidden_width = _matrix_f16(
        context,
        smolgen.dense1_w,
        input_width=_SQUARE_COUNT * compression_width,
        name=f"{prefix}/smolgen/dense1/w/w",
        path=f"{path}.dense1_w",
    )
    dense1_bias = _vector_f16(
        context,
        smolgen.dense1_b,
        name=f"{prefix}/smolgen/dense1/b/w",
        path=f"{path}.dense1_b",
    )
    ln1_gammas = _vector_f16(
        context,
        smolgen.ln1_gammas,
        name=f"{prefix}/smolgen/ln1/w/scale",
        path=f"{path}.ln1_gammas",
    )
    ln1_betas = _vector_f16(
        context,
        smolgen.ln1_betas,
        name=f"{prefix}/smolgen/ln1/w/bias",
        path=f"{path}.ln1_betas",
    )
    dense2_weights, generated_total_width = _matrix_f16(
        context,
        smolgen.dense2_w,
        input_width=hidden_width,
        name=f"{prefix}/smolgen/dense2/w/w",
        path=f"{path}.dense2_w",
    )
    generated_width = generated_total_width // head_count
    dense2_bias = _vector_f16(
        context,
        smolgen.dense2_b,
        name=f"{prefix}/smolgen/dense2/b/w",
        path=f"{path}.dense2_b",
    )
    ln2_gammas = _vector_f16(
        context,
        smolgen.ln2_gammas,
        name=f"{prefix}/smolgen/ln2/w/scale",
        path=f"{path}.ln2_gammas",
    )
    ln2_betas = _vector_f16(
        context,
        smolgen.ln2_betas,
        name=f"{prefix}/smolgen/ln2/w/bias",
        path=f"{path}.ln2_betas",
    )

    token_rows = context.batch_size * _SQUARE_COUNT
    compressed = _temporary_f16(context, element_count=token_rows * compression_width)
    matmul(
        context.builder,
        context.kernels,
        compressed,
        body,
        compress_weights,
        MatmulSpecialization(
            token_rows, compression_width, body_width, context.architecture
        ),
    )
    hidden = _temporary_f16(context, element_count=context.batch_size * hidden_width)
    matmul(
        context.builder,
        context.kernels,
        hidden,
        compressed,
        dense1_weights,
        MatmulSpecialization(
            context.batch_size,
            hidden_width,
            _SQUARE_COUNT * compression_width,
            context.architecture,
            has_bias=True,
        ),
        bias=dense1_bias,
    )
    layer_norm(
        context.builder,
        context.kernels,
        hidden,
        hidden,
        None,
        ln1_gammas,
        ln1_betas,
        LayerNormSpecialization(
            row_count=context.batch_size,
            width=hidden_width,
            activation="swish",
            has_skip=False,
            has_bias=False,
            architecture=context.architecture,
        ),
    )
    generated = _temporary_f16(
        context, element_count=context.batch_size * generated_total_width
    )
    matmul(
        context.builder,
        context.kernels,
        generated,
        hidden,
        dense2_weights,
        MatmulSpecialization(
            context.batch_size,
            generated_total_width,
            hidden_width,
            context.architecture,
            has_bias=True,
        ),
        bias=dense2_bias,
    )
    layer_norm(
        context.builder,
        context.kernels,
        generated,
        generated,
        None,
        ln2_gammas,
        ln2_betas,
        LayerNormSpecialization(
            row_count=context.batch_size,
            width=generated_total_width,
            activation="swish",
            has_skip=False,
            has_bias=False,
            architecture=context.architecture,
        ),
    )
    return generated, generated_width


def _attention(  # noqa: PLR0913
    context: _BuildContext,
    body: Buffer,
    smolgen: Buffer | None,
    body_width: int,
    encoder: net_pb2.Weights.EncoderLayer,
    *,
    prefix: str,
    head_count: int,
    shared_smolgen: net_pb2.Weights.Layer | None,
    generated_width: int,
    sites: frozenset[str] = frozenset(),
    body_codes: Buffer | None = None,
) -> tuple[Buffer, Buffer | None]:
    """Build the encoder Q/K/V attention path, with smolgen if the net has it.

    B3: `attn_in` int8 = the QKV projection on `cutlass_gemm_i8` from `body_codes`; `attn_out` int8 = a
    `quantise_operand` pass after the attention kernel, then the out-projection with alpha and the skip in its
    epilogue; `ffn_in` int8 = ln1 also writes its int8 copy (the second return value).
    """
    mha = encoder.mha
    path = f"weights.{prefix[1:]}"
    has_smolgen = shared_smolgen is not None and smolgen is not None
    if has_smolgen:
        shared_weights, _ = _matrix_f16(
            context,
            shared_smolgen,
            input_width=generated_width,
            name="/const/smolgen_w",
            path="weights.smolgen_w",
        )
    expected_smolgen_width = _SQUARE_COUNT * _SQUARE_COUNT

    element_count = len(mha.q_w.params) // _F16_SIZE_BYTES
    model_width = element_count // body_width
    head_depth = model_width // head_count

    weights_key = f"{prefix}/mha/qkv_weights"
    bias_key = f"{prefix}/mha/qkv_bias"
    if weights_key in context.shared_buffers:
        qkv_weights = context.shared_buffers[weights_key]
        qkv_bias = context.shared_buffers[bias_key]
    else:
        qkv_weights = context.builder.persistent_tensor(
            shape=(body_width, 3, model_width),
            dtype=lc0ex_pb2.Buffer.DATA_TYPE_F16,
            alignment_bytes=256,
        )
        qkv_weights[:, 0, :].external(f"{prefix}/mha/Q/w/w")
        qkv_weights[:, 1, :].external(f"{prefix}/mha/K/w/w")
        qkv_weights[:, 2, :].external(f"{prefix}/mha/V/w/w")
        context.shared_buffers[weights_key] = qkv_weights

        qkv_bias = context.builder.persistent_tensor(
            shape=(3, model_width),
            dtype=lc0ex_pb2.Buffer.DATA_TYPE_F16,
            alignment_bytes=256,
        )
        qkv_bias[0, :].external(f"{prefix}/mha/Q/b/w")
        qkv_bias[1, :].external(f"{prefix}/mha/K/b/w")
        qkv_bias[2, :].external(f"{prefix}/mha/V/b/w")
        context.shared_buffers[bias_key] = qkv_bias

    _fingerprint_layer(context, f"{path}.mha.q_w")
    _fingerprint_layer(context, f"{path}.mha.k_w")
    _fingerprint_layer(context, f"{path}.mha.v_w")
    _fingerprint_layer(context, f"{path}.mha.q_b")
    _fingerprint_layer(context, f"{path}.mha.k_b")
    _fingerprint_layer(context, f"{path}.mha.v_b")

    output_weights, _ = _matrix_f16(
        context,
        mha.dense_w,
        input_width=model_width,
        name=f"{prefix}/mha/out/dense/w/w",
        path=f"{path}.mha.dense_w",
    )
    output_bias = _vector_f16(
        context,
        mha.dense_b,
        name=f"{prefix}/mha/out/dense/b/w",
        path=f"{path}.mha.dense_b",
    )
    gammas = _vector_f16(
        context,
        encoder.ln1_gammas,
        name=f"{prefix}/ln1/w/scale",
        path=f"{path}.ln1_gammas",
    )
    betas = _vector_f16(
        context,
        encoder.ln1_betas,
        name=f"{prefix}/ln1/w/bias",
        path=f"{path}.ln1_betas",
    )
    scale = context.builder.persistent_buffer(
        name=f"{prefix}/mha/QK/scale/w",
        shape=(1,),
        dtype=lc0ex_pb2.Buffer.DATA_TYPE_F16,
        alignment_bytes=256,
    )
    alpha = context.builder.persistent_buffer(
        name=f"{prefix}/alpha*input/w",
        shape=(1,),
        dtype=lc0ex_pb2.Buffer.DATA_TYPE_F16,
        alignment_bytes=256,
    )

    token_rows = context.batch_size * _SQUARE_COUNT
    attention_batches = context.batch_size * head_count
    chunk = _segment_chunk(context.batch_size)
    if chunk and sites:
        message = f"{prefix}: B3's int8 sites are built for the unchunked segment (unset LC0EX_SEGMENT_CHUNK)"
        raise NotImplementedError(message)
    chunk_count = context.batch_size // chunk if chunk else 1
    chunk_rows = token_rows // chunk_count
    chunk_batches = attention_batches // chunk_count
    set_count = _chunk_set_count(_CHUNK_MODE, chunk_count)
    if chunk:
        _LOGGER.info(
            "batch size %d: segment chunked c=%d (%d chunks, mode %s, %d sets)",
            context.batch_size,
            chunk,
            chunk_count,
            _CHUNK_MODE,
            set_count,
        )

    # One set of scratch buffers per concurrency slot, allocated in the same
    # order as the unchunked builder so that with chunking off the packing plan,
    # and therefore the artifact, is byte-for-byte what it was.  `chain`
    # allocates one set and reuses it, which is what serializes the chunks;
    # `free` allocates one per chunk, which is what lets the graph overlap them.
    scratch = [
        (
            (
                _temporary_f16(
                    context, element_count=chunk_batches * expected_smolgen_width
                )
                if has_smolgen
                else None
            ),
            _temporary_f16(context, element_count=chunk_rows * (3 * model_width)),
            _temporary_f16(context, element_count=chunk_rows * model_width),
            (
                None
                if _CHUNK_LN_WHOLE
                else _temporary_f16(context, element_count=chunk_rows * body_width)
            ),
        )
        for _ in range(set_count)
    ]
    # Whole when the norm is whole: the out-projection writes it by row range and
    # the norm reads it once, so it never has to be a per-chunk buffer.
    whole_branch = (
        _temporary_f16(context, element_count=token_rows * body_width)
        if _CHUNK_LN_WHOLE
        else None
    )
    attended = _temporary_f16(context, element_count=token_rows * body_width)
    if not chunk:
        # Unchunked: one chunk over the whole batch, no views anywhere, so the
        # emitted graph is identical to the pre-chunking builder's.
        body_chunks = [body]
        smolgen_chunks = [smolgen]
        attended_chunks = [attended]
        branch_chunks = [whole_branch]
    else:
        body_view = body.as_tensor(
            (token_rows, body_width), lc0ex_pb2.Buffer.DATA_TYPE_F16
        )
        attended_view = attended.as_tensor(
            (token_rows, body_width), lc0ex_pb2.Buffer.DATA_TYPE_F16
        )
        body_chunks = [
            body_view[index * chunk_rows : (index + 1) * chunk_rows, :]
            for index in range(chunk_count)
        ]
        attended_chunks = [
            attended_view[index * chunk_rows : (index + 1) * chunk_rows, :]
            for index in range(chunk_count)
        ]
        if _CHUNK_LN_WHOLE:
            branch_view = whole_branch.as_tensor(
                (token_rows, body_width), lc0ex_pb2.Buffer.DATA_TYPE_F16
            )
            branch_chunks = [
                branch_view[index * chunk_rows : (index + 1) * chunk_rows, :]
                for index in range(chunk_count)
            ]
        else:
            branch_chunks = [None] * chunk_count
        if has_smolgen:
            smolgen_view = smolgen.as_tensor(
                (attention_batches, generated_width),
                lc0ex_pb2.Buffer.DATA_TYPE_F16,
            )
            smolgen_chunks = [
                smolgen_view[
                    index * chunk_batches : (index + 1) * chunk_batches, :
                ]
                for index in range(chunk_count)
            ]
        else:
            smolgen_chunks = [None] * chunk_count

    for index in range(chunk_count):
        smolgen_logits, qkv_activations, merged, scratch_branch = scratch[
            index % set_count
        ]
        branch = branch_chunks[index] if _CHUNK_LN_WHOLE else scratch_branch
        body_chunk = body_chunks[index]
        if has_smolgen:
            matmul(
                context.builder,
                context.kernels,
                smolgen_logits,
                smolgen_chunks[index],
                shared_weights,
                MatmulSpecialization(
                    chunk_batches,
                    expected_smolgen_width,
                    generated_width,
                    context.architecture,
                ),
            )
        if "attn_in" in sites:
            _q1_gemm(context, qkv_activations, body_codes, prefix, "attn_in", chunk_rows, 3 * model_width, body_width,
                     epilogue="bias")
        elif _CUTLASS_QKV:
            cutlass_matmul(
                context.builder,
                context.kernels,
                qkv_activations,
                body_chunk,
                qkv_weights,
                CutlassMatmulSpecialization(
                    chunk_rows,
                    3 * model_width,
                    body_width,
                    context.architecture,
                    has_bias=True,
                    activation="none",
                ),
                bias=qkv_bias,
            )
        else:
            matmul(
                context.builder,
                context.kernels,
                qkv_activations,
                body_chunk,
                qkv_weights,
                MatmulSpecialization(
                    chunk_rows,
                    3 * model_width,
                    body_width,
                    context.architecture,
                    has_bias=True,
                    activation="none",
                ),
                bias=qkv_bias,
            )
        fused_attention(
            context.builder,
            context.kernels,
            merged,
            qkv_activations,
            smolgen_logits,
            scale,
            FusedAttentionSpecialization(
                batch_count=chunk_batches,
                model_width=model_width,
                head_depth=head_depth,
                heads_per_sample=head_count,
                architecture=context.architecture,
                has_smolgen=has_smolgen,
            ),
        )
        if "attn_out" in sites:
            merged_codes = _q1_codes(context, chunk_rows * model_width)
            quantise_operand(
                context.builder,
                context.kernels,
                merged_codes,
                merged,
                _q1_vector(context, prefix, "attn_out", "r", model_width),
                None,
                QuantiseOperandSpecialization(chunk_rows, model_width, context.architecture),
            )
            _q1_gemm(context, branch, merged_codes, prefix, "attn_out", chunk_rows, body_width, model_width,
                     epilogue="residual", skip=body_chunk)
        elif _CUTLASS_OUTPROJ:
            cutlass_matmul(
                context.builder,
                context.kernels,
                branch,
                merged,
                output_weights,
                CutlassMatmulSpecialization(
                    chunk_rows,
                    body_width,
                    model_width,
                    context.architecture,
                    has_bias=True,
                    has_skip=True,
                ),
                bias=output_bias,
                skip=body_chunk,
                alpha=alpha,
            )
        else:
            matmul(
                context.builder,
                context.kernels,
                branch,
                merged,
                output_weights,
                MatmulSpecialization(
                    chunk_rows,
                    body_width,
                    model_width,
                    context.architecture,
                    has_bias=True,
                    has_skip=True,
                ),
                bias=output_bias,
                skip=body_chunk,
                alpha=alpha,
            )
        if not _CHUNK_LN_WHOLE:
            layer_norm(
                context.builder,
                context.kernels,
                attended_chunks[index],
                branch,
                None,
                gammas,
                betas,
                LayerNormSpecialization(
                    row_count=chunk_rows,
                    width=body_width,
                    activation="none",
                    has_skip=False,
                    has_bias=False,
                    architecture=context.architecture,
                ),
            )
    attended_codes = None
    if _CHUNK_LN_WHOLE:
        attended_codes = _q1_norm(
            context,
            attended,
            whole_branch,
            gammas,
            betas,
            token_rows,
            body_width,
            prefix if "ffn_in" in sites else None,
            "ffn_in",
        )
    elif "ffn_in" in sites:
        message = f"{prefix}: B3's ffn_in copy is written by the whole ln1 (LC0EX_CHUNK_LN=whole)"
        raise NotImplementedError(message)
    return attended, attended_codes

def _ffn(
    context: _BuildContext,
    body: Buffer,
    body_width: int,
    encoder: net_pb2.Weights.EncoderLayer,
    *,
    prefix: str,
    sites: frozenset[str] = frozenset(),
    body_codes: Buffer | None = None,
    next_attn_in: str | None = None,
) -> tuple[Buffer, Buffer | None]:
    """Build an encoder FFN and its DeepNorm residual layer normalization.

    B3: `ffn_in` int8 = FFN1 on `cutlass_gemm_i8` (Mish in its epilogue) from ln1's copy; `ffn_mid` int8 = FFN1's
    epilogue writes the hidden as int8 codes (or, when FFN1 stays FP16, a `quantise_operand` pass does) and FFN2 runs
    on `cutlass_gemm_i8` with alpha and the skip in its epilogue. `next_attn_in` (the next encoder's prefix) makes ln2
    write the int8 copy that encoder's QKV reads (the second return value).
    """
    path = f"weights.{prefix[1:]}"
    dense1_weights, hidden_width = _matrix_f16(
        context,
        encoder.ffn.dense1_w,
        input_width=body_width,
        name=f"{prefix}/ffn/dense1/w/w",
        path=f"{path}.ffn.dense1_w",
    )
    dense1_bias = _vector_f16(
        context,
        encoder.ffn.dense1_b,
        name=f"{prefix}/ffn/dense1/b/w",
        path=f"{path}.ffn.dense1_b",
    )
    dense2_weights, _ = _matrix_f16(
        context,
        encoder.ffn.dense2_w,
        input_width=hidden_width,
        name=f"{prefix}/ffn/dense2/w/w",
        path=f"{path}.ffn.dense2_w",
    )
    dense2_bias = _vector_f16(
        context,
        encoder.ffn.dense2_b,
        name=f"{prefix}/ffn/dense2/b/w",
        path=f"{path}.ffn.dense2_b",
    )
    gammas = _vector_f16(
        context,
        encoder.ln2_gammas,
        name=f"{prefix}/ln2/w/scale",
        path=f"{path}.ln2_gammas",
    )
    betas = _vector_f16(
        context,
        encoder.ln2_betas,
        name=f"{prefix}/ln2/w/bias",
        path=f"{path}.ln2_betas",
    )
    alpha = context.builder.persistent_buffer(
        name=f"{prefix}/ffn/alpha/w",
        shape=(1,),
        dtype=lc0ex_pb2.Buffer.DATA_TYPE_F16,
        alignment_bytes=256,
    )

    token_rows = context.batch_size * _SQUARE_COUNT
    if "ffn_in" in sites and body_codes is None:
        message = f"{prefix}: ffn_in is int8 but ln1 wrote no int8 copy"
        raise ValueError(message)
    mid_int8 = "ffn_mid" in sites
    hidden = (_q1_codes(context, token_rows * hidden_width) if mid_int8 and "ffn_in" in sites
              else _temporary_f16(context, element_count=token_rows * hidden_width))
    if "ffn_in" in sites:
        _q1_gemm(context, hidden, body_codes, prefix, "ffn_in", token_rows, hidden_width, body_width,
                 epilogue="mish", output_format="i8" if mid_int8 else "f16",
                 prescale=_q1_vector(context, prefix, "ffn_mid", "r", hidden_width) if mid_int8 else None)
    elif _CUTLASS_FFN1:
        cutlass_matmul(
            context.builder,
            context.kernels,
            hidden,
            body,
            dense1_weights,
            CutlassMatmulSpecialization(
                token_rows,
                hidden_width,
                body_width,
                context.architecture,
                has_bias=True,
                activation="mish",
            ),
            bias=dense1_bias,
        )
    else:
        matmul(
            context.builder,
            context.kernels,
            hidden,
            body,
            dense1_weights,
            MatmulSpecialization(
                token_rows,
                hidden_width,
                body_width,
                context.architecture,
                has_bias=True,
                activation="mish",
            ),
            bias=dense1_bias,
        )
    branch = _temporary_f16(context, element_count=token_rows * body_width)
    if mid_int8:
        hidden_codes = hidden
        if "ffn_in" not in sites:
            hidden_codes = _q1_codes(context, token_rows * hidden_width)
            quantise_operand(
                context.builder,
                context.kernels,
                hidden_codes,
                hidden,
                _q1_vector(context, prefix, "ffn_mid", "r", hidden_width),
                None,
                QuantiseOperandSpecialization(token_rows, hidden_width, context.architecture),
            )
        _q1_gemm(context, branch, hidden_codes, prefix, "ffn_mid", token_rows, body_width, hidden_width,
                 epilogue="residual", skip=body)
    elif _CUTLASS_FFN2:
        cutlass_matmul(
            context.builder,
            context.kernels,
            branch,
            hidden,
            dense2_weights,
            CutlassMatmulSpecialization(
                token_rows,
                body_width,
                hidden_width,
                context.architecture,
                has_bias=True,
                has_skip=True,
            ),
            bias=dense2_bias,
            skip=body,
            alpha=alpha,
        )
    else:
        matmul(
            context.builder,
            context.kernels,
            branch,
            hidden,
            dense2_weights,
            MatmulSpecialization(
                token_rows,
                body_width,
                hidden_width,
                context.architecture,
                has_bias=True,
                has_skip=True,
            ),
            bias=dense2_bias,
            skip=body,
            alpha=alpha,
        )
    output = _temporary_f16(context, element_count=token_rows * body_width)
    codes = _q1_norm(context, output, branch, gammas, betas, token_rows, body_width, next_attn_in, "attn_in")
    return output, codes


def _policy_head(
    context: _BuildContext,
    body: Buffer,
    body_width: int,
    weights: net_pb2.Weights,
) -> None:
    """Build the selected vanilla attention-policy branch."""
    policy = weights.policy_heads.vanilla

    embedding_weights_layer = _policy_embedding_layer(weights, "ip_pol_w")
    embedding_bias_layer = _policy_embedding_layer(weights, "ip_pol_b")
    embedding_weights, policy_width = _matrix_f16(
        context,
        embedding_weights_layer,
        input_width=body_width,
        name="/policy/dense1/matmul/w",
        path="weights.policy_heads.vanilla.ip_pol_w",
    )
    embedding_bias = _vector_f16(
        context,
        embedding_bias_layer,
        name="/policy/dense1/add/w",
        path="weights.policy_heads.vanilla.ip_pol_b",
    )
    query_weights, model_width = _matrix_f16(
        context,
        policy.ip2_pol_w,
        input_width=policy_width,
        name="/policy/Q/matmul/w",
        path="weights.policy_heads.vanilla.ip2_pol_w",
    )
    query_bias = _vector_f16(
        context,
        policy.ip2_pol_b,
        name="/policy/Q/add/w",
        path="weights.policy_heads.vanilla.ip2_pol_b",
    )
    key_weights, _ = _matrix_f16(
        context,
        policy.ip3_pol_w,
        input_width=policy_width,
        name="/policy/K/matmul/w",
        path="weights.policy_heads.vanilla.ip3_pol_w",
    )
    key_bias = _vector_f16(
        context,
        policy.ip3_pol_b,
        name="/policy/K/add/w",
        path="weights.policy_heads.vanilla.ip3_pol_b",
    )
    promotion_weights, _ = _matrix_f16(
        context,
        policy.ip4_pol_w,
        input_width=model_width,
        name="/policy/promotion/matmul/w",
        path="weights.policy_heads.vanilla.ip4_pol_w",
    )
    scale = context.builder.persistent_buffer(
        name="/policy/scale/w",
        shape=(1,),
        dtype=lc0ex_pb2.Buffer.DATA_TYPE_F16,
        alignment_bytes=256,
    )
    output = context.builder.buffer(
        name="/output/policy",
        shape=(context.batch_size, 1858),
        dtype=lc0ex_pb2.Buffer.DATA_TYPE_F32,
        writable=True,
    )

    token_rows = context.batch_size * _SQUARE_COUNT
    embedded = _temporary_f16(context, element_count=token_rows * policy_width)
    matmul(
        context.builder,
        context.kernels,
        embedded,
        body,
        embedding_weights,
        MatmulSpecialization(
            token_rows,
            policy_width,
            body_width,
            context.architecture,
            has_bias=True,
            activation="mish",
        ),
        bias=embedding_bias,
    )
    query = _temporary_f16(context, element_count=token_rows * model_width)
    key = _temporary_f16(context, element_count=token_rows * model_width)
    for projected, projection_weights, bias in (
        (query, query_weights, query_bias),
        (key, key_weights, key_bias),
    ):
        matmul(
            context.builder,
            context.kernels,
            projected,
            embedded,
            projection_weights,
            MatmulSpecialization(
                token_rows,
                model_width,
                policy_width,
                context.architecture,
                has_bias=True,
                activation="none",
            ),
            bias=bias,
        )

    records = _temporary_f16(context, element_count=context.batch_size * 4288)
    batched_matmul(
        context.builder,
        context.kernels,
        records,
        query,
        key,
        BatchedMatmulSpecialization(
            "policy_qk",
            context.batch_size,
            _SQUARE_COUNT,
            _SQUARE_COUNT,
            model_width,
            1,
            context.architecture,
        ),
        scale=scale,
    )
    promotion_logits(
        context.builder,
        context.kernels,
        records,
        key,
        promotion_weights,
        PromotionLogitsSpecialization(
            context.batch_size, model_width, context.architecture
        ),
    )
    mapping = context.builder.add_symbol(
        compile_symbol(architecture=f"sm_{context.architecture}")
    )
    policy_map(
        context.builder,
        context.kernels,
        output,
        records,
        mapping,
        PolicyMapSpecialization(context.batch_size, context.architecture),
    )


def _value_head(
    context: _BuildContext,
    body: Buffer,
    body_width: int,
    winner: net_pb2.Weights.ValueHead,
) -> None:
    """Build the selected winner WDL branch."""
    _dense_output_head(
        context,
        body,
        body_width,
        embed_weight=winner.ip_val_w,
        embed_bias=winner.ip_val_b,
        dense1_weight=winner.ip1_val_w,
        dense1_bias=winner.ip1_val_b,
        dense2_weight=winner.ip2_val_w,
        dense2_bias=winner.ip2_val_b,
        prefix="/value",
        path="weights.value_heads.winner",
        output_name="/output/wdl",
        output_width=3,
        final_activation="none",
    )


def _moves_left_head(
    context: _BuildContext,
    body: Buffer,
    body_width: int,
    weights: net_pb2.Weights,
) -> None:
    """Build the selected moves-left branch."""
    _dense_output_head(
        context,
        body,
        body_width,
        embed_weight=weights.ip_mov_w,
        embed_bias=weights.ip_mov_b,
        dense1_weight=weights.ip1_mov_w,
        dense1_bias=weights.ip1_mov_b,
        dense2_weight=weights.ip2_mov_w,
        dense2_bias=weights.ip2_mov_b,
        prefix="/mlh",
        path="weights",
        output_name="/output/mlh",
        output_width=1,
        final_activation="relu",
    )


def _dense_output_head(  # noqa: PLR0913
    context: _BuildContext,
    body: Buffer,
    body_width: int,
    *,
    embed_weight: net_pb2.Weights.Layer,
    embed_bias: net_pb2.Weights.Layer,
    dense1_weight: net_pb2.Weights.Layer,
    dense1_bias: net_pb2.Weights.Layer,
    dense2_weight: net_pb2.Weights.Layer,
    dense2_bias: net_pb2.Weights.Layer,
    prefix: str,
    path: str,
    output_name: str,
    output_width: int,
    final_activation: Literal["none", "relu"],
) -> None:
    """Build a per-square embedding followed by two flattened dense layers."""
    embed_weights, embed_width = _matrix_f16(
        context,
        embed_weight,
        input_width=body_width,
        name=f"{prefix}/embed/matmul/w",
        path=f"{path}.{('ip_val_w' if prefix == '/value' else 'ip_mov_w')}",
    )
    embed_bias_buffer = _vector_f16(
        context,
        embed_bias,
        name=f"{prefix}/embed/add/w",
        path=f"{path}.{('ip_val_b' if prefix == '/value' else 'ip_mov_b')}",
    )
    flattened_width = _SQUARE_COUNT * embed_width
    hidden_weights, hidden_width = _matrix_f16(
        context,
        dense1_weight,
        input_width=flattened_width,
        name=f"{prefix}/dense1/matmul/w",
        path=f"{path}.{'ip1_val_w' if prefix == '/value' else 'ip1_mov_w'}",
    )
    hidden_bias = _vector_f16(
        context,
        dense1_bias,
        name=f"{prefix}/dense1/add/w",
        path=f"{path}.{'ip1_val_b' if prefix == '/value' else 'ip1_mov_b'}",
    )
    # B3: an int8 artifact with D1 reads the analyser's refit of the value layer under its own names, so the FP16
    # twin loading the same carrier keeps the trained layer (`_q1.D1_TARGETS`).
    value_d1 = prefix == "/value" and any(context.q1_sites.values()) and _q1.d1_enabled()
    result_weights, _ = _matrix_f16(
        context,
        dense2_weight,
        input_width=hidden_width,
        name=_q1.d1_name(f"{prefix}/dense2/matmul/w") if value_d1 else f"{prefix}/dense2/matmul/w",
        path=f"{path}.{'ip2_val_w' if prefix == '/value' else 'ip2_mov_w'}",
    )
    result_bias = _vector_f16(
        context,
        dense2_bias,
        name=_q1.d1_name(f"{prefix}/dense2/add/w") if value_d1 else f"{prefix}/dense2/add/w",
        path=f"{path}.{'ip2_val_b' if prefix == '/value' else 'ip2_mov_b'}",
    )
    output = context.builder.buffer(
        name=output_name,
        shape=(context.batch_size, output_width),
        dtype=lc0ex_pb2.Buffer.DATA_TYPE_F32,
        writable=True,
    )

    token_rows = context.batch_size * _SQUARE_COUNT
    embedded = _temporary_f16(context, element_count=token_rows * embed_width)
    matmul(
        context.builder,
        context.kernels,
        embedded,
        body,
        embed_weights,
        MatmulSpecialization(
            token_rows,
            embed_width,
            body_width,
            context.architecture,
            has_bias=True,
            activation="mish",
        ),
        bias=embed_bias_buffer,
    )
    hidden = _temporary_f16(context, element_count=context.batch_size * hidden_width)
    matmul(
        context.builder,
        context.kernels,
        hidden,
        embedded,
        hidden_weights,
        MatmulSpecialization(
            context.batch_size,
            hidden_width,
            flattened_width,
            context.architecture,
            has_bias=True,
            activation="mish",
        ),
        bias=hidden_bias,
    )
    matmul(
        context.builder,
        context.kernels,
        output,
        hidden,
        result_weights,
        MatmulSpecialization(
            context.batch_size,
            output_width,
            hidden_width,
            context.architecture,
            has_bias=True,
            activation=final_activation,
            output_f32=True,
        ),
        bias=result_bias,
    )


def _policy_embedding_layer(
    weights: net_pb2.Weights,
    field: str,
) -> net_pb2.Weights.Layer:
    """Resolve LC0's head-local, shared multihead, then legacy policy field."""
    policy = weights.policy_heads.vanilla
    local, shared, legacy = {
        "ip_pol_w": (policy.ip_pol_w, weights.policy_heads.ip_pol_w, weights.ip_pol_w),
        "ip_pol_b": (policy.ip_pol_b, weights.policy_heads.ip_pol_b, weights.ip_pol_b),
    }[field]
    if local.params:
        return local
    if weights.policy_heads.HasField(field):
        return shared
    return legacy


def _default_activation(
    value: net_pb2.NetworkFormat.DefaultActivation,
) -> net_pb2.NetworkFormat.ActivationFunction:
    """Map an LC0 default-activation enum to its concrete activation enum."""
    activations = {
        net_pb2.NetworkFormat.DEFAULT_ACTIVATION_RELU: (
            net_pb2.NetworkFormat.ACTIVATION_RELU
        ),
        net_pb2.NetworkFormat.DEFAULT_ACTIVATION_MISH: (
            net_pb2.NetworkFormat.ACTIVATION_MISH
        ),
    }
    return activations[value]


def _resolve_activation(
    value: net_pb2.NetworkFormat.ActivationFunction,
    default: net_pb2.NetworkFormat.ActivationFunction,
) -> net_pb2.NetworkFormat.ActivationFunction:
    """Resolve an explicit activation or the format default."""
    return default if value == net_pb2.NetworkFormat.ACTIVATION_DEFAULT else value


def _fingerprint_layer(
    context: _BuildContext,
    path: str,
) -> None:
    """Mark one consumed source layer in the sparse fingerprint."""
    context.fingerprint_layers[path].SetInParent()


def _vector_f16(
    context: _BuildContext,
    layer: net_pb2.Weights.Layer,
    *,
    name: str,
    path: str,
) -> Buffer:
    """Declare an FP16 vector whose width is inferred from its payload."""
    width = len(layer.params) // _F16_SIZE_BYTES
    _fingerprint_layer(context, path)
    return context.builder.persistent_buffer(
        name=name,
        shape=(width,),
        dtype=lc0ex_pb2.Buffer.DATA_TYPE_F16,
        alignment_bytes=256,
    )


def _matrix_f16(
    context: _BuildContext,
    layer: net_pb2.Weights.Layer,
    *,
    input_width: int,
    name: str,
    path: str,
) -> tuple[Buffer, int]:
    """Infer and declare an ONNX-layout FP16 matrix from its known input width."""
    element_count = len(layer.params) // _F16_SIZE_BYTES
    _fingerprint_layer(context, path)
    output_width = element_count // input_width
    return (
        context.builder.persistent_buffer(
            name=name,
            shape=(input_width, output_width),
            dtype=lc0ex_pb2.Buffer.DATA_TYPE_F16,
            alignment_bytes=256,
        ),
        output_width,
    )


def _q1_plan(weights: net_pb2.Weights) -> dict[int, frozenset[str]]:
    """B3: every encoder's int8 sites from the analyser's file (LC0EX_QUANT_GEMM=int8), or none (the FP16 build)."""
    blocks = len(weights.encoder)
    if blocks == 0:
        return {}
    d_model = len(weights.encoder[0].ln1_gammas.params) // _F16_SIZE_BYTES
    hidden = len(weights.encoder[0].ffn.dense1_b.params) // _F16_SIZE_BYTES
    vectors = _q1.load_vectors(blocks=blocks, d_model=d_model, hidden=hidden)
    plan = dict(_q1.check_blocks(vectors, blocks))
    if vectors is not None:
        served = sum(len(sites) for sites in plan.values())
        _LOGGER.info("B3: %d int8 sites of %d from %s (%s), D1 %s", served, 4 * blocks, vectors.path.name,
                     vectors.digest, "on" if _q1.d1_enabled() else "off")
        for index, sites in plan.items():
            if len(sites) < len(_q1.site_widths(d_model, hidden)):
                _LOGGER.info("B3: encoder %d keeps %s in FP16", index,
                             sorted(set(_q1.site_widths(d_model, hidden)) - sites))
    return plan


def _q1_vector(context: _BuildContext, prefix: str, site: str, kind: str, width: int) -> Buffer:
    """One FP32 vector of an int8 site (`scale`, `bias`: its output width; `r`: its input width)."""
    return context.builder.persistent_buffer(
        name=_q1.site_names(prefix, site)[kind],
        shape=(width,),
        dtype=lc0ex_pb2.Buffer.DATA_TYPE_F32,
        alignment_bytes=256,
    )


def _q1_codes(context: _BuildContext, element_count: int) -> Buffer:
    """An int8 operand temporary (one byte per element)."""
    return context.builder.temporary_buffer(size_bytes=element_count, alignment_bytes=256)


def _q1_gemm(  # noqa: PLR0913
    context: _BuildContext,
    output: Buffer,
    codes: Buffer,
    prefix: str,
    site: str,
    rows: int,
    n: int,
    k: int,
    *,
    epilogue: str,
    output_format: str = "f16",
    skip: Buffer | None = None,
    prescale: Buffer | None = None,
) -> None:
    """One int8 GEMM site: `[rows, k]` codes x the carrier's int8 `[n, k]` weights, the epilogue's FP32 vectors."""
    names = _q1.site_names(prefix, site)
    weights = context.builder.persistent_buffer(
        name=names["w"],
        shape=(n, k),
        dtype=lc0ex_pb2.Buffer.DATA_TYPE_U8,
        alignment_bytes=256,
    )
    cutlass_gemm_i8(
        context.builder,
        context.kernels,
        output,
        codes,
        weights,
        _q1_vector(context, prefix, site, "scale", n),
        _q1_vector(context, prefix, site, "bias", n),
        CutlassGemmI8Specialization(rows, n, k, context.architecture, epilogue=epilogue, output=output_format),
        skip=skip,
        prescale=prescale,
    )


def _q1_norm(  # noqa: PLR0913
    context: _BuildContext,
    output: Buffer,
    source: Buffer,
    gammas: Buffer,
    betas: Buffer,
    rows: int,
    width: int,
    consumer: str | None,
    site: str,
) -> Buffer | None:
    """A post-norm; with `consumer` (the encoder whose `site` reads it) it also writes that site's int8 copy.

    The copy converts the norm's own FP32 row through the site's `r` (no offset: the analyser's BT4 files carry no
    `m`), so the FP16 stream is bit-identical to the plain norm's.
    """
    codes = _q1_codes(context, rows * width) if consumer is not None else None
    prescale = _q1_vector(context, consumer, site, "r", width) if consumer is not None else None
    layer_norm(
        context.builder,
        context.kernels,
        output,
        source,
        None,
        gammas,
        betas,
        LayerNormSpecialization(
            row_count=rows,
            width=width,
            activation="none",
            has_skip=False,
            has_bias=False,
            architecture=context.architecture,
            quantise="int8" if consumer is not None else "",
            # The FP16 norms keep the spec's default, so an FP16 build compiles exactly the kernels it did before B3.
            quant_offset=consumer is None,
        ),
        quant_output=codes,
        quant_prescale=prescale,
        quant_offset=prescale,
    )
    return codes


def _temporary_f16(
    context: _BuildContext,
    *,
    element_count: int,
    alignment_bytes: int = 256,
) -> Buffer:
    """Allocate one opaque FP16 temporary by its raw byte extent."""
    return context.builder.temporary_buffer(
        size_bytes=element_count * _F16_SIZE_BYTES,
        alignment_bytes=alignment_bytes,
    )
