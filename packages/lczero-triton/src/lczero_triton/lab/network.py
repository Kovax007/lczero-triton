"""Build an lc0ex executable for the lab's static-mapping nets.

The static 512x15 net follows BT4's grammar where the two agree -- input
embedding with a preprocessing projection, post-norm encoder blocks with
residual scale factors, attention policy with promotions, dense value and
moves-left heads -- and departs where the export does:

* a per-position **prologue** builds the attack-graph edge tensor `E` and the
  pair normalization `S` once, read by every block and by the policy head
  (`prologue_static`);
* attention is **static-mapping** (`attention_static`): no smolgen, five logit
  terms collapsed into one four-channel contraction with build-time folds;
* every FFN is a **sigmoid-gated GLU** (`glu_matmul`), resolved through
  `gate_for_lab_config`, never guessed from `network_format`;
* Mish lives in the embedding and the heads only;
* the policy logits are `Q K^T` **unscaled**, plus the pair bias
  (`policy_static_bias`) before promotions are derived.

Buffer names and shapes come from `lab._names.plan_network` -- the same plans
the carrier generator writes -- so the two cannot disagree about a name.
"""

import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field

from lc0ex import Buffer, ExecutableBuilder, ProgramBuilder
from lc0ex.proto import lc0ex_metadata_pb2, lc0ex_pb2, net_pb2

from lczero_triton.bt4.kernels._activation import gate_for_lab_config
from lczero_triton.bt4.kernels._autotune import active_architecture
from lczero_triton.bt4.kernels._cache import KernelCache
from lczero_triton.bt4.kernels.add_vectors import AddVectorsSpecialization, add_vectors
from lczero_triton.bt4.kernels.attention_egt import (
    BLOCK_TABLES,
    DEFAULT_CAPACITY,
    EGT_FULL,
    EGT_STATE_TERMS,
    AttentionEgtSpecialization,
    EgtEdgeList,
    EgtEdgeListSpecialization,
    attention_egt,
    block_table_names,
    edge_list_bytes,
    egt_edge_list,
)
from lczero_triton.bt4.kernels.attention_static import (
    STATIC_EDGE,
    STATIC_FULL,
    STATIC_SMOLGEN,
    StaticAttentionSpecialization,
    static_attention,
)
from lczero_triton.bt4.kernels.batched_matmul import BatchedMatmulSpecialization, batched_matmul
from lczero_triton.bt4.kernels.cutlass_gemm_i8 import CutlassGemmI8Specialization, cutlass_gemm_i8
from lczero_triton.bt4.kernels.cutlass_matmul import (
    CutlassMatmulSpecialization,
    cutlass_matmul,
    cutlass_supports,
)
from lczero_triton.bt4.kernels.quantise_operand import QuantiseOperandSpecialization, quantise_operand
from lczero_triton.bt4.kernels.triplet_site import (
    TripletSiteSpecialization,
    readback_table_name,
    triplet_out_ffn,
    triplet_readback,
    triplet_site,
    triplet_site_with_readback,
)
from lczero_triton.bt4.kernels.triplet_site import buffer_bytes as triplet_buffer_bytes
from lczero_triton.bt4.kernels.triplet_site import triplet_table_names
from lczero_triton.bt4.kernels.egt_state_tiles import (
    EGT_STATE_AMAX,
    TILE_TABLES,
    CastStateF8Specialization,
    CastStateI8Specialization,
    CastStateSpecialization,
    StateTilesSpecialization,
    cast_state,
    cast_state_f8,
    cast_state_i8,
    egt_state_tiles,
    state_scales as egt_state_scales,
    state_scales_i8 as egt_state_scales_i8,
    tiles_bytes,
)
from lczero_triton.bt4.kernels.edge_site import (
    EdgeSiteSpecialization,
    edge_site,
    site_table_names,
)
from lczero_triton.bt4.kernels.expand_planes import ExpandPlanesSpecialization, expand_planes
from lczero_triton.bt4.kernels.glu_matmul import GluMatmulSpecialization, glu_matmul
from lczero_triton.bt4.kernels.input_gating import InputGatingSpecialization, input_gating
from lczero_triton.bt4.kernels.layer_norm import LayerNormSpecialization, layer_norm
from lczero_triton.bt4.kernels.mapping_table import compile_symbol
from lczero_triton.bt4.kernels.matmul import MatmulSpecialization, matmul
from lczero_triton.bt4.kernels.nchw_to_nhwc import NchwToNhwcSpecialization, nchw_to_nhwc
from lczero_triton.bt4.kernels.policy_egt_bias import (
    PolicyEgtBiasSpecialization,
    policy_egt_bias,
)
from lczero_triton.bt4.kernels.policy_map import PolicyMapSpecialization, policy_map
from lczero_triton.bt4.kernels.policy_static_bias import (
    PolicyStaticBiasSpecialization,
    policy_static_bias,
)
from lczero_triton.bt4.kernels.preprocess_attention_body import (
    PreprocessAttentionBodySpecialization,
    preprocess_attention_body,
)
from lczero_triton.bt4.kernels.prologue_egt import (
    EDGE_CHANNELS as _EGT_EDGE_CHANNELS,
)
from lczero_triton.bt4.kernels.prologue_egt import (
    SEED_TABLES as _EGT_SEED_TABLES,
)
from lczero_triton.bt4.kernels.prologue_egt import (
    STATE_CHANNELS as _EGT_STATE_CHANNELS,
)
from lczero_triton.bt4.kernels.prologue_egt import (
    PrologueEgtSpecialization,
    prologue_egt,
)
from lczero_triton.bt4.kernels.prologue_static import PrologueStaticSpecialization, prologue_static
from lczero_triton.bt4.kernels.promotion_logits import (
    PromotionLogitsSpecialization,
    promotion_logits,
)
from lczero_triton.lab import _onnx
from lczero_triton.lab._mapping import LabNetwork
from lczero_triton.lab._names import (D1_NAMES, BufferPlan, encoder_prefix, i8_conversion_names, i8_site_names,
                                       int8_blocks, output_gate_form, padded_hidden, plan_network, quant_d1_source,
                                       quant_names, quantised_norms)

_LOGGER = logging.getLogger(__name__)
_INPUT_CHANNELS = 112
_POSITION_CHANNELS = 12
_SQUARES = 64
_POLICY_RECORD = 4288
_F16_BYTES = 2
_F32_BYTES = 4
_U64_BYTES = 8
# The same opt-in switches the BT4 builder reads (R117): CUTLASS for the plain
# and skip-fused projections. The GLU FFN has no CUTLASS form yet.
# Q1 (ruling 09-22, ask 2): which operand format a quantised site writes. int8 is SPEC v2's default; e4m3 is the
# fallback, and the one the 09-21 accuracy read finds free on the value head. Only read where quantisation is on.
_QUANT_OPERAND = os.environ.get("LC0EX_QUANT_OPERAND", "int8")
if _QUANT_OPERAND not in ("int8", "e4m3"):
    message = f"LC0EX_QUANT_OPERAND={_QUANT_OPERAND!r}; expected int8 or e4m3"
    raise ValueError(message)
# Q1 (round 26): where an int8 block's attention output becomes the out-projection's int8 operand. "epilogue" (default):
# `attention_egt` writes the codes itself (C2's first fusion); "pass": the FP16 output plus a `quantise_operand` pass --
# kept as the twin arm that prices the fusion.
_QUANT_ATTN_OUT = os.environ.get("LC0EX_QUANT_ATTN_OUT", "epilogue")
if _QUANT_ATTN_OUT not in ("epilogue", "pass"):
    message = f"LC0EX_QUANT_ATTN_OUT={_QUANT_ATTN_OUT!r}; expected epilogue or pass"
    raise ValueError(message)
# Round 26b CONTROLS, never served: a build that must FAIL the reference gate, so a pass is evidence. `LC0EX_OGATE_SCALE_CONTROL`
# replaces the export's output-gate scale (2.0) on the EGT2 blocks; `LC0EX_FINAL_NORM_CONTROL=skip` drops a pre-norm
# EGT2 tower's final norm. Unset = the export's graph.
_OGATE_SCALE_CONTROL = os.environ.get("LC0EX_OGATE_SCALE_CONTROL", "")
_FINAL_NORM_CONTROL = os.environ.get("LC0EX_FINAL_NORM_CONTROL", "")
if _FINAL_NORM_CONTROL not in ("", "skip"):
    message = f"LC0EX_FINAL_NORM_CONTROL={_FINAL_NORM_CONTROL!r}; expected unset or skip"
    raise ValueError(message)
# Round 26 C2: the edge site FFN's maps -- "ieee" (K3's FP32 class, default) or "fp16" (tensor cores, FP32 accumulate).
_SITE_DOT = os.environ.get("LC0EX_SITE_DOT", "ieee")
if _SITE_DOT not in ("ieee", "fp16"):
    message = f"LC0EX_SITE_DOT={_SITE_DOT!r}; expected ieee or fp16"
    raise ValueError(message)
_CUTLASS_QKV = os.environ.get("LC0EX_CUTLASS_QKV") == "1"
_CUTLASS_OUTPROJ = os.environ.get("LC0EX_CUTLASS_OUTPROJ") == "1"
_CUTLASS_FFN2 = os.environ.get("LC0EX_CUTLASS_FFN2") == "1"
# Round 20b P2: the sigmoid GLU as a CUTLASS dual GEMM (needs the P1-padded dff).
_CUTLASS_GLU = os.environ.get("LC0EX_CUTLASS_GLU") == "1"
# Round 20c item E (W): slots per position in the EGT2 edge list. K2b made the kernel exact for any E -- a
# position with more than capacity - 1 set bits takes the dense fallback -- so this trades only speed. The
# default 1024 has ~1.7x headroom over the worst position seen in 153,688 sampled ones (590 set bits).
_EGT_CAPACITY = int(os.environ.get("LC0EX_EGT_CAPACITY", DEFAULT_CAPACITY))
# Round 21 K2c: how a block reads the edge state. "f32" is K2 as served in W (the head program contracts the FP32
# state, 262 KB per head); "f16" reads an FP16 copy refreshed after every state write; "tiles" reads three FP16
# tiles per head that `egt_state_tiles` builds once per (block, round). Rounds bound the tile buffer to L2.
_EGT_STATE = os.environ.get("LC0EX_EGT_STATE", "f32")
_EGT_ROUNDS = int(os.environ.get("LC0EX_EGT_ROUNDS", "1"))
_EGT_TILES_REVERSE = os.environ.get("LC0EX_EGT_TILES_REVERSE") == "1"
# r23 F8: "f8" reads an e4m3 copy (half the FP16 copy's bytes) through a per-channel build-time scale. `auto`
# is unchanged -- it never selects f8.
# r23b I8: "i8" reads an int8 copy through a per-channel scale (the same amax; tight by default,
# `LC0EX_EGT_STATE_SAFETY_I8`). `LC0EX_EGT_STATE_AUTO_UPPER` picks the copy `auto` serves above the tile rungs:
# f16 (the default -- `auto` unchanged), f8 or i8.
_EGT_STATE_MODES = ("f32", "f16", "f8", "i8", "tiles", "auto")
_EGT_STATE_SAFETY = float(os.environ.get("LC0EX_EGT_STATE_SAFETY", "2.0"))
_EGT_STATE_AMAX = os.environ.get("LC0EX_EGT_STATE_AMAX")
_EGT_STATE_SAFETY_I8 = float(os.environ.get("LC0EX_EGT_STATE_SAFETY_I8", "1.0"))
_EGT_STATE_AUTO_UPPER = os.environ.get("LC0EX_EGT_STATE_AUTO_UPPER", "f16")
# Round 26 C3 (analyser 09-23 item 16a): the one-byte copy with a static per-channel scale at EVERY WRITE -- the seed and
# each update site -- instead of one scale for the whole net. A JSON file {"writes": [[amax per channel] x (1 + sites)]}
# in write order; each block reads with the scale of the last write before it. Unset = the single-scale form above.
_EGT_STATE_AMAX_WRITES = os.environ.get("LC0EX_EGT_STATE_AMAX_WRITES", "")
if _EGT_STATE_AUTO_UPPER not in ("f16", "f8", "i8"):
    message = f"LC0EX_EGT_STATE_AUTO_UPPER={_EGT_STATE_AUTO_UPPER!r}; expected f16, f8 or i8"
    raise ValueError(message)
# Round 22 K2e: how positions past the edge list's capacity are served exactly. "branch" = K2b's in-program
# branch (13 us per call on every position); "correction" = a list-only main launch plus a second, early-exiting
# launch that recomputes only the overflow positions on the dense path (bit-identical results).
_EGT_OVERFLOW = os.environ.get("LC0EX_EGT_OVERFLOW", "branch")
if _EGT_OVERFLOW not in ("branch", "correction"):
    message = f"LC0EX_EGT_OVERFLOW={_EGT_OVERFLOW!r}; expected branch or correction"
    raise ValueError(message)
# "auto": tiles at or below this rung, the FP16 copy above it (the tile buffer must stay inside L2; K2c report).
_EGT_TILES_MAX_BATCH = int(os.environ.get("LC0EX_EGT_TILES_MAX_BATCH", "16"))
# B9 (Menkib's kernel audit, 09-26): the EGT2 attention kernel's arithmetic. LC0EX_EGT_ACC: "f32" (default, the served
# FP32 accumulators) or "f16" (the six dots accumulate in FP16). LC0EX_EGT_STATE_MATH: "f32" (default) or "f16" (the
# read / door / gate tiles formed in FP16). LC0EX_EGT_LIST_F16=1 is the gate's CONTROL (FP16 prefix sums): never served.
_EGT_ACC = os.environ.get("LC0EX_EGT_ACC", "f32")
_EGT_STATE_MATH = os.environ.get("LC0EX_EGT_STATE_MATH", "f32")
_EGT_LIST_F16 = os.environ.get("LC0EX_EGT_LIST_F16") == "1"
# LC0EX_EGT_ARITH=auto (B9's shipping form): autotune picks, per rung and card, the served arithmetic or
# LC0EX_EGT_ACC=f16 + LC0EX_EGT_STATE_MATH=f16 (gated equivalent); exclusive with setting either of those two.
_EGT_ARITH_AUTO = os.environ.get("LC0EX_EGT_ARITH", "fixed") == "auto"
if _EGT_ARITH_AUTO and (_EGT_ACC == "f16" or _EGT_STATE_MATH == "f16"):
    message = "LC0EX_EGT_ARITH=auto chooses the arithmetic itself; unset LC0EX_EGT_ACC / LC0EX_EGT_STATE_MATH"
    raise ValueError(message)
for _name, _value in (("LC0EX_EGT_ACC", _EGT_ACC), ("LC0EX_EGT_STATE_MATH", _EGT_STATE_MATH)):
    if _value not in ("f32", "f16"):
        message = f"{_name}={_value!r}; expected f32 or f16"
        raise ValueError(message)
# Round 22 K4b: how a triplet site is staged. "scratch" is K5 as served (K4's prep/contract/out between K3's
# readback and ffn stages), "readback_prep" fuses K3's readback into prep (K4b step a), "fused" runs prep and
# contract in one program per (sample, direction, head) with no scratch buffers (K4b step b). `out` and the FFN
# stage are the same in every form.
_TRIPLET_FORM = os.environ.get("LC0EX_TRIPLET_FORM", "scratch")
_TRIPLET_FORMS = ("scratch", "readback_prep", "fused")
# Form "fused" only. LC0EX_TRIPLET_DOT: "ieee" keeps K4's FP32 class; "fp16" contracts A and V as FP16 operands on
# the tensor cores with FP32 accumulation (the served class, gated at the served rule). LC0EX_TRIPLET_STATE16=1
# reads e_hat from an FP16 copy that `triplet_readback` writes in place of K3's readback stage (same class).
_TRIPLET_DOT = os.environ.get("LC0EX_TRIPLET_DOT", "ieee")
_TRIPLET_STATE16 = os.environ.get("LC0EX_TRIPLET_STATE16") == "1"
# Form "fused" only. LC0EX_TRIPLET_OUT_FFN=1 folds the triplet's `out` into K3's FFN stage (`triplet_out_ffn`,
# K4b step c): e_hat2 never reaches memory and the site closes in one planar launch.
_TRIPLET_OUT_FFN = os.environ.get("LC0EX_TRIPLET_OUT_FFN") == "1"
# r23b §7 (pricing): which EGT2 blocks read the edge stream. The exported family reads it in every block; this serves
# the net that reads it only in the named blocks -- elsewhere the head program's state terms are compiled out, exact
# for a net whose read, door and gate weights and gate bias are zero there. "all" (default, unchanged), "none",
# "every:N[:OFFSET]", "segment:K" (the first K blocks of each segment the update sites delimit), or block indices
# "3,4,9".
_EGT_READ_BLOCKS = os.environ.get("LC0EX_EGT_READ_BLOCKS", "all")
_EGT_NO_READ_TERMS = EGT_FULL & ~EGT_STATE_TERMS


def _egt_state_mode(batch: int) -> str:
    """The read mode this rung uses."""
    if _EGT_STATE != "auto":
        return _EGT_STATE
    return "tiles" if batch <= _EGT_TILES_MAX_BATCH else _EGT_STATE_AUTO_UPPER  # r23b I8: f16 unless set


def _egt_state_scales(channels: int) -> tuple[float, ...]:
    """r23 F8: the e4m3 copy's per-channel scale, a build-time constant of this family.

    The default is the measured amax of the 512x15 EGT2 state (`egt_state_tiles.EGT_STATE_AMAX`);
    `LC0EX_EGT_STATE_AMAX` overrides it with a comma-separated per-channel amax for another net.
    """
    amax = (tuple(float(value) for value in _EGT_STATE_AMAX.split(","))
            if _EGT_STATE_AMAX else EGT_STATE_AMAX)
    if len(amax) != channels:
        message = f"LC0EX_EGT_STATE_AMAX has {len(amax)} channels; the network has {channels}"
        raise ValueError(message)
    return egt_state_scales(amax, _EGT_STATE_SAFETY)


def _egt_state_scales_i8(channels: int) -> tuple[float, ...]:
    """r23b I8: the int8 copy's per-channel scale, from the same amax as the e4m3 copy's (tight by default)."""
    amax = (tuple(float(value) for value in _EGT_STATE_AMAX.split(","))
            if _EGT_STATE_AMAX else EGT_STATE_AMAX)
    if len(amax) != channels:
        message = f"LC0EX_EGT_STATE_AMAX has {len(amax)} channels; the network has {channels}"
        raise ValueError(message)
    return egt_state_scales_i8(amax, _EGT_STATE_SAFETY_I8)


def _egt_copy_scales(mode: str, channels: int, write: int, writes: int) -> tuple[float, ...]:
    """The one-byte copy's per-channel scale for one write (round 26 C3), or the single-scale form; () unless f8/i8."""
    if mode not in ("f8", "i8"):
        return ()
    if not _EGT_STATE_AMAX_WRITES:
        return _egt_state_scales(channels) if mode == "f8" else _egt_state_scales_i8(channels)
    table = json.loads(open(_EGT_STATE_AMAX_WRITES, encoding="utf-8").read())["writes"]  # noqa: PTH123, SIM115
    if len(table) != writes or any(len(row) != channels for row in table):
        message = (f"LC0EX_EGT_STATE_AMAX_WRITES holds {len(table)} writes of {[len(row) for row in table]} channels; "
                   f"this net writes its state {writes} times, {channels} channels each")
        raise ValueError(message)
    amax = tuple(float(value) for value in table[write])
    return (egt_state_scales(amax, _EGT_STATE_SAFETY) if mode == "f8"
            else egt_state_scales_i8(amax, _EGT_STATE_SAFETY_I8))


def _egt_read_blocks(blocks: int, site_blocks: Sequence[int]) -> frozenset[int]:
    """r23b §7: the blocks whose attention reads the edge stream (`LC0EX_EGT_READ_BLOCKS`)."""
    spec = _EGT_READ_BLOCKS.strip()
    if spec == "all":
        return frozenset(range(blocks))
    if spec == "none":
        return frozenset()
    if spec.startswith("every:"):
        parts = [int(part) for part in spec.split(":")[1:]]
        stride, offset = parts[0], parts[1] if len(parts) > 1 else 0
        chosen = {index for index in range(offset, blocks) if (index - offset) % stride == 0}
    elif spec.startswith("segment:"):
        first = int(spec.split(":")[1])
        starts = [0, *sorted(site + 1 for site in site_blocks if site + 1 < blocks)]
        ends = [*starts[1:], blocks]
        chosen = {index for start, end in zip(starts, ends, strict=True) for index in range(start, min(start + first, end))}
    else:
        chosen = {int(part) for part in spec.split(",") if part.strip()}
    if not chosen <= set(range(blocks)):
        message = f"LC0EX_EGT_READ_BLOCKS={spec!r} names blocks outside 0..{blocks - 1}"
        raise ValueError(message)
    return frozenset(chosen)
# The lab config's `ffn_activation: ACTIVATION_SWIGLU` is the sigmoid gate (AC-2).
_GATE = gate_for_lab_config("ACTIVATION_SWIGLU")
_DATA_TYPES = {
    _onnx.FLOAT16: lc0ex_pb2.Buffer.DATA_TYPE_F16,
    _onnx.FLOAT32: lc0ex_pb2.Buffer.DATA_TYPE_F32,
    _onnx.UINT8: lc0ex_pb2.Buffer.DATA_TYPE_U8,  # Q1: int8 weights, raw bytes
}


@dataclass(slots=True)
class _Context:
    """Construction services for one batch-size program."""

    builder: ProgramBuilder
    kernels: KernelCache
    batch_size: int
    architecture: int
    plans: dict[str, BufferPlan]
    declared: dict[str, Buffer] = field(default_factory=dict)
    # The lab's `ffn_softcap` (BT6-test): c of the capped sigmoid GLU in every gated FFN of the net; 0 = uncapped.
    ffn_softcap: float = 0.0
    # Q1: the full prefixes of the norms that also emit an int8 copy (`_names.quantised_norms`), or empty.
    quant_norms: frozenset[str] = frozenset()
    # Q1 (round 26): the blocks whose four GEMM sites run on `cutlass_gemm_i8`, and whether the value head's WDL
    # layer is the D1 refit (`_names.D1_NAMES`).
    int8_blocks: frozenset[int] = frozenset()
    d1: bool = False
    # Round 26: the quantised norms whose vector file carries an offset `m` (none on the flagship's recipe).
    quant_offset_norms: frozenset[str] = frozenset()

    @property
    def rows(self) -> int:
        return self.batch_size * _SQUARES

    def weight(self, name: str) -> Buffer:
        """Declare (once per program) the persistent buffer a plan names."""
        buffer = self.declared.get(name)
        if buffer is None:
            plan = self.plans[name]
            buffer = self.builder.persistent_buffer(
                name=plan.name,
                shape=plan.shape,
                dtype=_DATA_TYPES[plan.data_type],
                alignment_bytes=256,
            )
            self.declared[name] = buffer
        return buffer

    def temporary(self, element_count: int) -> Buffer:
        return self.builder.temporary_buffer(size_bytes=element_count * _F16_BYTES, alignment_bytes=256)

    def raw(self, size_bytes: int) -> Buffer:
        """Declare a temporary of an explicit byte size (the EGT2 buffers are uint64, FP32 and int16)."""
        return self.builder.temporary_buffer(size_bytes=size_bytes, alignment_bytes=256)


def network_fingerprint(network: net_pb2.Net) -> net_pb2.Net:
    """Mirror lc0's `lc0ex::BuildNetworkFingerprint` for this network.

    The runtime rebuilds this message from the weights file it loads and refuses
    an executable whose metadata differs. Every marker the C++ side creates is
    created here; a lab carrier has no `weights` stanza, so the head count is 0
    and there are no encoder entries.
    """
    fingerprint = net_pb2.Net()
    source = network.format.network_format
    target = fingerprint.format.network_format
    for name in ("input", "output", "network", "policy", "value", "moves_left", "input_embedding"):
        setattr(target, name, getattr(source, name))
    target.default_activation = source.default_activation
    default = {
        net_pb2.NetworkFormat.DEFAULT_ACTIVATION_RELU: net_pb2.NetworkFormat.ACTIVATION_RELU,
        net_pb2.NetworkFormat.DEFAULT_ACTIVATION_MISH: net_pb2.NetworkFormat.ACTIVATION_MISH,
    }[source.default_activation]

    def resolve(value: int) -> int:
        return default if value == net_pb2.NetworkFormat.ACTIVATION_DEFAULT else value

    target.ffn_activation = resolve(source.ffn_activation)
    target.smolgen_activation = resolve(source.smolgen_activation)

    weights = fingerprint.weights
    weights.headcount = network.weights.headcount
    for layer in (
        weights.ip_emb_preproc_w, weights.ip_emb_preproc_b, weights.ip_emb_w, weights.ip_emb_b,
        weights.ip_emb_ln_gammas, weights.ip_emb_ln_betas, weights.ip_mult_gate, weights.ip_add_gate,
        weights.ip_emb_ffn.dense1_w, weights.ip_emb_ffn.dense1_b,
        weights.ip_emb_ffn.dense2_w, weights.ip_emb_ffn.dense2_b,
        weights.ip_emb_ffn_ln_gammas, weights.ip_emb_ffn_ln_betas,
        weights.smolgen_w,
        weights.ip_mov_w, weights.ip_mov_b, weights.ip1_mov_w, weights.ip1_mov_b,
        weights.ip2_mov_w, weights.ip2_mov_b,
    ):
        layer.SetInParent()
    policy = weights.policy_heads.vanilla
    for layer in (
        policy.ip_pol_w, policy.ip_pol_b, policy.ip2_pol_w, policy.ip2_pol_b,
        policy.ip3_pol_w, policy.ip3_pol_b, policy.ip4_pol_w,
    ):
        layer.SetInParent()
    winner = weights.value_heads.winner
    for layer in (
        winner.ip_val_w, winner.ip_val_b, winner.ip1_val_w, winner.ip1_val_b,
        winner.ip2_val_w, winner.ip2_val_b,
    ):
        layer.SetInParent()
    for _ in network.weights.encoder:
        encoder = weights.encoder.add()
        smolgen = encoder.mha.smolgen
        for layer in (
            smolgen.compress, smolgen.dense1_w, smolgen.dense1_b, smolgen.ln1_gammas, smolgen.ln1_betas,
            smolgen.dense2_w, smolgen.dense2_b, smolgen.ln2_gammas, smolgen.ln2_betas,
            encoder.mha.q_w, encoder.mha.q_b, encoder.mha.k_w, encoder.mha.k_b,
            encoder.mha.v_w, encoder.mha.v_b, encoder.mha.dense_w, encoder.mha.dense_b,
            encoder.ln1_gammas, encoder.ln1_betas,
            encoder.ffn.dense1_w, encoder.ffn.dense1_b, encoder.ffn.dense2_w, encoder.ffn.dense2_b,
            encoder.ln2_gammas, encoder.ln2_betas,
        ):
            layer.SetInParent()
    return fingerprint


def build(
    builder: ExecutableBuilder,
    network: net_pb2.Net,
    lab: LabNetwork,
    *,
    batch_sizes: Sequence[int],
) -> None:
    """Append one program per batch size and the network fingerprint."""
    if lab.egt is not None:
        # W: K1's prologue, K2/K2b's attention and K3's update sites serve the gcap family; a triplet
        # export still raises, because K4 is not built.
        _check_egt_supported(lab)
    plans = {plan.name: plan for plan in plan_network(lab)}
    kernels = KernelCache(builder)
    shape = lab.architecture
    _LOGGER.info(
        "building lab static graph for batch sizes %s: %d blocks, d_model %d, %d heads, pair stream %s, smolgen %s, "
        "%s blocks, output gate %s (%s form)",
        list(batch_sizes), shape.blocks, shape.d_model, shape.heads, lab.pair is not None, shape.smolgen_gen > 0,
        shape.block_style, shape.output_gate, output_gate_form(),
    )
    served = int8_blocks(lab.architecture)
    if served:
        _LOGGER.info("Q1: int8 GEMM sites in %d of %d blocks (fp16: %s), D1 %s", len(served), shape.blocks,
                     sorted(set(range(shape.blocks)) - served) or "none", quant_d1_source() or "off")
    for size in batch_sizes:
        name = "main" if len(batch_sizes) == 1 else f"batch-{size}"
        program = builder.program(
            name=name,
            metadata=lc0ex_metadata_pb2.ProgramMetadata(batch_size=size).SerializeToString(deterministic=True),
        )
        context = _Context(program, kernels, size, active_architecture(), plans,
                           ffn_softcap=lab.architecture.ffn_softcap,
                           quant_norms=frozenset(f"{prefix}/{norm}" for prefix, norm, _, _
                                                 in quantised_norms(lab.architecture)),
                           int8_blocks=int8_blocks(lab.architecture), d1=bool(quant_d1_source()),
                           quant_offset_norms=_quant_offset_norms(lab))
        if context.int8_blocks and lab.egt is None:
            message = "LC0EX_QUANT_GEMM=int8 is wired for the EGT2 family (the flagship) only"
            raise NotImplementedError(message)
        if lab.egt is not None:
            _network_egt(context, lab)
        else:
            _network(context, lab)
        _LOGGER.info("finished program %s", name)
    builder.set_metadata(network_fingerprint(network).SerializeToString(deterministic=True))


def _quant_offset_norms(lab: LabNetwork) -> frozenset[str]:
    """The quantised norms whose site carries an offset in the loaded vector file (probe mode: none, m = 0)."""
    from lczero_triton.lab._quant import loaded_prescale  # noqa: PLC0415

    norms = quantised_norms(lab.architecture)
    loaded = loaded_prescale()
    if loaded is None:
        return frozenset()
    return frozenset(f"{prefix}/{norm}" for prefix, norm, scope, site in norms
                     if (vectors := loaded.get(scope, site)) is not None and vectors.offset is not None)


def _network(context: _Context, lab: LabNetwork) -> None:
    shape = lab.architecture
    batch = context.batch_size
    masks = context.builder.buffer(
        name="/input/plane_masks", shape=(batch, _INPUT_CHANNELS), dtype=lc0ex_pb2.Buffer.DATA_TYPE_U64,
    )
    values = context.builder.buffer(
        name="/input/plane_values", shape=(batch, _INPUT_CHANNELS), dtype=lc0ex_pb2.Buffer.DATA_TYPE_F32,
    )
    planes = context.temporary(batch * _INPUT_CHANNELS * _SQUARES)
    expand_planes(
        context.builder, context.kernels, planes, masks, values,
        ExpandPlanesSpecialization(batch * _INPUT_CHANNELS, context.architecture),
    )
    # P5: E packed one byte per cell (at most 8 edge channels), read by attention and the policy bias.
    edges = context.builder.temporary_buffer(size_bytes=batch * _SQUARES * _SQUARES, alignment_bytes=256)
    edge_norm = context.temporary(batch * _SQUARES * _SQUARES)
    prologue_static(
        context.builder, context.kernels, edges, edge_norm, masks, values,
        context.weight("/prologue/channel_mix"), context.weight("/prologue/offsets"),
        PrologueStaticSpecialization(batch, context.architecture),
    )
    body, _ = _embedding(context, lab, planes)
    for index in range(shape.blocks):
        _LOGGER.info("batch size %d: building encoder %d/%d", batch, index + 1, shape.blocks)
        body = _encoder(context, lab, body, edges, edge_norm, index)
    if lab.final_norm is not None:
        # O: the pre-norm tower is normed once, here, before every head.
        normed = context.temporary(context.rows * shape.d_model)
        _norm(context, normed, body, "/encoder/final_norm", shape.d_model, "none")
        body = normed
    _policy_head(context, lab, body, edges, edge_norm)
    _dense_head(context, body, "/value", hidden_width=128, square_width=128, output_name="/output/wdl",
                output_width=3, final_activation="none")
    _dense_head(context, body, "/moves", hidden_width=128, square_width=8, output_name="/output/mlh",
                output_width=1, final_activation="relu")


def _cutlass_fits(prefix: str, out_width: int, in_width: int) -> bool:
    """Route to CUTLASS only when both widths meet its alignment of 8."""
    if cutlass_supports(out_width, in_width):
        return True
    _LOGGER.info("%s: widths %d -> %d are not multiples of 8; kept on Triton", prefix, in_width, out_width)
    return False


def _projection(  # noqa: PLR0913
    context: _Context, output: Buffer, activations: Buffer, prefix: str, rows: int, out_width: int,
    in_width: int, *, use_cutlass: bool, activation: str = "none",
) -> None:
    """`act(x @ W + b)` for the `{prefix}/w`, `{prefix}/b` pair."""
    weights, bias = context.weight(f"{prefix}/w"), context.weight(f"{prefix}/b")
    if use_cutlass and _cutlass_fits(prefix, out_width, in_width):
        cutlass_matmul(
            context.builder, context.kernels, output, activations, weights,
            CutlassMatmulSpecialization(rows, out_width, in_width, context.architecture, has_bias=True,
                                        activation=activation),
            bias=bias,
        )
    else:
        matmul(
            context.builder, context.kernels, output, activations, weights,
            MatmulSpecialization(rows, out_width, in_width, context.architecture, has_bias=True,
                                 activation=activation),
            bias=bias,
        )


def _skip_projection(  # noqa: PLR0913
    context: _Context, output: Buffer, activations: Buffer, prefix: str, rows: int, out_width: int,
    in_width: int, *, skip: Buffer, alpha: Buffer, use_cutlass: bool,
) -> None:
    """`(x @ W + b) * alpha + skip`, the residual branch of a post-norm block."""
    weights, bias = context.weight(f"{prefix}/w"), context.weight(f"{prefix}/b")
    if use_cutlass and _cutlass_fits(prefix, out_width, in_width):
        cutlass_matmul(
            context.builder, context.kernels, output, activations, weights,
            CutlassMatmulSpecialization(rows, out_width, in_width, context.architecture, has_bias=True,
                                        has_skip=True),
            bias=bias, skip=skip, alpha=alpha,
        )
    else:
        matmul(
            context.builder, context.kernels, output, activations, weights,
            MatmulSpecialization(rows, out_width, in_width, context.architecture, has_bias=True,
                                 has_skip=True),
            bias=bias, skip=skip, alpha=alpha,
        )


def _norm(context: _Context, output: Buffer, source: Buffer, prefix: str, width: int, activation: str) -> Buffer | None:
    """One post-norm, and -- at a quantised site (Q1) -- the int8 copy of its own output beside it.

    The copy is emitted here rather than by a second pass because this row already holds the FP32 value: it costs
    the bytes it writes and nothing else. ⚠ Until Q1's integer GEMM lands nothing reads `codes`; with
    `LC0EX_QUANT_PRESCALE=probe` that is deliberate, and prices the conversion in a served graph.
    """
    quantise = prefix in context.quant_norms
    codes = context.raw(context.rows * width) if quantise else None
    prescale, offset = quant_names(*prefix.rsplit("/", 1)) if quantise else ("", "")
    layer_norm(
        context.builder, context.kernels, output, source, None,
        context.weight(f"{prefix}/scale"), context.weight(f"{prefix}/bias"),
        LayerNormSpecialization(row_count=context.rows, width=width, activation=activation, has_skip=False,
                                has_bias=False, architecture=context.architecture,
                                quantise=_QUANT_OPERAND if quantise else "",
                                quant_offset=prefix in context.quant_offset_norms),
        quant_output=codes,
        quant_prescale=context.weight(prescale) if quantise else None,
        quant_offset=context.weight(offset) if quantise else None,
    )
    return codes


def _gated_ffn_branch(  # noqa: PLR0913
    context: _Context, prefix: str, source: Buffer, *, skip: Buffer, width: int, hidden: int,
    codes: Buffer | None = None,
) -> Buffer:
    """Sigmoid-gated GLU FFN of `source`; FFN2's epilogue scales it by alpha and adds `skip`. No norm (O).

    With `codes` (Q1, an int8 block) the same FFN runs on `cutlass_gemm_i8`: FFN1 reads the norm's int8 copy and
    writes the hidden as int8 codes through the `ffn_mid` vector (the GLU in its epilogue), FFN2 reads those codes.
    """
    hidden = padded_hidden(hidden)  # P1: zero-padded to CUTLASS's alignment, exact (a padded column's up half is 0)
    rows = context.rows
    if codes is not None:
        if _GATE != "glu":
            message = f"the int8 FFN serves the sigmoid GLU; this net's gate is {_GATE!r}"
            raise NotImplementedError(message)
        stems, conversions = i8_site_names(prefix), i8_conversion_names(prefix)
        hidden_codes = context.raw(rows * hidden)
        cutlass_gemm_i8(
            context.builder, context.kernels, hidden_codes, codes, context.weight(f"{stems['ffn_in']}/w"),
            context.weight(f"{stems['ffn_in']}/scale"), context.weight(f"{stems['ffn_in']}/bias"),
            CutlassGemmI8Specialization(rows, hidden, width, context.architecture, epilogue="glu", output="i8",
                                        glu_softcap=context.ffn_softcap),
            prescale=context.weight(conversions["ffn_mid"]),
        )
        branch = context.temporary(rows * width)
        cutlass_gemm_i8(
            context.builder, context.kernels, branch, hidden_codes, context.weight(f"{stems['ffn_mid']}/w"),
            context.weight(f"{stems['ffn_mid']}/scale"), context.weight(f"{stems['ffn_mid']}/bias"),
            CutlassGemmI8Specialization(rows, width, hidden, context.architecture, epilogue="residual"),
            skip=skip,
        )
        return branch
    hidden_buffer = context.temporary(rows * hidden)
    # BT6-test: with `ffn_softcap` both branches are capped before the product -- the same gate family, one more
    # constant in the epilogue (CUTLASS `glu_softcap`) or the Triton gate "glu_capped".
    softcap = context.ffn_softcap
    if _CUTLASS_GLU and _GATE == "glu" and _cutlass_fits(f"{prefix}/ffn/dense1", hidden, width):
        cutlass_matmul(
            context.builder, context.kernels, hidden_buffer, source, context.weight(f"{prefix}/ffn/dense1/w"),
            CutlassMatmulSpecialization(rows, hidden, width, context.architecture, has_bias=True, glu=True,
                                        glu_softcap=softcap),
            bias=context.weight(f"{prefix}/ffn/dense1/b"),
        )
    else:
        gate = gate_for_lab_config("ACTIVATION_SWIGLU", ffn_softcap=softcap) if softcap > 0.0 else _GATE
        glu_matmul(
            context.builder, context.kernels, hidden_buffer, source, context.weight(f"{prefix}/ffn/dense1/w"),
            GluMatmulSpecialization(m=rows, n=hidden, k=width, architecture=context.architecture, gate=gate,
                                    has_bias=True, softcap=softcap),
            bias=context.weight(f"{prefix}/ffn/dense1/b"),
        )
    branch = context.temporary(rows * width)
    _skip_projection(context, branch, hidden_buffer, f"{prefix}/ffn/dense2", rows, width, hidden, skip=skip,
                     alpha=context.weight(f"{prefix}/ffn/alpha/w"), use_cutlass=_CUTLASS_FFN2)
    return branch


def _gated_ffn(  # noqa: PLR0913
    context: _Context, prefix: str, norm_prefix: str, body: Buffer, width: int, hidden: int,
    codes: Buffer | None = None,
) -> tuple[Buffer, Buffer | None]:
    """Sigmoid-gated GLU FFN, residual scale and skip, then the post-norm -- and the norm's int8 copy, if any."""
    branch = _gated_ffn_branch(context, prefix, body, skip=body, width=width, hidden=hidden, codes=codes)
    output = context.temporary(context.rows * width)
    return output, _norm(context, output, branch, norm_prefix, width, "none")


def _embedding(context: _Context, lab: LabNetwork, planes: Buffer) -> tuple[Buffer, Buffer | None]:
    shape = lab.architecture
    batch, rows, width = context.batch_size, context.rows, shape.d_model
    # The preprocess dense writes `dense` features per square (the lab's `embedding.dense_size`): d_model on the 512
    # nets, 512 on the 1024 sponsor net -- BT4's own 1024 x 15 has the same 512.
    dense = shape.embedding_dense or width
    position_input_width = _SQUARES * _POSITION_CHANNELS
    position_width = _SQUARES * dense
    position_input = context.temporary(batch * position_input_width)
    nchw_to_nhwc(
        context.builder, context.kernels, position_input, planes,
        NchwToNhwcSpecialization(batch, _INPUT_CHANNELS, _POSITION_CHANNELS, 8, 8, context.architecture),
    )
    position = context.temporary(batch * position_width)
    matmul(
        context.builder, context.kernels, position, position_input, context.weight("/attn_body/preproc/w"),
        MatmulSpecialization(batch, position_width, position_input_width, context.architecture),
    )
    add_vectors(
        context.builder, context.kernels, position, position, context.weight("/attn_body/preproc/b"),
        AddVectorsSpecialization(batch * position_width, position_width, "none", context.architecture),
    )
    embedding_input_width = _INPUT_CHANNELS + dense
    embedding_input = context.temporary(rows * embedding_input_width)
    preprocess_attention_body(
        context.builder, context.kernels, embedding_input, planes, position,
        PreprocessAttentionBodySpecialization(batch, _INPUT_CHANNELS, dense, context.architecture),
    )
    projected = context.temporary(rows * width)
    _projection(context, projected, embedding_input, "/attn_body/matmul", rows, width, embedding_input_width,
                use_cutlass=False)
    normalized = context.temporary(rows * width)
    _norm(context, normalized, projected, "/attn_body/ln0", width, "mish")
    gated = context.temporary(rows * width)
    input_gating(
        context.builder, context.kernels, gated, normalized,
        context.weight("/attn_body/mult_gate/w"), context.weight("/attn_body/position/w"),
        InputGatingSpecialization(batch, _SQUARES, width, context.architecture),
    )
    return _gated_ffn(context, "/attn_body", "/attn_body/ln1", gated, width, shape.embedding_ffn_hidden)


def _smolgen(context: _Context, lab: LabNetwork, body: Buffer, prefix: str) -> Buffer:
    """BT4's smolgen chain on lab names: the `[batch * heads, 64 * 64]` generated logits.

    compress (per square, no bias) -> flatten -> dense1 -> swish+norm -> dense2 ->
    swish+norm -> the shared weight_gen (no bias). The export applies swish before
    each norm, the order `layer_norm`'s fused activation takes; epsilon 1e-3 both.
    """
    shape = lab.architecture
    batch, arch = context.batch_size, context.architecture
    channels, hidden = shape.smolgen_channels, shape.smolgen_hidden
    generated = shape.heads * shape.smolgen_gen
    compressed = context.temporary(context.rows * channels)
    matmul(
        context.builder, context.kernels, compressed, body, context.weight(f"{prefix}/smolgen/compress/w"),
        MatmulSpecialization(context.rows, channels, shape.d_model, arch),
    )
    hidden_buffer = context.temporary(batch * hidden)
    matmul(
        context.builder, context.kernels, hidden_buffer, compressed, context.weight(f"{prefix}/smolgen/dense1/w"),
        MatmulSpecialization(batch, hidden, _SQUARES * channels, arch, has_bias=True),
        bias=context.weight(f"{prefix}/smolgen/dense1/b"),
    )
    layer_norm(
        context.builder, context.kernels, hidden_buffer, hidden_buffer, None,
        context.weight(f"{prefix}/smolgen/ln1/scale"), context.weight(f"{prefix}/smolgen/ln1/bias"),
        LayerNormSpecialization(row_count=batch, width=hidden, activation="swish", has_skip=False,
                                has_bias=False, architecture=arch),
    )
    generated_buffer = context.temporary(batch * generated)
    matmul(
        context.builder, context.kernels, generated_buffer, hidden_buffer, context.weight(f"{prefix}/smolgen/dense2/w"),
        MatmulSpecialization(batch, generated, hidden, arch, has_bias=True),
        bias=context.weight(f"{prefix}/smolgen/dense2/b"),
    )
    layer_norm(
        context.builder, context.kernels, generated_buffer, generated_buffer, None,
        context.weight(f"{prefix}/smolgen/ln2/scale"), context.weight(f"{prefix}/smolgen/ln2/bias"),
        LayerNormSpecialization(row_count=batch, width=generated, activation="swish", has_skip=False,
                                has_bias=False, architecture=arch),
    )
    logits = context.temporary(batch * shape.heads * _SQUARES * _SQUARES)
    matmul(
        context.builder, context.kernels, logits, generated_buffer, context.weight("/smolgen/weight_gen/w"),
        MatmulSpecialization(batch * shape.heads, _SQUARES * _SQUARES, shape.smolgen_gen, arch),
    )
    return logits


def _encoder(context: _Context, lab: LabNetwork, body: Buffer, edges: Buffer, edge_norm: Buffer, index: int) -> Buffer:
    prefix = encoder_prefix(index)
    block = lab.blocks[index]
    shape = lab.architecture
    rows, width = context.rows, shape.d_model
    prenorm = shape.block_style == "prenorm"
    gated = block.gate_weight is not None
    packed_gate = gated and output_gate_form() == "packed"
    # O, pre-norm: LN1 feeds the projections; the residual stream `body` passes through un-normed.
    source = body
    if prenorm:
        source = context.temporary(rows * width)
        _norm(context, source, body, f"{prefix}/ln1", width, "none")
    # P3: one packed projection, [q | k | v] per row, read in place by static_attention. O: [q | k | v | g] with
    # the output gate's pre-activations as a fourth lane, or a separate gate projection of the same input.
    lanes = 4 if packed_gate else 3
    qkv = context.temporary(rows * lanes * width)
    _projection(context, qkv, source, f"{prefix}/mha/qkvg" if packed_gate else f"{prefix}/mha/qkv", rows,
                lanes * width, width, use_cutlass=_CUTLASS_QKV)
    gates = None
    if gated and not packed_gate:
        gates = context.temporary(rows * width)
        _projection(context, gates, source, f"{prefix}/mha/gate", rows, width, width, use_cutlass=_CUTLASS_QKV)
    queries = keys = values = qkv
    smolgen = _smolgen(context, lab, source, prefix) if block.smolgen is not None else None
    scale = context.weight("/encoder/qk_scale")
    if lab.pair is not None:
        logit_terms = STATIC_FULL
        scaled = context.weight(f"{prefix}/pair/scaled_codes")  # P6
        constant = context.weight(f"{prefix}/pair/constant_bias")
    else:
        # No pair stream: the kernel compiles those loads out and the pointer slots
        # take a placeholder, as `fused_attention` does for an absent smolgen.
        logit_terms, scaled, constant = STATIC_EDGE, scale, scale
    if smolgen is not None:
        logit_terms |= STATIC_SMOLGEN
    merged = context.temporary(rows * width)
    static_attention(
        context.builder, context.kernels, merged, queries, keys, values, edges, edge_norm,
        # P6: per-code tables; the per-channel pair tables stay in the carrier unread.
        context.weight(f"{prefix}/mha/edge/query_codes"), context.weight(f"{prefix}/mha/edge/key_codes"),
        context.weight(f"{prefix}/mha/edge/coefficient_codes"), scaled, constant, scale,
        StaticAttentionSpecialization(
            batch_count=context.batch_size * shape.heads, heads=shape.heads, tokens=_SQUARES,
            head_dim=shape.head_dim, edge_channels=shape.edge_channels, logit_terms=logit_terms,
            architecture=context.architecture, packed_qkv=True, output_gate=gated, gate_packed=packed_gate,
            gate_scale=block.gate_scale if gated else 2.0,
        ),
        smolgen_logits=smolgen,
        gates=gates,
    )
    branch = context.temporary(rows * width)
    _skip_projection(context, branch, merged, f"{prefix}/mha/out", rows, width, width, skip=body,
                     alpha=context.weight(f"{prefix}/mha/alpha/w"), use_cutlass=_CUTLASS_OUTPROJ)
    if prenorm:
        # O: `branch` is the stream; LN2 feeds the GLU, FFN2 adds into the stream, nothing norms it here.
        normed = context.temporary(rows * width)
        _norm(context, normed, branch, f"{prefix}/ln2", width, "none")
        return _gated_ffn_branch(context, prefix, normed, skip=branch, width=width, hidden=shape.ffn_hidden)
    attended = context.temporary(rows * width)
    _norm(context, attended, branch, f"{prefix}/ln1", width, "none")
    return _gated_ffn(context, prefix, f"{prefix}/ln2", attended, width, shape.ffn_hidden)[0]


def _check_egt_supported(lab: LabNetwork) -> None:
    """Refuse an EGT2 export this builder cannot serve exactly.

    The gcap family (readback plus the edge FFN at the three sites) and the two triplet exports (K4's operator
    between them, `path` or `ag` by the reader's contraction) are what `prologue_egt`, `attention_egt`,
    `edge_site` and `triplet_site` implement.
    """
    egt = lab.egt
    if egt is None:  # pragma: no cover - the caller checks
        return
    unknown = [site.after_block for site in egt.sites
               if site.triplet is not None and site.triplet.contraction not in ("path", "ag")]
    if unknown:
        message = f"EGT2 export with an unknown triplet contraction at sites {unknown}"
        raise NotImplementedError(message)
    if any(block.smolgen is not None for block in lab.blocks):
        message = "EGT2 export with smolgen blocks: `attention_egt` has no smolgen term"
        raise NotImplementedError(message)
    if lab.architecture.output_gate and output_gate_form() != "packed":
        message = ("EGT2 export with the attention output gate: `attention_egt` reads it as the packed projection's "
                   "fourth lane only (LC0EX_OGATE_FORM=packed)")
        raise NotImplementedError(message)
    shape = lab.architecture
    if lab.pair is None:
        message = "EGT2 export without a pair stream: the logit assembly reads S and the folded pair tables"
        raise ValueError(message)
    if (shape.edge_channels, egt.state_channels) != (_EGT_EDGE_CHANNELS, _EGT_STATE_CHANNELS):
        message = (
            f"EGT2 export with {shape.edge_channels} edge and {egt.state_channels} state channels; the kernels are "
            f"built for {_EGT_EDGE_CHANNELS} and {_EGT_STATE_CHANNELS}"
        )
        raise ValueError(message)


def _network_egt(context: _Context, lab: LabNetwork) -> None:
    """Append one EGT2 (gcap) program: the 34-channel prologue, the edge list, 15 blocks and three update sites."""
    shape = lab.architecture
    egt = lab.egt
    assert egt is not None  # noqa: S101 - `build` checked it
    batch, cells = context.batch_size, context.batch_size * _SQUARES * _SQUARES
    sites = {site.after_block: site for site in egt.sites}
    if _EGT_STATE not in _EGT_STATE_MODES:
        message = f"LC0EX_EGT_STATE={_EGT_STATE!r}; expected one of {_EGT_STATE_MODES}"
        raise ValueError(message)
    if _EGT_ROUNDS < 1 or shape.heads % _EGT_ROUNDS:
        message = f"LC0EX_EGT_ROUNDS={_EGT_ROUNDS} must divide the head count {shape.heads}"
        raise ValueError(message)
    if _TRIPLET_FORM not in _TRIPLET_FORMS:
        message = f"LC0EX_TRIPLET_FORM={_TRIPLET_FORM!r}; expected one of {_TRIPLET_FORMS}"
        raise ValueError(message)
    if any(site.triplet is not None for site in egt.sites):
        _LOGGER.info("batch size %d: triplet sites staged as %r (LC0EX_TRIPLET_FORM), dot %r, state16 %s, "
                     "out_ffn %s", batch, _TRIPLET_FORM, _TRIPLET_DOT, _TRIPLET_STATE16, _TRIPLET_OUT_FFN)
    _LOGGER.info(
        "building lab EGT2 graph for batch size %d: %d blocks, d_model %d, %d heads, %d edge channels, "
        "%d state channels, sites after blocks %s, gate cap %s, edge-list capacity %d, state read mode %s, "
        "%d head rounds%s",
        batch, shape.blocks, shape.d_model, shape.heads, shape.edge_channels, egt.state_channels,
        sorted(sites), egt.cap, _EGT_CAPACITY, _egt_state_mode(batch), _EGT_ROUNDS,
        " (reverse)" if _EGT_TILES_REVERSE else "",
    )
    _LOGGER.info("EGT2 overflow mode %s", _EGT_OVERFLOW)
    read_blocks = _egt_read_blocks(shape.blocks, sorted(sites))  # r23b §7
    _LOGGER.info("EGT2 blocks reading the edge stream (LC0EX_EGT_READ_BLOCKS=%s): %d of %d %s", _EGT_READ_BLOCKS,
                 len(read_blocks), shape.blocks, sorted(read_blocks))
    masks = context.builder.buffer(
        name="/input/plane_masks", shape=(batch, _INPUT_CHANNELS), dtype=lc0ex_pb2.Buffer.DATA_TYPE_U64,
    )
    values = context.builder.buffer(
        name="/input/plane_values", shape=(batch, _INPUT_CHANNELS), dtype=lc0ex_pb2.Buffer.DATA_TYPE_F32,
    )
    planes = context.temporary(batch * _INPUT_CHANNELS * _SQUARES)
    expand_planes(
        context.builder, context.kernels, planes, masks, values,
        ExpandPlanesSpecialization(batch * _INPUT_CHANNELS, context.architecture),
    )
    # K1: E packed one uint64 per cell (bit c = export channel c), S in FP16 and the edge-stream seed e0 in FP32.
    # Two state buffers alternate: a site writes the one the following blocks read, and never in place.
    edges = context.raw(_U64_BYTES * cells)
    edge_norm = context.temporary(cells)
    state = (context.raw(_F32_BYTES * egt.state_channels * cells),
             context.raw(_F32_BYTES * egt.state_channels * cells))
    prologue_egt(
        context.builder, context.kernels, edges, edge_norm, state[0], masks, values,
        {name: context.weight(name) for name in _EGT_SEED_TABLES},
        PrologueEgtSpecialization(batch, context.architecture),
    )
    # K2: E's set bits, listed once per batch and read by every block. A position with more than capacity - 1 set
    # bits takes the kernel's exact dense fallback (K2b), so this only trades speed.
    list_specialization = EgtEdgeListSpecialization(batch, context.architecture, _EGT_CAPACITY)
    sizes = edge_list_bytes(list_specialization)
    edge_list = EgtEdgeList(
        context.raw(sizes["edge_list cells"]), context.raw(sizes["edge_list channels"]),
        context.raw(sizes["edge_list prefix"]), context.raw(sizes["edge_list counts"]),
        context.raw(_F32_BYTES * batch * _SQUARES), context.raw(_F32_BYTES * batch * _SQUARES),
    )
    egt_edge_list(context.builder, context.kernels, edge_list, edges, list_specialization)
    # H (post-door, pre-softmax) is exported only at the blocks a site follows; one buffer serves all three,
    # because each site reads it before the next exporting block overwrites it.
    logits = context.raw(_F32_BYTES * batch * shape.heads * _SQUARES * _SQUARES)
    # K2c "f16": the blocks read an FP16 copy of whichever state buffer is current; refreshed after every write.
    mode = _egt_state_mode(batch)
    copy_bytes = {"f16": _F16_BYTES, "f8": 1, "i8": 1}.get(mode, 0)  # r23b I8: one byte, like e4m3
    copy = context.raw(copy_bytes * egt.state_channels * cells) if copy_bytes and read_blocks else None  # r23b §7
    writes = 1 + len(sites)

    def copy_specialization(write: int) -> object:
        # Round 26 C3: the scale of THIS write (the seed is write 0, the k-th site write k + 1).
        scales = _egt_copy_scales(mode, egt.state_channels, write, writes)
        if mode == "f8":
            return CastStateF8Specialization(batch, context.architecture, scales, states=egt.state_channels)
        if mode == "i8":  # r23b I8
            return CastStateI8Specialization(batch, context.architecture, scales, states=egt.state_channels)
        return CastStateSpecialization(batch, context.architecture, states=egt.state_channels)

    write_copy = {"f8": cast_state_f8, "i8": cast_state_i8}.get(mode, cast_state)
    written = 0  # the write the copy currently holds
    if copy is not None:
        write_copy(context.builder, context.kernels, copy, state[0], copy_specialization(written))
    body, codes = _embedding(context, lab, planes)
    current = 0
    for index in range(shape.blocks):
        _LOGGER.info("batch size %d: building EGT2 encoder %d/%d", batch, index + 1, shape.blocks)
        body, codes = _encoder_egt(context, lab, body, edges, edge_norm, copy if copy is not None else state[current],
                                   edge_list, logits, index, export_h=index in sites, state=state[current],
                                   reads=index in read_blocks, body_codes=codes,  # r23b §7; Q1
                                   copy_scales=_egt_copy_scales(mode, egt.state_channels, written, writes))
        if index in sites:
            # `rev_edge` (BT6-test): the site's FFN also reads the reverse edge through `ffn/dense1_rev/w`.
            reverse = sites[index].ffn_rev_weight is not None
            site_tables = {name: context.weight(name) for name in site_table_names(index, reverse=reverse)}

            def site_specialization(stage: str) -> EdgeSiteSpecialization:
                # The readback stage runs no FFN, so it is the same kernel with or without the reverse edge.
                return EdgeSiteSpecialization(batch_count=batch, architecture=context.architecture, stage=stage,
                                              heads=shape.heads, states=egt.state_channels, hidden=egt.site_hidden,
                                              reverse=reverse and stage != "readback",
                                              dot=_SITE_DOT if stage != "readback" else "ieee")

            if sites[index].triplet is None:
                # K3: e' = rms(e_hat + W2 relu(W1 rms(e_hat) + b1)), e_hat = e + O_e H, into the other state buffer.
                edge_site(context.builder, context.kernels, state[1 - current], state[current], logits, site_tables,
                          site_specialization("site"), after_block=index)
                current = 1 - current
            else:
                # K5: K3's split form around K4's operator. readback writes e_hat to the other buffer, the triplet
                # adds its branch in place, ffn writes e' back into the buffer e came from (e is dead by then).
                # K4b: `LC0EX_TRIPLET_FORM` picks the staging (`_TRIPLET_FORMS`); the fused form has no scratch.
                triplet = sites[index].triplet
                fused = _TRIPLET_FORM == "fused"
                triplet_specialization = TripletSiteSpecialization(
                    batch_count=batch, architecture=context.architecture, contraction=triplet.contraction,
                    states=egt.state_channels, form="fused" if fused else "scratch", site_heads=shape.heads,
                    dot=_TRIPLET_DOT if fused else "ieee", state_f16=fused and _TRIPLET_STATE16,
                )
                triplet_tables = {name: context.weight(name) for name in triplet_table_names(index)}
                scratch = triplet_buffer_bytes(triplet_specialization)
                if _TRIPLET_FORM == "readback_prep":
                    # (a): K3's readback fused into K4's prep; e_hat goes to the other buffer as before.
                    triplet_site_with_readback(
                        context.builder, context.kernels, state[1 - current], state[current], logits,
                        context.raw(scratch["values"]), context.raw(scratch["gates"]), triplet_tables,
                        site_tables[readback_table_name(index)], triplet_specialization, after_block=index,
                    )
                else:
                    e_hat_copy = None
                    if scratch["copy"]:
                        # (b) with the FP16 e_hat read: K3's readback plus the copy, in one stage.
                        e_hat_copy = context.raw(scratch["copy"])
                        triplet_readback(context.builder, context.kernels, state[1 - current], e_hat_copy,
                                         state[current], logits, site_tables[readback_table_name(index)],
                                         triplet_specialization)
                    else:
                        edge_site(context.builder, context.kernels, state[1 - current], state[current], logits,
                                  site_tables, site_specialization("readback"), after_block=index)
                    # (b) "fused": `values` is the va buffer and there is no gates scratch.
                    va_buffer = context.raw(scratch["values"] or scratch["va"])
                    triplet_site(
                        context.builder, context.kernels, state[1 - current], va_buffer,
                        context.raw(scratch["gates"]) if scratch["gates"] else None,
                        triplet_tables, triplet_specialization, after_block=index, copy=e_hat_copy,
                        out=not (fused and _TRIPLET_OUT_FFN),
                    )
                    if fused and _TRIPLET_OUT_FFN:
                        if reverse:
                            message = ("LC0EX_TRIPLET_OUT_FFN=1 folds the triplet's `out` into an FFN stage that does "
                                       "not read the reverse edge; this net has `rev_edge` -- unset it")
                            raise NotImplementedError(message)
                        # (c): `out` folded into the FFN stage; e' goes back into the buffer e came from.
                        triplet_out_ffn(context.builder, context.kernels, state[current], state[1 - current],
                                        va_buffer, triplet_tables, site_tables, triplet_specialization,
                                        after_block=index)
                if not (fused and _TRIPLET_OUT_FFN):
                    edge_site(context.builder, context.kernels, state[current], state[1 - current], None,
                              site_tables, site_specialization("ffn"), after_block=index)
            written += 1
            if copy is not None:
                write_copy(context.builder, context.kernels, copy, state[current], copy_specialization(written))
    if lab.final_norm is not None and _FINAL_NORM_CONTROL != "skip":
        # O: the pre-norm tower is normed once, here, before every head (as `_network`).
        normed = context.temporary(context.rows * shape.d_model)
        _norm(context, normed, body, "/encoder/final_norm", shape.d_model, "none")
        body = normed
    _policy_head(context, lab, body, edges, edge_norm)
    _dense_head(context, body, "/value", hidden_width=128, square_width=128, output_name="/output/wdl",
                output_width=3, final_activation="none")
    _dense_head(context, body, "/moves", hidden_width=128, square_width=8, output_name="/output/mlh",
                output_width=1, final_activation="relu")


def _encoder_egt(  # noqa: PLR0913
    context: _Context, lab: LabNetwork, body: Buffer, edges: Buffer, edge_norm: Buffer, edge_state: Buffer,
    edge_list: EgtEdgeList, logits: Buffer, index: int, *, export_h: bool, state: Buffer, reads: bool = True,
    body_codes: Buffer | None = None, copy_scales: tuple[float, ...] | None = None,
) -> tuple[Buffer, Buffer | None]:
    """One EGT2 encoder block: packed QKV, `attention_egt`, then the static residual branch, LN1, GLU FFN, LN2.

    `edge_state` is what the head program reads in the "f32" and "f16" modes; `state` is the FP32 state the
    "tiles" mode builds its read tiles from. O (round 26b): a pre-norm export norms inside the branches (LN1 feeds
    the projections, LN2 the FFN, the stream is never normed here), and the attention output gate rides the packed
    projection as a fourth lane, `[q | k | v | g]`, applied by `attention_egt` before its store.
    """
    prefix = encoder_prefix(index)
    shape = lab.architecture
    assert lab.egt is not None  # noqa: S101 - `build` checked it
    edge = lab.egt.blocks[index]
    block = lab.blocks[index]
    rows, width = context.rows, shape.d_model
    prenorm = shape.block_style == "prenorm"
    gated = block.gate_weight is not None
    lanes = 4 if gated else 3
    int8 = index in context.int8_blocks
    source = body
    if prenorm:
        # O: LN1 feeds only the projections. Q1 (round 26b): in an int8 block its int8 copy is what they read.
        source = context.temporary(rows * width)
        body_codes = _norm(context, source, body, f"{prefix}/ln1", width, "none")
    # P3: one packed projection, [q | k | v] per row (O: [q | k | v | g]), read in place by attention_egt.
    qkv = context.temporary(rows * lanes * width)
    if int8 and body_codes is None:
        message = f"{prefix}: an int8 block without its producer's int8 copy (the attn_in norm did not quantise)"
        raise ValueError(message)
    stems = i8_site_names(prefix)
    if int8:
        # Q1: `attn_in` -- the codes the previous block's ln2 (the embedding's ln1 at block 0) wrote.
        cutlass_gemm_i8(
            context.builder, context.kernels, qkv, body_codes, context.weight(f"{stems['attn_in']}/w"),
            context.weight(f"{stems['attn_in']}/scale"), context.weight(f"{stems['attn_in']}/bias"),
            CutlassGemmI8Specialization(rows, lanes * width, width, context.architecture, epilogue="bias"),
        )
    else:
        _projection(context, qkv, source, f"{prefix}/mha/qkvg" if gated else f"{prefix}/mha/qkv", rows, lanes * width,
                    width, use_cutlass=_CUTLASS_QKV)
    # Q1: an int8 block's attention writes the out-projection's operand directly (`attn_out` codes, one byte each),
    # unless the conversion is priced as a separate pass (LC0EX_QUANT_ATTN_OUT=pass).
    fused_codes = int8 and _QUANT_ATTN_OUT == "epilogue"
    merged = context.raw(rows * width) if fused_codes else context.temporary(rows * width)
    attn_prescale = context.weight(i8_conversion_names(prefix)["attn_out"]) if fused_codes else None
    tables = {short: context.weight(name)
              for short, name in zip(BLOCK_TABLES, block_table_names(prefix), strict=True)}
    common = {
        "batch_count": context.batch_size * shape.heads, "heads": shape.heads, "head_dim": shape.head_dim,
        "architecture": context.architecture, "cap": edge.cap, "export_h": export_h, "capacity": _EGT_CAPACITY,
        "overflow_exact": _EGT_OVERFLOW == "branch", "quant_output": fused_codes,
        "output_gate": gated,
        "gate_scale": (float(_OGATE_SCALE_CONTROL) if _OGATE_SCALE_CONTROL else block.gate_scale) if gated else 2.0,
        "accumulate_f16": _EGT_ACC == "f16", "state_math_f16": _EGT_STATE_MATH == "f16", "list_f16": _EGT_LIST_F16,
        "arithmetic_auto": _EGT_ARITH_AUTO,
    }
    if not reads:
        # r23b §7: this block does not read the edge stream. q.k and the four E terms stay; the edge read, door and
        # gate are compiled out of the head program, and neither the state nor its copy is loaded.
        attention_egt(
            context.builder, context.kernels, merged, qkv, edge_list, edges, edge_norm, state, tables,
            AttentionEgtSpecialization(**common, logit_terms=_EGT_NO_READ_TERMS, state_f32=True),
            logits=logits if export_h else None, quant_prescale=attn_prescale,
        )
    elif _egt_state_mode(context.batch_size) == "tiles":
        # K2c: per round, the three FP16 read tiles of round_heads heads, then attention over those heads. Each
        # round's tile buffer is its own temporary; the reuse planner folds them.
        round_heads = shape.heads // _EGT_ROUNDS
        for round_index in range(_EGT_ROUNDS):
            head_base = round_index * round_heads
            tile_specialization = StateTilesSpecialization(
                context.batch_size, context.architecture, round_heads=round_heads, head_base=head_base,
                states=lab.egt.state_channels, reverse=_EGT_TILES_REVERSE,
            )
            tiles = context.raw(tiles_bytes(tile_specialization))
            egt_state_tiles(context.builder, context.kernels, tiles, state,
                            {name: tables[name] for name in TILE_TABLES}, tile_specialization)
            attention_egt(
                context.builder, context.kernels, merged, qkv, edge_list, edges, edge_norm, tiles, tables,
                AttentionEgtSpecialization(**common, state_f32=False, state_tiles=True, round_heads=round_heads,
                                           head_base=head_base),
                logits=logits if export_h else None, quant_prescale=attn_prescale,
            )
    else:
        mode = _egt_state_mode(context.batch_size)
        f8 = mode == "f8"  # r23 F8: the e4m3 copy, read back through its per-channel scale
        i8 = mode == "i8"  # r23b I8: the int8 copy, the same fold
        scales = (copy_scales if copy_scales is not None
                  else _egt_state_scales(lab.egt.state_channels) if f8
                  else _egt_state_scales_i8(lab.egt.state_channels) if i8 else ())
        attention_egt(
            context.builder, context.kernels, merged, qkv, edge_list, edges, edge_norm, edge_state, tables,
            AttentionEgtSpecialization(**common, state_f32=mode == "f32", state_f8=f8, state_i8=i8,
                                       state_scales=scales),
            logits=logits if export_h else None, quant_prescale=attn_prescale,
        )
    branch = context.temporary(rows * width)
    if int8:
        # Q1: `attn_out` -- the attention output's int8 codes (written by the attention kernel, or by a conversion
        # pass when that is being priced), then the out-projection with alpha and the skip in the epilogue.
        merged_codes = merged
        if not fused_codes:
            merged_codes = context.raw(rows * width)
            quantise_operand(
                context.builder, context.kernels, merged_codes, merged,
                context.weight(i8_conversion_names(prefix)["attn_out"]), None,
                QuantiseOperandSpecialization(rows, width, context.architecture),
            )
        cutlass_gemm_i8(
            context.builder, context.kernels, branch, merged_codes, context.weight(f"{stems['attn_out']}/w"),
            context.weight(f"{stems['attn_out']}/scale"), context.weight(f"{stems['attn_out']}/bias"),
            CutlassGemmI8Specialization(rows, width, width, context.architecture, epilogue="residual"),
            skip=body,
        )
    else:
        _skip_projection(context, branch, merged, f"{prefix}/mha/out", rows, width, width, skip=body,
                         alpha=context.weight(f"{prefix}/mha/alpha/w"), use_cutlass=_CUTLASS_OUTPROJ)
    if prenorm:
        # O: `branch` is the stream; LN2 feeds the GLU, FFN2 adds into the stream, nothing norms it here. Q1: in an
        # int8 block LN2's int8 copy is the GLU's operand.
        normed = context.temporary(rows * width)
        ffn_codes = _norm(context, normed, branch, f"{prefix}/ln2", width, "none")
        if int8 and ffn_codes is None:
            message = f"{prefix}: a pre-norm int8 block whose ln2 wrote no int8 copy for ffn_in"
            raise ValueError(message)
        return _gated_ffn_branch(context, prefix, normed, skip=branch, width=width, hidden=shape.ffn_hidden,
                                 codes=ffn_codes if int8 else None), None
    attended = context.temporary(rows * width)
    attended_codes = _norm(context, attended, branch, f"{prefix}/ln1", width, "none")
    if int8 and attended_codes is None:
        message = f"{prefix}: an int8 block whose ln1 wrote no int8 copy for ffn_in"
        raise ValueError(message)
    return _gated_ffn(context, prefix, f"{prefix}/ln2", attended, width, shape.ffn_hidden,
                      codes=attended_codes if int8 else None)


def _policy_head(context: _Context, lab: LabNetwork, body: Buffer, edges: Buffer, edge_norm: Buffer) -> None:
    # `width` is the policy head's own width from here on: the trunk enters through the embedding projection only.
    trunk = lab.architecture.d_model
    width = lab.architecture.policy_width or trunk
    batch, rows = context.batch_size, context.rows
    embedded = context.temporary(rows * width)
    _projection(context, embedded, body, "/policy/embedding", rows, width, trunk, use_cutlass=False,
                activation="mish")
    query, key = context.temporary(rows * width), context.temporary(rows * width)
    _projection(context, query, embedded, "/policy/Q", rows, width, width, use_cutlass=False)
    _projection(context, key, embedded, "/policy/K", rows, width, width, use_cutlass=False)
    records = context.temporary(batch * _POLICY_RECORD)
    # The export applies no scale to Q K^T (nodes 4160-4161), but the policy_qk
    # kernel always reads one: `/policy/qk_scale` is 1.0.
    batched_matmul(
        context.builder, context.kernels, records, query, key,
        BatchedMatmulSpecialization("policy_qk", batch, _SQUARES, _SQUARES, width, 1, context.architecture),
        scale=context.weight("/policy/qk_scale"),
    )
    if lab.egt is not None:
        # The EGT2 policy term is the same fold over 34 channels, read from K1's packed E.
        policy_egt_bias(
            context.builder, context.kernels, records, edges, edge_norm,
            context.weight("/policy/pair/scaled_coefficients"), context.weight("/policy/pair/constant_bias"),
            PolicyEgtBiasSpecialization(batch, context.architecture),
        )
    elif lab.pair is not None:
        policy_static_bias(
            context.builder, context.kernels, records, edges, edge_norm,
            context.weight("/policy/pair/scaled_coefficients"), context.weight("/policy/pair/constant_bias"),
            PolicyStaticBiasSpecialization(batch, context.architecture),
        )
    promotion_logits(
        context.builder, context.kernels, records, key, context.weight("/policy/promotion/w"),
        PromotionLogitsSpecialization(batch, width, context.architecture),
    )
    # The export's gather table equals lc0's policy map (checked 09-10), so the
    # built-in symbol serves it and `/policy/map/w` is not uploaded.
    mapping = context.builder.add_symbol(compile_symbol(architecture=f"sm_{context.architecture}"))
    output = context.builder.buffer(
        name="/output/policy", shape=(batch, 1858), dtype=lc0ex_pb2.Buffer.DATA_TYPE_F32, writable=True,
    )
    policy_map(context.builder, context.kernels, output, records, mapping,
               PolicyMapSpecialization(batch, context.architecture))


def _dense_head(  # noqa: PLR0913
    context: _Context, body: Buffer, prefix: str, *, hidden_width: int, square_width: int, output_name: str,
    output_width: int, final_activation: str,
) -> None:
    """Per-square projection with Mish, flatten, dense with Mish, dense output."""
    width = context.plans[f"{prefix}/embedding/w"].shape[0]
    batch, rows = context.batch_size, context.rows
    embedded = context.temporary(rows * square_width)
    _projection(context, embedded, body, f"{prefix}/embedding", rows, square_width, width, use_cutlass=False,
                activation="mish")
    hidden = context.temporary(batch * hidden_width)
    _projection(context, hidden, embedded, f"{prefix}/dense1", batch, hidden_width, _SQUARES * square_width,
                use_cutlass=False, activation="mish")
    output = context.builder.buffer(
        name=output_name, shape=(batch, output_width), dtype=lc0ex_pb2.Buffer.DATA_TYPE_F32, writable=True,
    )
    # Q1: the int8 artifact's WDL layer is the analyser's D1 refit (the FP16 artifact never carries it).
    weight_name, bias_name = (D1_NAMES if context.d1 and prefix == "/value"
                              else (f"{prefix}/dense2/w", f"{prefix}/dense2/b"))
    matmul(
        context.builder, context.kernels, output, hidden, context.weight(weight_name),
        MatmulSpecialization(batch, output_width, hidden_width, context.architecture, has_bias=True,
                             activation=final_activation, output_f32=True),
        bias=context.weight(bias_name),
    )
