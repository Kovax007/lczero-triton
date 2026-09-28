"""B3 (2026-09-28): Q1's int8 GEMM sites on the classic BT4 path -- the names, the weight layouts and the vector file.

The builder (`bt4/network.py`) and the carrier writer (`tools/make_carrier_bt4_q1.py`) both read this module, so the
buffers an int8 BT4 artifact declares and the initializers its carrier holds cannot drift apart.

What differs from the lab path (`lab/_names.py`, the EGT2 flagship)
-------------------------------------------------------------------
* The source is a classic `.pb.gz`. lc0ex never reads its `weights`: `UploadWeights` converts the net to ONNX in
  process (`MakeConverterOptions`: opset 17, FP16, the winner value head) and fills every buffer from an initializer of
  the same name. A BT4 carrier is therefore the classic net, byte for byte (the fingerprint reads its format and its
  weights' structure), plus an `onnx_model` holding the converter's initializers and the int8 sites' extra ones.
* The four sites per encoder, by the analyser's names (`REPORT_analyser_wk40_a3_int8_bt4_ffn_mid_hole_three_recipes_
  d1_0925.md` §1: attn_in = qkv, attn_out = out, ffn_in = ffn1, ffn_mid = ffn2). A site the vector file does not carry
  stays FP16 -- that is how R54 keeps `ffn_mid` FP16 in layers 7 and 9-13.
* FFN1 is Mish, not a GLU: its int8 epilogue is `cutlass_gemm_i8`'s `mish` (the FFN hidden written as `ffn_mid` codes
  when that site is int8, as FP16 when it is not).
* The residual sites fold the block's alpha into their scale and bias exactly as the FP16 epilogue applies it
  (`act(acc + b) * alpha + skip`, `cutlass_matmul`'s `Lc0ResidualOp`): scale = D * w[n] * alpha, bias = b * alpha.
* The producers: block L's `attn_in` codes come from block L-1's ln2 (the embedding's last norm, the one after its
  FFN, at L = 0); `ffn_in` from block L's ln1; `attn_out` from a `quantise_operand` pass after the attention kernel;
  `ffn_mid` from FFN1's epilogue (or a pass, when FFN1 itself stays FP16).
"""

from collections.abc import Mapping
from dataclasses import dataclass

from lczero_triton.lab._names import quant_d1_source, quant_gemm_mode, quant_prescale_source
from lczero_triton.lab._quant import SITES, QuantPrescale, prescale_for

# The converter's value-head output layer, and the names the int8 artifact reads its D1 refit under: the FP16 twin
# loads the same carrier and must keep the trained layer.
D1_TARGETS = {"W": "/value/dense2/matmul/w", "b": "/value/dense2/add/w"}
D1_SUFFIX = "/q1d1"


@dataclass(frozen=True, slots=True)
class SiteSource:
    """Where one site's FP16 weights live in the converter's ONNX: `[k, n]` blocks concatenated along n."""

    weights: tuple[str, ...]
    biases: tuple[str, ...]
    alpha: str  # "" or the block's one-element residual alpha initializer


def site_sources(prefix: str) -> dict[str, SiteSource]:
    """The converter initializers behind each site of one encoder (`prefix` = "/encoder<index>")."""
    return {
        "attn_in": SiteSource(
            (f"{prefix}/mha/Q/w/w", f"{prefix}/mha/K/w/w", f"{prefix}/mha/V/w/w"),
            (f"{prefix}/mha/Q/b/w", f"{prefix}/mha/K/b/w", f"{prefix}/mha/V/b/w"),
            "",
        ),
        "attn_out": SiteSource((f"{prefix}/mha/out/dense/w/w",), (f"{prefix}/mha/out/dense/b/w",),
                               f"{prefix}/alpha*input/w"),
        "ffn_in": SiteSource((f"{prefix}/ffn/dense1/w/w",), (f"{prefix}/ffn/dense1/b/w",), ""),
        "ffn_mid": SiteSource((f"{prefix}/ffn/dense2/w/w",), (f"{prefix}/ffn/dense2/b/w",), f"{prefix}/ffn/alpha/w"),
    }


def site_names(prefix: str, site: str) -> dict[str, str]:
    """The four buffers of one int8 site: `w` (int8 `[n, k]`), `scale` and `bias` (FP32 `[n]`), and `r` (FP32 `[k]`,
    the producer's conversion vector)."""
    stem = f"{prefix}/q1/{site}"
    return {"w": f"{stem}/w", "scale": f"{stem}/scale", "bias": f"{stem}/bias", "r": f"{stem}/r"}


def site_widths(d_model: int, hidden: int) -> dict[str, int]:
    """Each site's input channel count: what its vectors must be as long as."""
    return {"attn_in": d_model, "attn_out": d_model, "ffn_in": d_model, "ffn_mid": hidden}


def load_vectors(*, blocks: int, d_model: int, hidden: int) -> QuantPrescale | None:
    """The analyser's file when this build serves int8 (LC0EX_QUANT_GEMM=int8), validated against BT4's widths."""
    if not quant_gemm_mode():
        return None
    return prescale_for(quant_prescale_source(), blocks=blocks, widths=site_widths(d_model, hidden))


def served_sites(vectors: QuantPrescale | None, index: int) -> frozenset[str]:
    """The sites of encoder `index` this build serves int8: those the file carries WITH `s` (the weight fold needs it)."""
    if vectors is None:
        return frozenset()
    scope = f"encoder{index}"
    served = set()
    for site in SITES:
        vector = vectors.get(scope, site)
        if vector is None:
            continue
        if vector.smoothing is None:
            message = f"{vectors.path.name}: {scope}/{site} has no 's', so its weights cannot be folded"
            raise ValueError(message)
        served.add(site)
    return frozenset(served)


def d1_enabled() -> bool:
    """Whether this int8 build reads the value layer under the D1 names (LC0EX_QUANT_D1 set)."""
    return bool(quant_d1_source())


def d1_name(name: str) -> str:
    """The buffer an int8 artifact with D1 reads in place of one of `D1_TARGETS`."""
    return name + D1_SUFFIX if d1_enabled() else name


def check_blocks(vectors: QuantPrescale | None, blocks: int) -> Mapping[int, frozenset[str]]:
    """Every block's served sites, and a refusal if the file names a scope this net does not have."""
    if vectors is None:
        return {index: frozenset() for index in range(blocks)}
    for scope, _site in vectors.sites:
        if scope == "embedding" or int(scope[7:]) >= blocks:
            message = f"{vectors.path.name}: {scope} is not an encoder of this {blocks}-block net"
            raise ValueError(message)
    return {index: served_sites(vectors, index) for index in range(blocks)}
