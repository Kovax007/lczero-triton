"""Q1: the per-site quantiser vectors an artifact carries, and where they come from.

The backend does not choose these numbers. The analyser measures them on the net being served
(`TASK_analyser_smoothquant_ptq_read_512_then_bt6test_0921.md`) and delivers one `.npz`; this module is
the only place that reads it, so the two lanes cannot drift on what a key means.

What a vector is
----------------
Per quantised site, over its INPUT channel axis:

    s_j = amax_j(x)^a / max_k |W[k, j]|^(1 - a)     the SmoothQuant migration, `a` the analyser's choice
    D   = max_j (amax_j / s_j) / 127                the ONE activation step the integer GEMM sees
    r_j = 1 / (s_j * D)                             what the artifact carries
    m_j                                             optional: the calibration mean, if the offset folds

and the conversion is `q_j = clamp(floor((x_j - m_j) * r_j + 0.5), -127, 127)`, with `W[:, j] * s_j`
quantised per output channel offline and `W . m` added to the GEMM bias offline. Only `r` and `m` are
served; `s`, `D` and `a` are the analyser's working, kept in the file for provenance and checked for
consistency when they are present.

⛔ `r` is NOT `1 / D`. A scalar step is the special case `s = 1`, which is what an unsmoothed site gets,
and what a PRE-norm net gets once the fold moves into the norm -- `identity_prescale` writes exactly that.
The flagship is post-norm, so the general case is the one that is built.

The file
--------
A `.npz` of 1-D float arrays, one per key:

    <site>/r     required, positive, finite, one entry per input channel
    <site>/m     optional, finite, the same length
    <site>/s     optional, provenance; if present, `r * s` must be constant over j to 1e-4 (that constant
                 is 1 / D, and a mismatch means the two were computed against different steps)
    <site>/amax  optional, provenance only

`<site>` is `encoder<index>/<name>` with `<name>` one of the four quantised sites, or `embedding/<name>`.
Anything else is refused: an unread key is a vector that was measured and then silently not served.
`producing_norm` maps a key to the norm whose artifact buffers (`_names.quant_names`) carry it.

This module is numpy-free on purpose -- no venv on the serving fleet carries numpy, and `lab/_onnx.py`
already decodes raw tensors by hand. The reader takes `.npy` v1/v2 members of a zip, little-endian
float16 / float32 / float64, C order, one dimension.
"""

import ast
import hashlib
import math
import struct
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from lczero_triton.lab._names import EMBEDDING_PREFIX, encoder_prefix, quant_names

# One process plans and then fills; both read the file through here, so a file that passed the planner's checks
# is the file the carrier writes.
_LOADED: dict[str, "QuantPrescale"] = {}

INT8_MAX = 127.0

# The four sites of SPEC v2 §3, by the name the analyser prints, and the axis each one's vector lives on.
SITE_FFN_IN = "ffn_in"
SITE_ATTN_IN = "attn_in"
SITE_FFN_MID = "ffn_mid"
SITE_ATTN_OUT = "attn_out"
SITES = (SITE_FFN_IN, SITE_ATTN_IN, SITE_FFN_MID, SITE_ATTN_OUT)
# Which sites a producer can absorb the vector into for free, and which wait for Q1's epilogue.
FOLDS_INTO_NORM = (SITE_FFN_IN, SITE_ATTN_IN)
FOLDS_INTO_EPILOGUE = (SITE_FFN_MID, SITE_ATTN_OUT)

_SCALARS = {"<f2": (2, "e"), "<f4": (4, "f"), "<f8": (8, "d")}
_MAGIC = b"\x93NUMPY"
_HEADER_LENGTH = {1: 2, 2: 4}
# `r * s` is 1 / D at every channel; the analyser rounds both to float32, so the agreement is checked at
# the precision float32 can carry over a 1024-wide vector, not at an exact equality.
_STEP_TOLERANCE = 1e-4


class QuantFormatError(ValueError):
    """The prescale file is not one this artifact can serve."""


def _read_npy(payload: bytes, where: str) -> tuple[float, ...]:
    """Decode one 1-D little-endian float `.npy` member."""
    if not payload.startswith(_MAGIC):
        message = f"{where}: not a .npy member"
        raise QuantFormatError(message)
    major = payload[6]
    if major not in _HEADER_LENGTH:
        message = f"{where}: .npy version {major} is not supported"
        raise QuantFormatError(message)
    width = _HEADER_LENGTH[major]
    start = 8 + width
    length = int.from_bytes(payload[8:start], "little")
    try:
        header = ast.literal_eval(payload[start:start + length].decode("latin-1").strip())
    except (SyntaxError, ValueError) as error:
        message = f"{where}: unreadable .npy header"
        raise QuantFormatError(message) from error
    if not isinstance(header, dict):
        message = f"{where}: unreadable .npy header"
        raise QuantFormatError(message)
    descr, order, shape = header.get("descr"), header.get("fortran_order"), header.get("shape")
    if descr not in _SCALARS:
        message = f"{where}: dtype {descr!r}; the reader takes little-endian float16/32/64 only"
        raise QuantFormatError(message)
    if order:
        message = f"{where}: Fortran order"
        raise QuantFormatError(message)
    if not isinstance(shape, tuple) or len(shape) != 1:
        message = f"{where}: shape {shape!r}; a quantiser vector is one-dimensional"
        raise QuantFormatError(message)
    size, code = _SCALARS[descr]
    data = payload[start + length:]
    count = int(shape[0])
    if len(data) < count * size:
        message = f"{where}: {len(data)} bytes for {count} values"
        raise QuantFormatError(message)
    return struct.unpack(f"<{count}{code}", data[:count * size])


def _parse_key(key: str, blocks: int) -> tuple[str, str]:
    """Split a member name into `(scope, site)`, refusing anything the builder would not serve."""
    stem = key[:-4] if key.endswith(".npy") else key
    parts = stem.split("/")
    expected = 3
    if len(parts) != expected or parts[2] not in {"r", "m", "s", "amax"}:
        message = f"{key!r} is not <scope>/<site>/<r|m|s|amax>"
        raise QuantFormatError(message)
    scope, site, _ = parts
    if site not in SITES:
        message = f"{key!r}: site {site!r} is not one of {SITES}"
        raise QuantFormatError(message)
    if scope != "embedding":
        if not scope.startswith("encoder") or not scope[7:].isdigit():
            message = f"{key!r}: scope {scope!r} is not 'embedding' or 'encoder<index>'"
            raise QuantFormatError(message)
        if int(scope[7:]) >= blocks:
            message = f"{key!r}: block {scope[7:]} is past this network's {blocks} blocks"
            raise QuantFormatError(message)
    return scope, site


@dataclass(frozen=True, slots=True)
class SitePrescale:
    """One site's served vectors: `r`, and `m` when the offset folds."""

    scope: str
    site: str
    prescale: tuple[float, ...]
    offset: tuple[float, ...] | None = None
    # `s`, when the file carries it: Q1's integer GEMM folds it into the weight rows (`W'[j, :] = s_j W[j, :]`),
    # so a site without it can be converted (the norm's copy) but not served by the GEMM.
    smoothing: tuple[float, ...] | None = None

    @property
    def channels(self) -> int:
        """The input channel count this vector covers."""
        return len(self.prescale)

    @property
    def step(self) -> float:
        """`D`, the one activation step the integer GEMM sees: `1 / (r_j s_j)`, constant over j (checked at load)."""
        if self.smoothing is None:
            message = f"{self.scope}/{self.site}: no 's' in the file, so D (and the weight fold) is unknown"
            raise QuantFormatError(message)
        products = [r * s for r, s in zip(self.prescale, self.smoothing, strict=True)]
        return len(products) / math.fsum(products)


@dataclass(frozen=True, slots=True)
class QuantPrescale:
    """Everything one net's quantiser needs, keyed `(scope, site)`."""

    sites: Mapping[tuple[str, str], SitePrescale]
    digest: str
    path: Path

    def get(self, scope: str, site: str) -> SitePrescale | None:
        """The vectors for one site, or None if this file does not quantise it."""
        return self.sites.get((scope, site))

    def scopes(self, site: str) -> tuple[str, ...]:
        """Every scope that carries `site`, in file order."""
        return tuple(scope for scope, name in self.sites if name == site)

    def __len__(self) -> int:
        """How many sites this file quantises."""
        return len(self.sites)


def _members(archive: zipfile.ZipFile, blocks: int) -> Iterator[tuple[tuple[str, str], str, tuple[float, ...]]]:
    for name in archive.namelist():
        scope, site = _parse_key(name, blocks)
        kind = (name[:-4] if name.endswith(".npy") else name).split("/")[2]
        yield (scope, site), kind, _read_npy(archive.read(name), name)


def load_prescale(path: Path, *, blocks: int, widths: Mapping[str, int] | None = None) -> QuantPrescale:
    """Read the analyser's `.npz`, refusing anything this artifact would have to guess about.

    `widths` maps a site name to the channel count the architecture says it has; a vector of another
    length is refused rather than broadcast, because a wrong-width vector quantises a real net into
    plausible-looking nonsense.
    """
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:8]
    collected: dict[tuple[str, str], dict[str, tuple[float, ...]]] = {}
    with zipfile.ZipFile(path) as archive:
        for key, kind, values in _members(archive, blocks):
            collected.setdefault(key, {})[kind] = values
    sites: dict[tuple[str, str], SitePrescale] = {}
    for (scope, site), parts in collected.items():
        where = f"{path.name} {scope}/{site}"
        prescale = parts.get("r")
        if prescale is None:
            message = f"{where}: carries {sorted(parts)} but no 'r'"
            raise QuantFormatError(message)
        if any(not math.isfinite(value) or value <= 0.0 for value in prescale):
            message = f"{where}: 'r' must be positive and finite at every channel"
            raise QuantFormatError(message)
        if widths is not None and site in widths and len(prescale) != widths[site]:
            message = f"{where}: 'r' has {len(prescale)} channels; the network's {site} has {widths[site]}"
            raise QuantFormatError(message)
        offset = parts.get("m")
        if offset is not None and len(offset) != len(prescale):
            message = f"{where}: 'm' has {len(offset)} channels against 'r'-s {len(prescale)}"
            raise QuantFormatError(message)
        smoothing = parts.get("s")
        if smoothing is not None:
            _check_one_step(prescale, smoothing, where)
        sites[(scope, site)] = SitePrescale(scope, site, tuple(prescale), tuple(offset) if offset else None,
                                            tuple(smoothing) if smoothing else None)
    return QuantPrescale(sites, digest, path)


def _check_one_step(prescale: Sequence[float], smoothing: Sequence[float], where: str) -> None:
    """`r_j * s_j` is `1 / D` at every channel, or the two were built against different steps."""
    if len(smoothing) != len(prescale):
        message = f"{where}: 's' has {len(smoothing)} channels against 'r'-s {len(prescale)}"
        raise QuantFormatError(message)
    products = [r * s for r, s in zip(prescale, smoothing, strict=True)]
    low, high = min(products), max(products)
    if low <= 0.0 or (high - low) > _STEP_TOLERANCE * high:
        message = (f"{where}: r * s ranges over [{low:.6g}, {high:.6g}]; it is 1 / D and must be one "
                   "constant, so 'r' and 's' were not built against the same activation step")
        raise QuantFormatError(message)


def prescale_for(path: str | Path, *, blocks: int, widths: Mapping[str, int] | None = None) -> "QuantPrescale":
    """Load and validate one file once per process, and remember it for the carrier."""
    key = str(path)
    if key not in _LOADED:
        _LOADED[key] = load_prescale(Path(path), blocks=blocks, widths=widths)
    return _LOADED[key]


def loaded_prescale() -> "QuantPrescale | None":
    """The file this process validated while planning, if any."""
    return next(iter(_LOADED.values()), None)


def producing_norm(scope: str, site: str, *, style: str = "postnorm") -> tuple[str, str] | None:
    """Which norm emits this site's int8 copy, as `(buffer prefix, norm)` -- or None for a GEMM site.

    ⚠ The off-by-one that would silently ruin a net: under post-norm a block's ATTENTION input is the *previous*
    block's `ln2` output (the embedding's `ln1` at block 0), while its FFN input is its own `ln1`. A vector
    folded into the wrong norm still loads, still runs and still returns moves. Under pre-norm (round 26b) there
    is no shift: a block's own `ln1` feeds its attention and its own `ln2` its FFN (`_names.quantised_norms`).
    """
    if site in FOLDS_INTO_EPILOGUE:
        return None
    if style == "prenorm":
        if scope == "embedding":
            message = f"embedding/{site}: a pre-norm net's embedding feeds no quantised GEMM"
            raise QuantFormatError(message)
        return (encoder_prefix(int(scope[7:])), "ln1" if site == SITE_ATTN_IN else "ln2")
    if site == SITE_FFN_IN:
        return (EMBEDDING_PREFIX if scope == "embedding" else encoder_prefix(int(scope[7:])), "ln1")
    if scope == "embedding":
        message = "embedding/attn_in: the embedding has no attention to feed"
        raise QuantFormatError(message)
    index = int(scope[7:])
    return ((EMBEDDING_PREFIX, "ln1") if index == 0 else (encoder_prefix(index - 1), "ln2"))


def buffer_names(scope: str, site: str) -> tuple[str, str] | None:
    """The artifact buffers that carry this site's `r` and `m`, or None for a GEMM-epilogue site."""
    norm = producing_norm(scope, site)
    return None if norm is None else quant_names(*norm)


def smoothing_vector(activation_amax: Sequence[float], weight_amax: Sequence[float], alpha: float) -> tuple[float, ...]:
    """SmoothQuant's migration vector `s_j = amax_j^a / wmax_j^(1 - a)` (the analyser's formula, here to be shared)."""
    return tuple(a ** alpha / w ** (1.0 - alpha) for a, w in zip(activation_amax, weight_amax, strict=True))


def activation_step(activation_amax: Sequence[float], smoothing: Sequence[float]) -> float:
    """`D = max_j (amax_j / s_j) / 127`: the one step the integer GEMM sees after the migration."""
    return max(a / s for a, s in zip(activation_amax, smoothing, strict=True)) / INT8_MAX


def prescale(smoothing: Sequence[float], step: float) -> tuple[float, ...]:
    """`r_j = 1 / (s_j * D)`, what the artifact carries."""
    return tuple(1.0 / (s * step) for s in smoothing)


def identity_prescale(channels: int, step: float) -> tuple[float, ...]:
    """The unsmoothed vector `r_j = 1 / D`: a per-tensor scale in the same shape, for a site (or a
    pre-norm net) that needs no migration. The kernel is unchanged; only the numbers are flat."""
    return (1.0 / step,) * channels


def fold_into_affine(
    gammas: Sequence[float],
    betas: Sequence[float],
    site: SitePrescale,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """`gamma * r` and `(beta - m) * r`: the affine a norm applies when it absorbs the vector itself.

    ⛔ This is NOT what the served post-norm kernel does. There the norm keeps `gamma` and `beta` for the FP16
    stream and converts its own FP32 output through `r` and `m` directly, which keeps that stream bit-identical
    to the unquantised kernel's at an equal warp count -- `fold_into_affine` would not, because it perturbs the
    shared `normalized`. The pair is for the PRE-norm case, where the norm's output feeds only its GEMM and the
    vector belongs in the norm's own weights, and for anyone folding the vector offline. The two forms agree to
    a rounding in the last place; `test_quantise_operand.py` measures the code-flip rate on the served shape.
    """
    offset = site.offset or (0.0,) * site.channels
    if not len(gammas) == len(betas) == site.channels:
        message = (f"{site.scope}/{site.site}: the affine pair is {len(gammas)}/{len(betas)} wide against "
                   f"the vector's {site.channels}")
        raise QuantFormatError(message)
    return (
        tuple(gamma * r for gamma, r in zip(gammas, site.prescale, strict=True)),
        tuple((beta - m) * r for beta, m, r in zip(betas, offset, site.prescale, strict=True)),
    )
