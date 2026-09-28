"""CUDA tests for EGT2 attention (item E, K2), gated per block against onnxruntime.

The reference set ``ref/egt_k2`` (``k2_ref.json``, SHA256 manifest verified once per module) holds 64 of K1's
positions, category-balanced, with every one of the 34 edge channels present. For gcap blocks 0, 3 (e0, capped), 4
(the first site-fed e) and 14 (the last), and triplet_path block 3 (uncapped gate), ORT (CPU, optimisations off)
dumped the kernel's inputs (the packed q|k|v Gemm rows, S, e, E as K1's uint64) and outputs (H, A, and A.v before the
out-projection). The fold tables are R0's: read from the gcap carrier, and planned from the triplet_path export with
R0's reader (there is no triplet_path carrier).

Gates (``K2-GATE`` lines, ``pytest -s``):
* FP32 inputs: H and A in the FP32 class (peak-relative, the map's 1e-5 rule), output in the FP16 class;
* served precision (FP16 q|k|v and S): H, A and output in the class of FP16 projections;
* the cap: on for gcap, off for triplet_path, and switching it off on gcap must break A;
* each logit term against a float64 reference built from the same carrier tables;
* the edge list exactly equal to E's set bits, and `counts` exact even when the list truncates;
* the dense fallback: forced small capacities, a mixed batch, the capacity - 1 / capacity boundary,
  and the fast path bit-identical to the kernel as it stood before the fallback;
* the H export layout against K3's ORT logits (``ref/egt_k3``) on the positions both sets share.
"""

import hashlib
import io
import json
import os
from pathlib import Path

import pytest
import torch
from lc0ex import ExecutableBuilder
from lc0ex.proto import lc0ex_pb2
from lczero_triton.bt4.kernels._cache import KernelCache
from lczero_triton.bt4.kernels.attention_egt import (
    BLOCK_TABLES,
    _attention_egt_kernel,
    CELLS,
    EGT_ATTACK,
    EGT_DOOR,
    EGT_EDGE_READ,
    EGT_FULL,
    EGT_GATE,
    EGT_KEY,
    EGT_QK,
    EGT_QUERY,
    EGT_SCALED,
    AttentionEgtSpecialization,
    EgtEdgeList,
    EgtEdgeListSpecialization,
    attention_egt,
    block_table_names,
    buffer_bytes,
    compile_attention_egt,
    compile_egt_edge_count,
    compile_egt_edge_list,
    compile_egt_edge_offset,
    edge_list_bytes,
    egt_edge_list,
    launch_attention_egt,
    launch_egt_edge_list,
    term_flags,
)
from lczero_triton.bt4.kernels.prologue_egt import unpack_edges

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]

_TREE = Path(__file__).resolve().parents[3]
_REFERENCE = Path(os.environ.get("LC0EX_EGT_K2_REF", str(_TREE / "ref/egt_k2")))
_K3_REFERENCE = Path(os.environ.get("LC0EX_EGT_K3_REF", str(_TREE / "ref/egt_k3")))
_CARRIER = Path(os.environ.get(
    "LC0EX_EGT_CARRIER", str(Path.home() / "spsa/lc0ex_5080/work/r20c_itemE_R0/egt2_gcap_512x15_lc0ex_r0.pb.gz")))
_GOLDEN = Path(os.environ.get("LC0EX_EGT_K2B_GOLDEN", str(_TREE / "ref/egt_k2b/golden_k2.json")))
_NETS = Path(os.environ.get("LC0EX_EGT_NETS", str(Path.home() / "spsa/lc0ex_5080/work/nets_r20")))
_TRIPLET_PATH = "egt2_triplet_path_512x15_50000_vw.pb.gz"
_COUNT, _HEADS, _DEPTH = 64, 32, 16
_CAPACITY = 1024
_CHUNK = 8
# FP32 class: the map's single-block pass rule, peak-relative. FP16 output storage: half a unit in the last place.
_F32_RELATIVE = 1e-5
_F16_STORAGE_RELATIVE = 1e-3
# Served precision (FP16 q|k|v and S): the class of FP16 projections (the map's probe: 2.2e-3 on H, whole tower).
_SERVED_RELATIVE = 5e-3
_BLOCKS = [("gcap", 0), ("gcap", 3), ("gcap", 4), ("gcap", 14), ("triplet_path", 3)]
_LOADED: dict = {}


def _architecture() -> int:
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def _read(relative: str, dtype: torch.dtype, *shape: int) -> torch.Tensor:
    return torch.frombuffer(bytearray((_REFERENCE / relative).read_bytes()), dtype=dtype).view(*shape)


@pytest.fixture(scope="module")
def reference() -> dict:
    header_path = _REFERENCE / "k2_ref.json"
    if not header_path.exists():
        pytest.skip(f"reference not found at {_REFERENCE}")
    for line in (_REFERENCE / "SHA256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((_REFERENCE / name).read_bytes()).hexdigest() == digest, f"{name} fails its sha256"
    header = json.loads(header_path.read_text())
    assert header["count"] == _COUNT
    edges = _read("edges.bin", torch.int64, _COUNT, 64, 64).view(torch.uint64).cuda()
    return {"header": header, "edges": edges}


def _tables(label: str, block: int) -> dict[str, torch.Tensor]:
    """The block's R0 FP32 plans, keyed by BLOCK_TABLES."""
    key = (label, block)
    if key in _LOADED:
        return _LOADED[key]
    names = block_table_names(f"/encoder{block}")
    if label == "gcap":
        if not _CARRIER.exists():
            pytest.skip(f"carrier not found at {_CARRIER}")
        from lczero_triton.lab._onnx import load_carrier  # noqa: PLC0415

        if "gcap_graph" not in _LOADED:
            _LOADED["gcap_graph"] = load_carrier(_CARRIER)[1]
        graph = _LOADED["gcap_graph"]
        raw = {name: (graph.initializers[name].raw_data, graph.initializers[name].dims,
                      graph.initializers[name].data_type) for name in names}
    else:
        path = _NETS / _TRIPLET_PATH
        if not path.exists():
            pytest.skip(f"export not found at {path}")
        from lczero_triton.lab import _onnx  # noqa: PLC0415
        from lczero_triton.lab._mapping import read_network  # noqa: PLC0415
        from lczero_triton.lab._names import plan_network  # noqa: PLC0415
        from lczero_triton.lab.carrier import _build_payload  # noqa: PLC0415

        if "triplet_plans" not in _LOADED:
            _, graph = _onnx.load_carrier(path)
            _LOADED["triplet_plans"] = (graph, {plan.name: plan for plan in plan_network(read_network(graph))})
        graph, plans = _LOADED["triplet_plans"]
        raw = {name: (_build_payload(graph, plans[name])[0], plans[name].shape, plans[name].data_type)
               for name in names}
    tables = {}
    for short, name in zip(BLOCK_TABLES, names, strict=True):
        payload, dims, data_type = raw[name]
        assert data_type == 1, f"{name} is not FP32"  # noqa: PLR2004
        tables[short] = torch.frombuffer(bytearray(payload), dtype=torch.float32).view(*dims).cuda()
    _LOADED[key] = tables
    return tables


def _block_inputs(reference: dict, label: str, block: int, *, served: bool) -> dict[str, torch.Tensor]:
    meta = reference["header"]["arms"][label]["blocks"][str(block)]
    qkv = _read(f"{label}/block{block}_qkv.bin", torch.float32, _COUNT, 64, 1536).cuda()
    norm = _read(f"{label}/S.bin", torch.float32, _COUNT, 64, 64).cuda()
    return {
        "qkv": qkv.half() if served else qkv,
        "S": norm.half() if served else norm,
        "state": _read(f"{label}/{meta['state_file']}", torch.float32, _COUNT, 16, 64, 64).cuda(),
        "H": _read(f"{label}/block{block}_H.bin", torch.float32, _COUNT, 32, 64, 64).cuda(),
        "A": _read(f"{label}/block{block}_A.bin", torch.float32, _COUNT, 32, 64, 64).cuda(),
        "attended": _read(f"{label}/block{block}_attended.bin", torch.float32, _COUNT, 32, 64, 16).cuda(),
        "cap": meta["cap"],
    }


def _edge_list(reference: dict, capacity: int = _CAPACITY) -> tuple[torch.Tensor, ...]:
    batch = reference["edges"].shape[0]
    buffers = (
        torch.full((batch, capacity), -1, dtype=torch.int16, device="cuda"),
        torch.full((batch, capacity), -1, dtype=torch.int8, device="cuda"),
        torch.full((batch, 2, CELLS), -1, dtype=torch.int16, device="cuda"),
        torch.full((batch,), -1, dtype=torch.int32, device="cuda"),
        torch.empty((batch, 64), dtype=torch.int32, device="cuda"),
        torch.empty((batch, 64), dtype=torch.int32, device="cuda"),
    )
    launch_egt_edge_list(*buffers, reference["edges"], capacity)
    torch.cuda.synchronize()
    return buffers


def _run(reference: dict, inputs: dict, tables: dict, *, terms: int = EGT_FULL, cap: bool | None = None,
         edge_list: tuple | None = None, capacity: int = _CAPACITY, warps: int | None = None) -> dict[str, torch.Tensor]:
    edge_list = edge_list or _edge_list(reference, capacity)
    count = inputs["qkv"].shape[0]  # the boundary test runs a two-position batch
    batch_count = count * _HEADS
    specialization = AttentionEgtSpecialization(
        batch_count=batch_count, heads=_HEADS, head_dim=_DEPTH, architecture=_architecture(),
        cap=inputs["cap"] if cap is None else cap, export_h=True, logit_terms=terms, capacity=capacity,
        norm_f32=inputs["S"].dtype == torch.float32, export_weights=True,
    )
    output = torch.full((count, 64, 512), -7.0, dtype=torch.float16, device="cuda")
    logits = torch.full((batch_count, 64, 64), -7.0, dtype=torch.float32, device="cuda")
    weights = torch.full((batch_count, 64, 64), -7.0, dtype=torch.float32, device="cuda")
    if warps is None:
        launch_attention_egt(output, inputs["qkv"], edge_list[:4], reference["edges"], inputs["S"], inputs["state"],
                             tables, specialization, logits=logits, weights=weights)
    else:  # the bit-identity gate pins the configuration, so the autotuner cannot change the arithmetic
        flags = term_flags(specialization.logit_terms)
        _attention_egt_kernel.fn[(specialization.batch_count,)](
            output, logits, weights, inputs["qkv"], *edge_list[:4], reference["edges"], inputs["S"], inputs["state"],
            *(tables[name] for name in BLOCK_TABLES), tables["qk_scale"], specialization.batch_count, _HEADS,
            _DEPTH, capacity,
            specialization.cap, True, True, *flags.values(), False, _HEADS, 0, False, False, True, False, num_warps=warps)
    torch.cuda.synchronize()
    return {"H": logits.view(count, 32, 64, 64), "A": weights.view(count, 32, 64, 64),
            "attended": output.view(count, 64, 32, 16).permute(0, 2, 1, 3)}


def _error(got: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = (got.double() - expected.double()).abs()
    return {"max_abs": float(difference.max()), "rel": float(difference.max() / expected.double().abs().max())}


def _digest(tensor: "torch.Tensor") -> str:
    """A byte digest without numpy (the tree venv has none): torch's own serialization of the CPU tensor."""
    buffer = io.BytesIO()
    torch.save(tensor.contiguous().cpu(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()[:16]


def _line(label: str, errors: dict) -> str:
    return f"K2-GATE {label}: " + "  ".join(f"{k} abs {v['max_abs']:.2e} rel {v['rel']:.2e}" for k, v in errors.items())


def test_edge_list_is_exactly_the_set_bits(reference: dict) -> None:
    cells, channels, prefix, counts, _, _ = _edge_list(reference)
    E = unpack_edges(reference["edges"])
    expected = E.flatten(1).sum(1)
    assert torch.equal(counts.long(), expected), "counts"
    assert int(counts.max()) < _CAPACITY - 1, "the reference set must fit the capacity"
    per_cell = E.permute(0, 2, 3, 1).reshape(_COUNT, CELLS, 34)
    for z in range(_COUNT):
        n = int(counts[z])
        listed = per_cell[z].nonzero()  # (cell, channel), cell order then channel order
        assert torch.equal(cells[z, :n].long(), listed[:, 0]) and torch.equal(channels[z, :n].long(), listed[:, 1]), z
        population = per_cell[z].sum(1)
        end = population.cumsum(0)
        assert torch.equal(prefix[z, 1].long(), end) and torch.equal(prefix[z, 0].long(), end - population), z
    print(f"K2-GATE edge list: exact on {_COUNT} positions; set bits max {int(counts.max())} "
          f"median {int(counts.median())}", flush=True)


def test_counts_are_never_clamped(reference: dict) -> None:
    """A small list truncates, but `counts` still reports the true number of set bits: the kernel branches on it."""
    capacity = 256
    _, _, prefix, counts, _, _ = _edge_list(reference, capacity)
    expected = unpack_edges(reference["edges"]).flatten(1).sum(1)
    assert torch.equal(counts.long(), expected)
    assert int(prefix.max()) == capacity - 1, "spans clamp at capacity - 1"
    print(f"K2-GATE counts exact at capacity {capacity}: {int((counts > capacity - 1).sum())} of {_COUNT} positions "
          f"overflow", flush=True)


@pytest.mark.parametrize("capacity", [64, 128, 256])
def test_overflow_fallback_matches_ort(reference: dict, capacity: int) -> None:
    """Past capacity - 1 set bits a position takes the dense fallback; it must stay in the FP32 class."""
    inputs = _block_inputs(reference, "gcap", 3, served=False)
    edge_list = _edge_list(reference, capacity)
    overflow = edge_list[3] > capacity - 1
    assert bool(overflow.any()), f"capacity {capacity} should overflow on this reference set"
    got = _run(reference, inputs, _tables("gcap", 3), edge_list=edge_list, capacity=capacity)
    errors = {name: _error(got[name], inputs[name]) for name in ("H", "A")}
    print(_line(f"capacity {capacity} ({int(overflow.sum())} of {_COUNT} dense) gcap block 3", errors), flush=True)
    assert errors["H"]["rel"] <= _F32_RELATIVE and errors["A"]["rel"] <= _F32_RELATIVE


def test_mixed_batch_takes_both_paths(reference: dict) -> None:
    """At capacity 256 the batch holds both kinds: the dense ones match ORT, the others their capacity-1024 values."""
    capacity = 256
    inputs = _block_inputs(reference, "gcap", 3, served=False)
    tables = _tables("gcap", 3)
    edge_list = _edge_list(reference, capacity)
    overflow = edge_list[3] > capacity - 1
    fast = ~overflow
    assert bool(overflow.any()) and bool(fast.any()), "capacity 256 should mix both paths"
    mixed = _run(reference, inputs, tables, edge_list=edge_list, capacity=capacity)
    base = _run(reference, inputs, tables)
    for name in ("H", "A", "attended"):
        assert torch.equal(mixed[name][fast], base[name][fast]), f"{name}: a fast position changed"
    errors = {name: _error(mixed[name][overflow], inputs[name][overflow]) for name in ("H", "A")}
    print(_line(f"mixed batch ({int(overflow.sum())} dense, {int(fast.sum())} list) gcap block 3", errors), flush=True)
    assert errors["H"]["rel"] <= _F32_RELATIVE and errors["A"]["rel"] <= _F32_RELATIVE


def test_the_capacity_boundary_is_exact(reference: dict) -> None:
    """capacity - 1 set bits still take the list path; capacity takes the dense one. Both against float64."""
    capacity = 64
    inputs = _block_inputs(reference, "gcap", 3, served=False)
    tables = _tables("gcap", 3)
    synthetic = torch.zeros((2, 64, 64), dtype=torch.int64)
    for sample, bits in enumerate((capacity - 1, capacity)):
        for bit in range(bits):
            synthetic[sample, bit // 34, (bit * 7) % 64] |= 1 << (bit % 34)
    edges = synthetic.cuda().view(torch.uint64)
    small = {"header": reference["header"], "edges": edges}
    edge_list = _edge_list(small, capacity)
    assert [int(x) for x in edge_list[3]] == [capacity - 1, capacity]
    pair = {name: tensor[:2].contiguous() for name, tensor in inputs.items() if name != "cap"} | {"cap": True}
    got = _run(small, pair, tables, edge_list=edge_list, capacity=capacity)
    H, _ = _float64_reference(pair, tables, small, EGT_FULL, cap=True)
    errors = {f"sample {z} ({'list' if z == 0 else 'dense'})": _error(got["H"][z], H[z]) for z in (0, 1)}
    print(_line(f"capacity boundary {capacity - 1} / {capacity} set bits", errors), flush=True)
    assert all(e["rel"] <= _F32_RELATIVE for e in errors.values())


def test_fast_path_is_bit_identical_to_the_pre_fallback_kernel(reference: dict) -> None:
    """The digests were taken from the kernel as it stood before the fallback (K2b), at the same num_warps."""
    if not _GOLDEN.exists():
        pytest.skip(f"golden digests not found at {_GOLDEN}")
    golden = json.loads(_GOLDEN.read_text())
    tables = _tables("gcap", golden["block"])
    for mode, digests in golden["digests"].items():
        inputs = _block_inputs(reference, "gcap", golden["block"], served=(mode == "served"))
        got = _run(reference, inputs, tables, capacity=golden["capacity"], warps=golden["num_warps"])
        now = {name: _digest(got[name]) for name in got}
        print(f"K2-GATE bit-identity ({mode}) vs {golden['source']}: {now == digests}", flush=True)
        assert now == digests, (mode, now, digests)


@pytest.mark.parametrize(("label", "block"), _BLOCKS, ids=[f"{a}-b{b}" for a, b in _BLOCKS])
def test_block_matches_ort_fp32_inputs(reference: dict, label: str, block: int) -> None:
    inputs = _block_inputs(reference, label, block, served=False)
    got = _run(reference, inputs, _tables(label, block))
    errors = {name: _error(got[name], inputs[name]) for name in ("H", "A", "attended")}
    print(_line(f"{label} block {block} FP32 inputs cap={inputs['cap']}", errors), flush=True)
    assert errors["H"]["rel"] <= _F32_RELATIVE and errors["A"]["rel"] <= _F32_RELATIVE
    assert errors["attended"]["rel"] <= _F16_STORAGE_RELATIVE


@pytest.mark.parametrize(("label", "block"), _BLOCKS, ids=[f"{a}-b{b}" for a, b in _BLOCKS])
def test_block_matches_ort_served_precision(reference: dict, label: str, block: int) -> None:
    inputs = _block_inputs(reference, label, block, served=True)
    got = _run(reference, inputs, _tables(label, block))
    errors = {name: _error(got[name], inputs[name]) for name in ("H", "A", "attended")}
    print(_line(f"{label} block {block} served FP16 q|k|v,S cap={inputs['cap']}", errors), flush=True)
    assert all(e["rel"] <= _SERVED_RELATIVE for e in errors.values())


def test_the_cap_matters(reference: dict) -> None:
    """gcap block 3 with the cap switched off must miss ORT's A where the gate clips."""
    inputs = _block_inputs(reference, "gcap", 3, served=False)
    uncapped = _run(reference, inputs, _tables("gcap", 3), cap=False)
    error = _error(uncapped["A"], inputs["A"])
    print(f"K2-GATE gcap block 3 with cap OFF: A abs {error['max_abs']:.2e} rel {error['rel']:.2e} (must be large)",
          flush=True)
    assert error["rel"] > 1e-2  # noqa: PLR2004


_TERMS = {
    "qk": EGT_QK, "qk+attack": EGT_QK | EGT_ATTACK, "qk+key": EGT_QK | EGT_KEY, "qk+query": EGT_QK | EGT_QUERY,
    "qk+scaled": EGT_QK | EGT_SCALED, "qk+edge_read": EGT_QK | EGT_EDGE_READ, "qk+door": EGT_QK | EGT_DOOR,
    "qk+gate": EGT_QK | EGT_GATE, "full": EGT_FULL,
}


def _float64_reference(inputs: dict, tables: dict, reference: dict, terms: int, cap: bool) -> tuple:
    t = {name: tensor.double() for name, tensor in tables.items()}
    logits, weights = [], []
    for s0 in range(0, inputs["qkv"].shape[0], _CHUNK):
        qkv = inputs["qkv"][s0:s0 + _CHUNK].double()
        b = qkv.shape[0]
        q = qkv[..., :512].view(b, 64, 32, 16).permute(0, 2, 1, 3)
        k = qkv[..., 512:1024].view(b, 64, 32, 16).permute(0, 2, 1, 3)
        E = unpack_edges(reference["edges"][s0:s0 + _CHUNK]).double()
        S = inputs["S"][s0:s0 + _CHUNK].double()
        e = inputs["state"][s0:s0 + _CHUNK].double()
        F = torch.zeros((b, 32, 64, 64), dtype=torch.float64, device="cuda")
        if terms & EGT_QK:
            F += t["qk_scale"][None, :, None, None] * (q @ k.transpose(-1, -2))
        if terms & EGT_ATTACK:
            F += torch.einsum("hc,zcij->zhij", t["attack"], E)
        if terms & EGT_KEY:
            F += torch.einsum("zhjc,zcij->zhij", torch.einsum("zhjd,hcd->zhjc", k, t["key"]), E)
        if terms & EGT_QUERY:
            F += torch.einsum("zhic,zcij->zhij", torch.einsum("zhid,hcd->zhic", q, t["query"]), E)
        if terms & EGT_SCALED:
            F += S[:, None] * (torch.einsum("hc,zcij->zhij", t["scaled_coefficients"], E) + t["constant_bias"][None])
        if terms & EGT_EDGE_READ:
            F += torch.einsum("hc,zcij->zhij", t["edge_read/w"], e)
        if terms & EGT_DOOR:
            F = F * (1.0 + torch.einsum("hc,zcij->zhij", t["door/w"], e))
        A = torch.softmax(F, -1)
        if terms & EGT_GATE:
            gate = 2.0 * torch.sigmoid(torch.einsum("hc,zcij->zhij", t["gate/w"], e) + t["gate/b"][None, :, None, None])
            A = A * (gate.clamp(max=1.0) if cap else gate)
        logits.append(F)
        weights.append(A)
    return torch.cat(logits), torch.cat(weights)


@pytest.mark.parametrize("terms", list(_TERMS), ids=list(_TERMS))
def test_terms_one_at_a_time(reference: dict, terms: str) -> None:
    inputs = _block_inputs(reference, "gcap", 3, served=False)
    tables = _tables("gcap", 3)
    got = _run(reference, inputs, tables, terms=_TERMS[terms])
    H, A = _float64_reference(inputs, tables, reference, _TERMS[terms], cap=True)
    errors = {"H": _error(got["H"], H), "A": _error(got["A"], A)}
    print(_line(f"term {terms} (gcap block 3) vs float64", errors), flush=True)
    assert errors["H"]["rel"] <= _F32_RELATIVE and errors["A"]["rel"] <= _F32_RELATIVE


def test_h_export_layout_matches_k3(reference: dict) -> None:
    """Our export of block 3 against K3's ORT logits, on the positions the two reference sets share."""
    header_path = _K3_REFERENCE / "k3_ref.json"
    if not header_path.exists():
        pytest.skip(f"K3 reference not found at {_K3_REFERENCE}")
    k3 = json.loads(header_path.read_text())
    ours = reference["header"]["k1_indices"]
    shared = [(ours.index(index), row) for row, index in enumerate(k3["k1_indices"]) if index in ours]
    if not shared:
        pytest.skip("no shared positions")
    inputs = _block_inputs(reference, "gcap", 3, served=False)
    got = _run(reference, inputs, _tables("gcap", 3))["H"]
    count = k3["count"]
    theirs = torch.frombuffer(bytearray((_K3_REFERENCE / "logits_block3.bin").read_bytes()), dtype=torch.float32)
    theirs = theirs.view(count * 32, 64, 64)
    rows = torch.stack([theirs[row * 32:(row + 1) * 32] for _, row in shared]).cuda()
    mine = torch.stack([got[z] for z, _ in shared])
    error = _error(mine, rows)
    print(f"K2-GATE H export vs K3 ORT logits_block3 on {len(shared)} shared positions: abs {error['max_abs']:.2e} "
          f"rel {error['rel']:.2e}", flush=True)
    assert error["rel"] <= _F32_RELATIVE


def test_quant_output_is_the_int8_of_the_fp16_output() -> None:
    """Q1 (round 26): with `quant_output` the kernel writes floor(o * r + 0.5) clamped to +-127 instead of o.

    Synthetic inputs (the kernel against itself, no ORT reference needed): the FP16 run rounds o once more before we
    can see it, so a code may differ by one where o * r sits within one FP16 step of a tie -- never by more, rarely.
    """
    torch.manual_seed(26)
    count = 16
    width = _HEADS * _DEPTH
    bits = torch.randint(0, 2**31, (count, 64, 64), dtype=torch.int64) & torch.randint(0, 2**31, (count, 64, 64),
                                                                                        dtype=torch.int64)
    edges = bits.to(torch.uint64).cuda()
    qkv = (0.5 * torch.randn(count, 64, 3 * width)).half().cuda()
    norm = (0.1 + torch.rand(count, 64, 64)).half().cuda()
    state = torch.randn(count, 16, 64, 64).cuda()
    tables = {"qk_scale": torch.full((_HEADS,), 0.25), "attack": 0.1 * torch.randn(_HEADS, 34),
              "key": 0.1 * torch.randn(_HEADS, 34, _DEPTH), "query": 0.1 * torch.randn(_HEADS, 34, _DEPTH),
              "scaled_coefficients": 0.1 * torch.randn(_HEADS, 34), "constant_bias": 0.1 * torch.randn(_HEADS, 64, 64),
              "edge_read/w": 0.1 * torch.randn(_HEADS, 16), "door/w": 0.1 * torch.randn(_HEADS, 16),
              "gate/w": 0.1 * torch.randn(_HEADS, 16), "gate/b": 0.1 * torch.randn(_HEADS)}
    tables = {name: tensor.float().cuda() for name, tensor in tables.items()}
    buffers = (torch.empty((count, _CAPACITY), dtype=torch.int16, device="cuda"),
               torch.empty((count, _CAPACITY), dtype=torch.int8, device="cuda"),
               torch.empty((count, 2, CELLS), dtype=torch.int16, device="cuda"),
               torch.empty((count,), dtype=torch.int32, device="cuda"),
               torch.empty((count, 64), dtype=torch.int32, device="cuda"),
               torch.empty((count, 64), dtype=torch.int32, device="cuda"))
    launch_egt_edge_list(*buffers, edges, _CAPACITY)
    torch.cuda.synchronize()
    common = {"batch_count": count * _HEADS, "heads": _HEADS, "head_dim": _DEPTH, "architecture": _architecture(),
              "cap": True, "capacity": _CAPACITY}
    fp16 = torch.empty((count, 64, width), dtype=torch.float16, device="cuda")
    launch_attention_egt(fp16, qkv, buffers[:4], edges, norm, state, tables, AttentionEgtSpecialization(**common))
    prescale = (20.0 + 80.0 * torch.rand(width)).cuda()
    codes = torch.full((count, 64, width), 99, dtype=torch.int8, device="cuda")
    launch_attention_egt(codes, qkv, buffers[:4], edges, norm, state, tables,
                         AttentionEgtSpecialization(**common, quant_output=True), quant_prescale=prescale)
    torch.cuda.synchronize()
    expected = torch.clamp(torch.floor(fp16.float() * prescale + 0.5), -127, 127)
    difference = (codes.float() - expected).abs()
    rate = float((difference > 0).float().mean())
    saturated = float((expected.abs() == 127).float().mean())
    print(f"Q1 attention int8 output: max |code diff| {float(difference.max()):.0f}, rate {rate:.2e}, "
          f"saturated {saturated:.2e}, |o| max {float(fp16.float().abs().max()):.3f}", flush=True)
    assert float(difference.max()) <= 1.0
    assert rate < 1e-2
    assert saturated < 0.5  # the test must exercise the rounding, not only the clamp


def test_output_gate_is_the_ungated_output_times_two_sigmoid() -> None:
    """O on the EGT2 blocks (round 26b): with `output_gate` the kernel reads `[q | k | v | g]` (stride 4 width) and
    stores `gate_scale * sigmoid(g) * o`, o being exactly what the ungated kernel computes from `[q | k | v]`.

    Synthetic inputs, the kernel against itself: both runs share every logit and the FP32 accumulator, so the only
    differences are the gate's FP32 product and the store's rounding -- within 2 FP16 ulps of |o|. The int8 store takes
    the gated value too (Q1's `attn_out` codes of the gated branch).
    """
    torch.manual_seed(261)
    count = 16
    width = _HEADS * _DEPTH
    bits = torch.randint(0, 2**31, (count, 64, 64), dtype=torch.int64) & torch.randint(0, 2**31, (count, 64, 64),
                                                                                        dtype=torch.int64)
    edges = bits.to(torch.uint64).cuda()
    qkv = (0.5 * torch.randn(count, 64, 3 * width)).half().cuda()
    gate = (1.5 * torch.randn(count, 64, width)).half().cuda()
    qkvg = torch.cat([qkv, gate], dim=2).contiguous()
    norm = (0.1 + torch.rand(count, 64, 64)).half().cuda()
    state = torch.randn(count, 16, 64, 64).cuda()
    tables = {"qk_scale": torch.full((_HEADS,), 0.25), "attack": 0.1 * torch.randn(_HEADS, 34),
              "key": 0.1 * torch.randn(_HEADS, 34, _DEPTH), "query": 0.1 * torch.randn(_HEADS, 34, _DEPTH),
              "scaled_coefficients": 0.1 * torch.randn(_HEADS, 34), "constant_bias": 0.1 * torch.randn(_HEADS, 64, 64),
              "edge_read/w": 0.1 * torch.randn(_HEADS, 16), "door/w": 0.1 * torch.randn(_HEADS, 16),
              "gate/w": 0.1 * torch.randn(_HEADS, 16), "gate/b": 0.1 * torch.randn(_HEADS)}
    tables = {name: tensor.float().cuda() for name, tensor in tables.items()}
    buffers = (torch.empty((count, _CAPACITY), dtype=torch.int16, device="cuda"),
               torch.empty((count, _CAPACITY), dtype=torch.int8, device="cuda"),
               torch.empty((count, 2, CELLS), dtype=torch.int16, device="cuda"),
               torch.empty((count,), dtype=torch.int32, device="cuda"),
               torch.empty((count, 64), dtype=torch.int32, device="cuda"),
               torch.empty((count, 64), dtype=torch.int32, device="cuda"))
    launch_egt_edge_list(*buffers, edges, _CAPACITY)
    torch.cuda.synchronize()
    common = {"batch_count": count * _HEADS, "heads": _HEADS, "head_dim": _DEPTH, "architecture": _architecture(),
              "cap": True, "capacity": _CAPACITY}
    plain = torch.empty((count, 64, width), dtype=torch.float16, device="cuda")
    launch_attention_egt(plain, qkv, buffers[:4], edges, norm, state, tables, AttentionEgtSpecialization(**common))
    gated = torch.empty((count, 64, width), dtype=torch.float16, device="cuda")
    launch_attention_egt(gated, qkvg, buffers[:4], edges, norm, state, tables,
                         AttentionEgtSpecialization(**common, output_gate=True, gate_scale=2.0))
    prescale = (20.0 + 80.0 * torch.rand(width)).cuda()
    codes = torch.full((count, 64, width), 99, dtype=torch.int8, device="cuda")
    launch_attention_egt(codes, qkvg, buffers[:4], edges, norm, state, tables,
                         AttentionEgtSpecialization(**common, output_gate=True, gate_scale=2.0, quant_output=True),
                         quant_prescale=prescale)
    torch.cuda.synchronize()
    factor = 2.0 * torch.sigmoid(gate.float())
    expected = plain.float() * factor
    error = (gated.float() - expected).abs()
    bound = 2.0 * 2.0**-10 * expected.abs() + 2.0**-24
    ratio = float((error / bound).max())
    changed = float(((gated.float() - plain.float()).abs() > 1e-3).float().mean())
    codes_expected = torch.clamp(torch.floor(gated.float() * prescale + 0.5), -127, 127)
    code_difference = (codes.float() - codes_expected).abs()
    print(f"O attention_egt gate: max |err| / (2 ulp) {ratio:.3f}, outputs moved by the gate {changed:.3f}, "
          f"int8 max |code diff| {float(code_difference.max()):.0f}, rate {float((code_difference > 0).float().mean()):.2e}",
          flush=True)
    assert ratio <= 1.0
    assert changed > 0.5  # the gate must actually act: 2 sigmoid(g) spans (0.1, 1.9) at |g| ~ 1.5
    assert float(code_difference.max()) <= 1.0
    assert float((code_difference > 0).float().mean()) < 1e-2


def test_compiles_to_lc0ex_artifacts() -> None:
    samples = 8
    specialization = AttentionEgtSpecialization(batch_count=samples * _HEADS, heads=_HEADS, head_dim=_DEPTH,
                                                architecture=_architecture(), cap=True, export_h=True)
    null = (lc0ex_pb2.PARAMETER_TYPE_NULL_POINTER,) * 2
    artifact = compile_attention_egt(specialization)
    assert artifact.grid == (samples * _HEADS, 1, 1)
    assert artifact.parameters == (lc0ex_pb2.PARAMETER_TYPE_POINTER,) * 22 + null
    list_specialization = EgtEdgeListSpecialization(samples, _architecture())
    count = compile_egt_edge_count(list_specialization)
    offset = compile_egt_edge_offset(list_specialization)
    listing = compile_egt_edge_list(list_specialization)
    assert count.grid == (samples * 64, 1, 1) and offset.grid == (samples, 1, 1) and listing.grid == (samples * 64, 1, 1)
    assert (count.parameters[:2], offset.parameters[:3], listing.parameters[:5]) == (
        (lc0ex_pb2.PARAMETER_TYPE_POINTER,) * 2, (lc0ex_pb2.PARAMETER_TYPE_POINTER,) * 3,
        (lc0ex_pb2.PARAMETER_TYPE_POINTER,) * 5)

    executable = ExecutableBuilder()
    builder = executable.program(name="main")
    kernels = KernelCache(executable)
    sizes = buffer_bytes(specialization)

    def temporary(size: int) -> object:
        return builder.temporary_buffer(size_bytes=size, alignment_bytes=256)

    edges = builder.buffer(name="/k2/edges", shape=(samples, 64, 64), dtype=lc0ex_pb2.Buffer.DATA_TYPE_U64)
    list_sizes = edge_list_bytes(list_specialization)
    edge_list = EgtEdgeList(
        temporary(list_sizes["edge_list cells"]), temporary(list_sizes["edge_list channels"]),
        temporary(list_sizes["edge_list prefix"]), temporary(list_sizes["edge_list counts"]),
        temporary(4 * samples * 64), temporary(4 * samples * 64),
    )
    egt_edge_list(builder, kernels, edge_list, edges, list_specialization)
    shapes = {"qk_scale": (_HEADS,), "attack": (_HEADS, 34), "key": (_HEADS, 34, _DEPTH), "query": (_HEADS, 34, _DEPTH),
              "scaled_coefficients": (_HEADS, 34), "constant_bias": (_HEADS, 64, 64), "edge_read/w": (_HEADS, 16),
              "door/w": (_HEADS, 16), "gate/w": (_HEADS, 16), "gate/b": (_HEADS,)}
    tables = {
        short: builder.persistent_buffer(name=name, shape=shapes[short], dtype=lc0ex_pb2.Buffer.DATA_TYPE_F32,
                                         alignment_bytes=256)
        for short, name in zip(BLOCK_TABLES, block_table_names("/encoder3"), strict=True)
    }
    attention_egt(
        builder, kernels, temporary(sizes["output"]), temporary(2 * samples * 64 * 1536), edge_list, edges,
        temporary(2 * samples * CELLS), temporary(4 * 16 * samples * CELLS), tables, specialization,
        logits=temporary(sizes["logits (export_h)"]),
    )
