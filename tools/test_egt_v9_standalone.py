#!/usr/bin/env python3
"""Standalone check of the v9 EGT2 attention CUBIN (`attention_egt_v9` / `egt_v9.cu`) against a float64 reference and the served
Triton kernel, on random edge bits and on real positions' E. It compiles the package kernel with exactly the builder's flags
(`compile_arguments`) plus a host launcher, so what it checks is what an artifact carries.

Checks per (shape, batch): int8 codes (or FP16 output) against float64 -- mismatch rate and max code distance, within B9's class --,
H (export) against float64, determinism (two runs byte-identical); then, unless --check-only, graph-timed microseconds per call for v9
and the served kernel (served FP32 and B9 FP16 arithmetic), interleaved rounds.

usage: test_egt_v9_standalone.py [--real-e E_bits.bin] [--shapes 32x32] [--batches 16,64,84] [--export [--h16]] [--f16-out]
                                 [--check-only]
       E_bits.bin = K1's 1,024 real positions (Kovax/briefs_2026-09-11/r20c_itemE/K1/ref_egt/E_bits.bin, sha256 82bb95b2...).
"""

import argparse
import ctypes
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from lczero_triton.bt4.kernels import attention_egt as egt
from lczero_triton.bt4.kernels.attention_egt_v9 import (
    SOURCE,
    AttentionEgtV9Specialization,
    compile_arguments,
    launch_pack_egt_v9_tables,
    packed_bytes,
)

NVCC = "/usr/local/cuda-12.9/bin/nvcc"
CAPACITY = 1024
_HOST = """
#include "{source}"
extern "C" int egt_v9_host_launch(void* out, void* hexp, const void* qkv, const void* edges, const void* norm, const void* state,
                                  const void* t0, const void* t1, const void* t2, const void* t3, const void* t4, const void* t5,
                                  const void* t6, const void* t7, const void* t8, const void* t9, const void* prescale, int grid,
                                  void* stream) {{
  static bool configured = false;
  if (!configured) {{
    cudaFuncSetAttribute(egt_v9_attention, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
    configured = true;
  }}
  egt_v9_attention<<<grid, kThreads, kSmem, static_cast<cudaStream_t>(stream)>>>(
      out, hexp, static_cast<const __half*>(qkv), static_cast<const unsigned long long*>(edges),
      static_cast<const __half*>(norm), static_cast<const __half*>(state), static_cast<const float*>(t0),
      static_cast<const float*>(t1), static_cast<const float*>(t2), static_cast<const float*>(t3), static_cast<const float*>(t4),
      static_cast<const float*>(t5), static_cast<const float*>(t6), static_cast<const float*>(t7), static_cast<const float*>(t8),
      static_cast<const float*>(t9), static_cast<const float*>(prescale));
  return static_cast<int>(cudaGetLastError());
}}
"""


def _arch() -> int:
    major, minor = torch.cuda.get_device_capability()
    return 10 * major + minor


def build(spec: AttentionEgtV9Specialization, directory: Path) -> ctypes.CDLL:
    source = directory / "host.cu"
    source.write_text(_HOST.format(source=os.environ.get("EGT_V9_SOURCE", SOURCE)), encoding="utf-8")
    library = directory / (f"egt_v9_{spec.heads}x{spec.head_dim}_b{spec.batch_size}_h{int(spec.export_h)}_f{int(not spec.quant_output)}"
                           f"{'_packed' if spec.packed else ''}{'_h16' if spec.h_f16 else ''}.so")
    command = [NVCC, "-O3", f"-arch=sm_{spec.architecture}", "-std=c++17", "-shared", "-Xcompiler", "-fPIC", "-Xptxas", "-v", "-w",
               *compile_arguments(spec), *os.environ.get("EGT_V9_NVCC_EXTRA", "").split(), str(source), "-o", str(library)]
    done = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
    if done.returncode:
        sys.exit(done.stderr)
    usage = [line.split("info    :")[-1].strip() for line in done.stderr.splitlines() if "registers" in line or "spill" in line]
    print(f"# built {library.name}: {' | '.join(usage)}")
    lib = ctypes.CDLL(str(library))
    lib.egt_v9_host_launch.restype = ctypes.c_int
    lib.egt_v9_host_launch.argtypes = [ctypes.c_void_p] * 17 + [ctypes.c_int, ctypes.c_void_p]
    return lib


def real_bits(path: Path, batch: int) -> torch.Tensor:
    """[batch, 64, 64, 34] bool from K1's set ([position][34][64][64] numpy packbits, big-endian), positions cycled."""
    raw = torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.uint8).view(-1, 17408)
    index = torch.arange(batch) % raw.shape[0]
    bits = (raw[index].unsqueeze(-1) >> torch.arange(7, -1, -1, dtype=torch.uint8)) & 1
    return bits.view(batch, 34, 64, 64).permute(0, 2, 3, 1).contiguous().bool()


def make_inputs(batch: int, heads: int, depth: int, real_e: Path | None, seed: int = 28) -> dict:
    """The inputs of the backend's v9 harness (`briefs_2026-09-28/v9_0928/egt_ref_v9_0928.py`), in its draw order, so timings
    compare across the two tools."""
    width = heads * depth
    torch.manual_seed(seed)
    bits = real_bits(real_e, batch) if real_e else torch.rand(batch, 64, 64, 34) < 0.003
    edges = (bits.long() << torch.arange(34)).sum(-1).contiguous().view(torch.uint64).cuda()
    lists = (torch.empty((batch, CAPACITY), dtype=torch.int16, device="cuda"),
             torch.empty((batch, CAPACITY), dtype=torch.int8, device="cuda"),
             torch.empty((batch, 2, egt.CELLS), dtype=torch.int16, device="cuda"),
             torch.empty((batch,), dtype=torch.int32, device="cuda"),
             torch.zeros((batch, 64), dtype=torch.int32, device="cuda"),
             torch.zeros((batch, 64), dtype=torch.int32, device="cuda"))
    egt.launch_egt_edge_list(*lists, edges, CAPACITY)
    qkv = (torch.randn(batch, 64, 4 * width, device="cuda") * 0.5).half()
    norm = torch.rand(batch, 64, 64, device="cuda").half()
    state16 = (torch.randn(batch, 16, 64, 64, device="cuda") * 0.7).half()
    scale = (64 / depth) ** 0.5
    tables = {"qk_scale": torch.full((heads,), depth ** -0.5, device="cuda"),
              "attack": torch.randn(heads, 34, device="cuda") * 0.3,
              "key": torch.randn(heads, 34, depth, device="cuda") * 0.05 * scale,
              "query": torch.randn(heads, 34, depth, device="cuda") * 0.05 * scale,
              "scaled_coefficients": torch.randn(heads, 34, device="cuda") * 0.3,
              "constant_bias": torch.randn(heads, 64, 64, device="cuda") * 0.3,
              "edge_read/w": torch.randn(heads, 16, device="cuda") * 0.3,
              "door/w": torch.randn(heads, 16, device="cuda") * 0.05,
              "gate/w": torch.randn(heads, 16, device="cuda") * 0.3, "gate/b": torch.randn(heads, device="cuda")}
    x = {"heads": heads, "depth": depth, "bits": bits.cuda(), "edges": edges, "lists": lists[:4], "qkv": qkv, "norm": norm,
         "state16": state16, "state_cm": state16.permute(0, 2, 3, 1).contiguous(), "tables": tables,
         "prescale": torch.full((width,), 40.0, device="cuda"), "gate_scale": 1.5}
    torch.cuda.synchronize()
    return x


def reference(x: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Float64: (output before rounding [B, 64, W], H [B * heads, 64, 64]). H3: cap off, output gate on."""
    d = torch.float64
    heads, depth = x["heads"], x["depth"]
    batch = x["qkv"].shape[0]
    t = {k: v.to(d) for k, v in x["tables"].items()}
    q, k, v, g = (x["qkv"].to(d).view(batch, 64, 4, heads, depth)[:, :, n].permute(0, 2, 1, 3) for n in range(4))
    bits, norm, state = x["bits"].to(d), x["norm"].to(d), x["state16"].to(d)
    scores = torch.einsum("bhid,bhjd->bhij", q, k) * t["qk_scale"][None, :, None, None]
    row = torch.einsum("bhid,hcd->bhic", q, t["query"]) + t["attack"][None, :, None, :]
    col = torch.einsum("bhjd,hcd->bhjc", k, t["key"])
    scores = scores + (torch.einsum("bijc,bhic->bhij", bits, row) + torch.einsum("bijc,bhjc->bhij", bits, col)
                       + norm[:, None] * torch.einsum("bijc,hc->bhij", bits, t["scaled_coefficients"]))
    scores = scores + norm[:, None] * t["constant_bias"][None] + torch.einsum("hs,bsij->bhij", t["edge_read/w"], state)
    h_logits = scores * (1.0 + torch.einsum("hs,bsij->bhij", t["door/w"], state))
    gate = torch.einsum("hs,bsij->bhij", t["gate/w"], state) + t["gate/b"][None, :, None, None]
    probabilities = torch.softmax(h_logits, dim=-1) * (2.0 * torch.sigmoid(gate))
    out = torch.einsum("bhij,bhjd->bhid", probabilities, v) * (x["gate_scale"] * torch.sigmoid(g))
    return out.permute(0, 2, 1, 3).reshape(batch, 64, heads * depth), h_logits.reshape(batch * heads, 64, 64)


def packed_tables(x: dict) -> dict:
    """The served form's inputs: the package's pack kernel writes the FP16 packs from the FP32 plans (as every batch in an artifact);
    the result is keyed by BLOCK_TABLES' short names as `served_tables` keys the builder's buffers."""
    t = x["tables"]
    packs = {name: torch.empty(size // 2, dtype=torch.float16, device="cuda")
             for name, size in packed_bytes(x["heads"], x["depth"]).items()}
    launch_pack_egt_v9_tables(packs, t, x["heads"], x["depth"])
    torch.cuda.synchronize()
    return t | {"query": packs["query"], "key": packs["key"], "edge_read/w": packs["weights"],
                "constant_bias": packs["constant_bias"]}


def v9_launch(lib: ctypes.CDLL, spec: AttentionEgtV9Specialization, x: dict, out: torch.Tensor, hexp: torch.Tensor | None):
    source = packed_tables(x) if x.get("packed") else x["tables"]
    tables = [source[name] for name in egt.BLOCK_TABLES]

    def launch() -> None:
        rc = lib.egt_v9_host_launch(out.data_ptr(), hexp.data_ptr() if hexp is not None else None, x["qkv"].data_ptr(),
                                    x["edges"].data_ptr(), x["norm"].data_ptr(), x["state_cm"].data_ptr(),
                                    *(tensor.data_ptr() for tensor in tables), x["prescale"].data_ptr(), spec.grid,
                                    torch.cuda.current_stream().cuda_stream)
        if rc:
            message = f"egt_v9 launch error {rc}"
            raise RuntimeError(message)
    return launch


def triton_launch(x: dict, f16: bool, export_h: bool, quant: bool, out: torch.Tensor, logits: torch.Tensor | None):
    heads, depth = x["heads"], x["depth"]
    spec = egt.AttentionEgtSpecialization(
        batch_count=x["qkv"].shape[0] * heads, heads=heads, head_dim=depth, architecture=_arch(), cap=False, export_h=export_h,
        capacity=CAPACITY, state_f32=False, overflow_exact=True, quant_output=quant, output_gate=True,
        gate_scale=x["gate_scale"], accumulate_f16=f16, state_math_f16=f16)
    return lambda: egt.launch_attention_egt(out, x["qkv"], x["lists"], x["edges"], x["norm"], x["state16"], x["tables"], spec,
                                            logits=logits if export_h else None, quant_prescale=x["prescale"] if quant else None)


def graph_time(launch, inner: int = 20, replays: int = 7) -> float:
    launch()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        for _ in range(inner):
            launch()
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(replays):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / inner)
    return sorted(times)[len(times) // 2]


def main() -> int:  # noqa: C901, PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-e", type=Path, default=None)
    parser.add_argument("--shapes", default="32x32")
    parser.add_argument("--batches", default="16,64,84")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--h16", action="store_true", help="H exported in FP16 (step 3's form; needs --export)")
    parser.add_argument("--f16-out", action="store_true", help="FP16 output (the FP16 twin's form) instead of int8 codes")
    parser.add_argument("--control", default="")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--fp32-tables", action="store_true", help="the kernel reads the FP32 plans (not served; a test switch)")
    args = parser.parse_args()
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"# {torch.cuda.get_device_name()} sm_{_arch()} ({sms} SMs); E {'real: ' + str(args.real_e) if args.real_e else 'random'}; "
          f"export_h={args.export} h16={args.h16} f16_out={args.f16_out} control={args.control!r}")
    failed = False
    with tempfile.TemporaryDirectory(prefix="egt_v9_") as tmp:
        for shape in args.shapes.split(","):
            heads, depth = (int(v) for v in shape.split("x"))
            for batch in (int(b) for b in args.batches.split(",")):
                spec = AttentionEgtV9Specialization(batch_size=batch, heads=heads, head_dim=depth, architecture=_arch(), sms=sms,
                                                    cap=False, gate_scale=1.5, export_h=args.export, h_f16=args.h16,
                                                    quant_output=not args.f16_out, control=args.control,
                                                    packed=not args.fp32_tables)
                lib = build(spec, Path(tmp))
                x = make_inputs(batch, heads, depth, args.real_e)
                x["packed"] = spec.packed
                width = heads * depth
                ref_out, ref_h = reference(x)
                out_dtype = torch.float16 if args.f16_out else torch.int8
                out = torch.zeros((batch, 64, width), dtype=out_dtype, device="cuda")
                h_dtype = torch.float16 if args.h16 else torch.float32
                hexp = (torch.full((batch * heads, 64, 64), float("nan"), dtype=h_dtype, device="cuda") if args.export
                        else None)
                v9_launch(lib, spec, x, out, hexp)()
                again = torch.zeros_like(out)
                v9_launch(lib, spec, x, again, hexp)()
                torch.cuda.synchronize()
                line = f"{heads}x{depth} b{batch}:"
                if args.f16_out:
                    err = (out.double() - ref_out).abs()
                    line += f" FP16 out max |err| {err.max().item():.2e} (ref |max| {ref_out.abs().max().item():.2f})"
                    ok = err.max().item() < 2e-2
                else:
                    value = ref_out * x["prescale"].double()[None, None, :]
                    codes = torch.clamp(torch.floor(value + 0.5), -127, 127).to(torch.int8)
                    diff = (out.int() - codes.int()).abs()
                    rate = (diff > 0).double().mean().item()
                    line += f" codes: mismatches {100 * rate:.4f} % max |d| {diff.max().item()}"
                    ok = rate < 0.004 and diff.max().item() <= 1
                if hexp is not None:
                    dh = hexp.double() - ref_h
                    herr = dh.abs().max().item()
                    big = ref_h.abs() > 1.0
                    rel = (dh.abs()[big] / ref_h.abs()[big]).max().item() if big.any() else 0.0
                    # the readback's view of the error: e_hat = e + O H with a unit-scale O [16, heads] / sqrt(heads)
                    readback = torch.randn(16, heads, dtype=torch.float64, device="cuda", generator=torch.Generator(
                        device="cuda").manual_seed(5)) / heads ** 0.5
                    e_err = torch.einsum("ch,bhij->bcij", readback, dh.view(batch, heads, 64, 64)).abs().max().item()
                    line += (f"; H max |err| {herr:.2e} (|H| max {ref_h.abs().max().item():.1f}, rel {rel:.1e} above 1), "
                             f"O.dH max {e_err:.2e}")
                    ok = ok and herr < 2e-2
                if os.environ.get("EGT_V9_DUMP"):  # A/B of two kernel sources on identical inputs
                    torch.save({"out": out.cpu(), "h": hexp.cpu() if hexp is not None else None},
                               f"{os.environ['EGT_V9_DUMP']}_b{batch}.pt")
                deterministic = torch.equal(out, again)
                line += f"; deterministic {deterministic}"
                ok = ok and deterministic
                print(("PASS " if ok else "FAIL ") + line)
                failed |= not ok
                if args.check_only:
                    continue
                tri = torch.zeros_like(out)
                logits = torch.empty((batch * heads, 64, 64), device="cuda") if args.export else None
                calls = {"v9": v9_launch(lib, spec, x, out, hexp),
                         "Triton FP32": triton_launch(x, False, args.export, not args.f16_out, tri, logits),
                         "Triton B9 FP16": triton_launch(x, True, args.export, not args.f16_out, tri, logits)}
                rounds = {name: [] for name in calls}
                for _ in range(3):
                    for name, call in calls.items():
                        rounds[name].append(graph_time(call))
                best = {name: sorted(v)[1] for name, v in rounds.items()}
                b9 = best["Triton B9 FP16"]
                print("     " + "  ".join(f"{name} {us:7.1f} us ({100 * (us / b9 - 1):+.1f} %)" for name, us in best.items()))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
