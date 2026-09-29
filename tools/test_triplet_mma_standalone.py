"""Standalone check of `triplet_mma.cu` (the triplet operator on the tensor cores) against float64 and the served Triton form.

Per batch: va from the CUBIN (reading the FP16 cell-major e_hat copy) and from the served `fused` Triton kernel (`dot="fp16"`, FP32
e_hat), both against a float64 reference of `triplet_site`'s math; determinism; then, unless --check-only, graph-timed microseconds
per call: the CUBIN, the CUBIN + the cell-major FP16 cast it reads (`cast_state_cell_major`), and the Triton fused kernel.

usage: test_triplet_mma_standalone.py [--batches 16,64,84] [--contraction path|ag] [--va16] [--check-only]
       env TRIPLET_MMA_SOURCE (another kernel file), TRIPLET_MMA_NVCC_EXTRA
"""

import argparse
import ctypes
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from lczero_triton.bt4.kernels import triplet_site as ts
from lczero_triton.bt4.kernels.egt_state_tiles import CastStateCellMajorSpecialization, launch_cast_state_cell_major

NVCC = "/usr/local/cuda-12.9/bin/nvcc"
SOURCE = Path(__file__).resolve().parents[1] / "packages/lczero-triton/src/lczero_triton/bt4/kernels/triplet_mma.cu"
_HOST = """
#include "{source}"
extern "C" int triplet_mma_host_launch(void* va, const void* copy, const void* vw, const void* gw, const void* gb, int grid,
                                       void* stream) {{
  static bool configured = false;
  if (!configured) {{
    cudaFuncSetAttribute(triplet_mma, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
    configured = true;
  }}
  triplet_mma<<<grid, kThreads, kSmem, static_cast<cudaStream_t>(stream)>>>(
      va, static_cast<const __half*>(copy), static_cast<const float*>(vw), static_cast<const float*>(gw),
      static_cast<const float*>(gb));
  return static_cast<int>(cudaGetLastError());
}}
"""


def _arch() -> int:
    major, minor = torch.cuda.get_device_capability()
    return 10 * major + minor


def build(batch: int, contraction: str, directory: Path, va16: bool = False) -> ctypes.CDLL:
    source = directory / "host.cu"
    source.write_text(_HOST.format(source=os.environ.get("TRIPLET_MMA_SOURCE", SOURCE)), encoding="utf-8")
    library = directory / f"triplet_mma_b{batch}_{contraction}{'_va16' if va16 else ''}.so"
    command = [NVCC, "-O3", f"-arch=sm_{_arch()}", "-std=c++17", "-shared", "-Xcompiler", "-fPIC", "-Xptxas", "-v", "-w",
               f"-DSAMPLES={batch}", f"-DCONTRACTION={ts._CONTRACTIONS[contraction]}", f"-DVA_F16={int(va16)}",  # noqa: SLF001
               *os.environ.get("TRIPLET_MMA_NVCC_EXTRA", "").split(), str(source), "-o", str(library)]
    done = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
    if done.returncode:
        sys.exit(done.stderr)
    usage = [line.split("info    :")[-1].strip() for line in done.stderr.splitlines() if "registers" in line or "spill" in line]
    print(f"# built {library.name}: {' | '.join(usage)}")
    lib = ctypes.CDLL(str(library))
    lib.triplet_mma_host_launch.restype = ctypes.c_int
    lib.triplet_mma_host_launch.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int, ctypes.c_void_p]
    return lib


def make_inputs(batch: int, seed: int = 29) -> dict:
    torch.manual_seed(seed)
    e_hat = torch.randn(batch, 16, 64, 64, device="cuda") * 0.7
    return {
        "e_hat": e_hat,
        "copy": e_hat.permute(0, 2, 3, 1).contiguous().half(),
        "value_weight": torch.randn(32, 16, device="cuda") * 0.35,
        "gate_weight": torch.randn(16, 16, device="cuda") * 0.5,
        "gate_bias": torch.randn(16, device="cuda") * 0.3,
    }


def reference(x: dict, contraction: str) -> torch.Tensor:
    """Float64 va [B, 32, 64, 64] of `triplet_site`'s docstring math."""
    d = torch.float64
    e = x["e_hat"].to(d)
    batch = e.shape[0]
    n = e / torch.sqrt((e * e).mean(dim=1, keepdim=True) + ts.EPSILON)
    v = torch.einsum("oc,bcij->boij", x["value_weight"].to(d), n)
    eg = torch.einsum("oc,bcij->boij", x["gate_weight"].to(d), n) + x["gate_bias"].to(d)[None, :, None, None]
    e_in, g_in, e_out, g_out = eg[:, 0:4], eg[:, 4:8], eg[:, 8:12], eg[:, 12:16]
    a_in = torch.softmax(e_in, dim=-1) * torch.sigmoid(g_in)       # [b, h, i, k]
    a_out = torch.softmax(e_out, dim=-2) * torch.sigmoid(g_out)    # [b, h, k, i], softmax over k
    v_in, v_out = v[:, :16].reshape(batch, 4, 4, 64, 64), v[:, 16:].reshape(batch, 4, 4, 64, 64)
    if contraction == "path":
        va_in = torch.einsum("bhik,bhdkj->bhdij", a_in, v_in)
        va_out = torch.einsum("bhki,bhdjk->bhdij", a_out, v_out)
    else:
        va_in = torch.einsum("bhik,bhdjk->bhdij", a_in, v_in)
        va_out = torch.einsum("bhki,bhdkj->bhdij", a_out, v_out)
    return torch.cat([va_in.reshape(batch, 16, 64, 64), va_out.reshape(batch, 16, 64, 64)], dim=1)


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="16,64,84")
    parser.add_argument("--contraction", default="path", choices=("path", "ag"))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--va16", action="store_true", help="va stored FP16, by both kernels")
    args = parser.parse_args()
    print(f"# {torch.cuda.get_device_name()} sm_{_arch()}; contraction {args.contraction}")
    failed = False
    with tempfile.TemporaryDirectory(prefix="triplet_mma_") as tmp:
        for batch in (int(b) for b in args.batches.split(",")):
            lib = build(batch, args.contraction, Path(tmp), args.va16)
            x = make_inputs(batch)
            ref = reference(x, args.contraction)
            va_dtype = torch.float16 if args.va16 else torch.float32
            va = torch.full((batch, 32, 64, 64), float("nan"), dtype=va_dtype, device="cuda")

            def mma_launch(out: torch.Tensor = va) -> None:
                rc = lib.triplet_mma_host_launch(out.data_ptr(), x["copy"].data_ptr(), x["value_weight"].data_ptr(),
                                                 x["gate_weight"].data_ptr(), x["gate_bias"].data_ptr(), batch * 8,
                                                 torch.cuda.current_stream().cuda_stream)
                if rc:
                    message = f"triplet_mma launch error {rc}"
                    raise RuntimeError(message)
            mma_launch()
            again = torch.full_like(va, float("nan"))
            mma_launch(again)
            spec = ts.TripletSiteSpecialization(batch_count=batch, architecture=_arch(), contraction=args.contraction,
                                                form="fused", dot="fp16")
            tri = torch.full_like(va, float("nan"))
            ts.launch_triplet_fused(tri, x["e_hat"], x["value_weight"], x["gate_weight"], x["gate_bias"], spec)
            torch.cuda.synchronize()
            scale = ref.abs().max().item()
            err_mma = (va.double() - ref).abs().max().item()
            err_tri = (tri.double() - ref).abs().max().item()
            rms_mma = (va.double() - ref).pow(2).mean().sqrt().item()
            rms_tri = (tri.double() - ref).pow(2).mean().sqrt().item()
            deterministic = torch.equal(va, again)
            ok = deterministic and err_mma <= 2.0 * max(err_tri, 1e-3) and not torch.isnan(va).any().item()
            print(("PASS " if ok else "FAIL ") + f"b{batch}: |va| max {scale:.2f}; max |err| MMA {err_mma:.2e} (rms {rms_mma:.1e}), "
                  f"Triton fused FP16 dot {err_tri:.2e} (rms {rms_tri:.1e}); deterministic {deterministic}")
            failed |= not ok
            if args.check_only:
                continue
            copy = torch.empty_like(x["copy"])
            cast_spec = CastStateCellMajorSpecialization(batch, _arch(), states=16)
            calls = {
                "MMA": mma_launch,
                "cast + MMA": lambda: (launch_cast_state_cell_major(copy, x["e_hat"], cast_spec), mma_launch()),
                "Triton fused": lambda: ts.launch_triplet_fused(tri, x["e_hat"], x["value_weight"], x["gate_weight"],
                                                                x["gate_bias"], spec),
            }
            rounds = {name: [] for name in calls}
            for _ in range(3):
                for name, call in calls.items():
                    rounds[name].append(graph_time(call))
            best = {name: sorted(v)[1] for name, v in rounds.items()}
            base = best["Triton fused"]
            print("     " + "  ".join(f"{name} {us:6.1f} us ({100 * (us / base - 1):+.1f} %)" for name, us in best.items()))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
