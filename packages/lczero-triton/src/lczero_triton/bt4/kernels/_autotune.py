"""Shared launch candidates for BT4 autotuning, and how every candidate is timed.

Timing (the standing rule of 09-28, `RULING_sm120_report_b3_int8_bt4_first_…_0928.md` §2): every autotune candidate is
timed inside a CUDA graph, as lc0ex serves. A plain-stream loop rounds every kernel period up to a multiple of 2.048 us
on some sm_120 drivers (RTX 5090, 590.48; the 4090 on 610 does not), which turns close candidates into coin flips.
The cold semantics are kept: the default benchmarker flushes L2 before each call (as `triton.testing.do_bench` does)
and `cold_do_bench` re-uploads the operands (as before); the graph measures `[prologue, call] x N` minus
`[prologue] x N`. A call that cannot be captured falls back to the plain benchmarker, with a warning.
`LC0EX_AUTOTUNE_TIMING=plain` restores the old timing for an A/B.
"""

import logging
import os
import weakref
from collections.abc import Callable
from typing import cast

import torch
import triton
import triton.testing

_LOGGER = logging.getLogger(__name__)
_TIMING = os.environ.get("LC0EX_AUTOTUNE_TIMING", "graph")
if _TIMING not in ("graph", "plain"):
    message = f"LC0EX_AUTOTUNE_TIMING={_TIMING!r}; expected 'graph' or 'plain'"
    raise ValueError(message)
_GRAPH_CALLS = 10
_GRAPH_REPLAYS = 5
_UNCAPTURABLE: set[str] = set()

_ELEMENTWISE_CONFIGURATIONS = (
    (64, 1),
    (128, 2),
    (256, 4),
    (256, 8),
    (512, 4),
    (512, 8),
    (1024, 8),
)
_PREPROCESS_CONFIGURATIONS = (
    (128, 2),
    (256, 4),
    (256, 8),
    (512, 4),
    (512, 8),
    (1024, 4),
    (1024, 8),
)


def elementwise_configs() -> list[triton.Config]:
    """Return independent Triton configurations for a flat elementwise kernel."""
    return [
        triton.Config({"block_size": block_size}, num_warps=num_warps)
        for block_size, num_warps in _ELEMENTWISE_CONFIGURATIONS
    ]


def preprocess_configs() -> list[triton.Config]:
    """Return channel-tile candidates for attention-input preprocessing."""
    return [
        triton.Config({"block_size": block_size}, num_warps=num_warps)
        for block_size, num_warps in _PREPROCESS_CONFIGURATIONS
    ]


def active_architecture() -> int:
    """Return the compute capability of the active CUDA device."""
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    return major * 10 + minor


_MAX_HOST_MIRRORS = 64
_MirrorEntry = tuple["weakref.ReferenceType[torch.Tensor]", torch.Tensor]
_HOST_MIRROR_CACHE: dict[
    tuple[int, tuple[int, ...], torch.dtype, tuple[int, ...]], _MirrorEntry
] = {}


def _extract_gpu_tensors(fn: object) -> list[torch.Tensor]:
    """Extract CUDA tensors from a kernel call closure or callable object."""
    closure = getattr(fn, "__closure__", None)
    if not closure:
        return []
    gpu_tensors: list[torch.Tensor] = []
    seen_ptrs: set[int] = set()
    for cell in closure:
        val = cell.cell_contents
        if isinstance(val, (list, tuple)):
            for item in val:
                if (
                    isinstance(item, torch.Tensor)
                    and item.is_cuda
                    and item.data_ptr() not in seen_ptrs
                ):
                    gpu_tensors.append(item)
                    seen_ptrs.add(item.data_ptr())
        elif isinstance(val, dict):
            for item in val.values():
                if (
                    isinstance(item, torch.Tensor)
                    and item.is_cuda
                    and item.data_ptr() not in seen_ptrs
                ):
                    gpu_tensors.append(item)
                    seen_ptrs.add(item.data_ptr())
        elif (
            isinstance(val, torch.Tensor)
            and val.is_cuda
            and val.data_ptr() not in seen_ptrs
        ):
            gpu_tensors.append(val)
            seen_ptrs.add(val.data_ptr())
    return gpu_tensors


def _summarize_timings(
    times: list[float],
    quantiles: tuple[float, ...] | list[float] | None,
    return_mode: str,
) -> list[float] | float:
    """Summarize sorted execution times into quantiles or a single metric."""
    if quantiles is not None:
        times_tensor = torch.tensor(times, dtype=torch.float32)
        q_tensor = torch.tensor(list(quantiles), dtype=torch.float32)
        return cast("list[float]", torch.quantile(times_tensor, q_tensor).tolist())

    if return_mode == "median":
        return times[len(times) // 2]
    if return_mode == "min":
        return times[0]
    if return_mode == "max":
        return times[-1]
    return sum(times) / len(times)


def _graph_milliseconds(prologue: Callable[[], object], call: Callable[[], object], where: str) -> float | None:
    """Milliseconds per `call`: `[prologue, call] x N` minus `[prologue] x N`, each one CUDA graph, best of 5 replays.

    None when the call cannot be captured (the caller then times it the plain way).
    """
    stream = torch.cuda.Stream()
    try:
        call()  # compile and warm outside the capture
        torch.cuda.synchronize()
        graphs = []
        for with_call in (True, False):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(_GRAPH_CALLS):
                    prologue()
                    if with_call:
                        call()
            graphs.append(graph)
    except Exception as error:  # noqa: BLE001 - any capture failure means "time it the plain way"
        torch.cuda.synchronize()
        if where not in _UNCAPTURABLE:
            _UNCAPTURABLE.add(where)
            _LOGGER.warning("autotune: %s cannot be captured in a CUDA graph (%s); timed on a plain stream",
                            where, type(error).__name__)
        return None
    start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
    stop = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
    best: list[float] = []
    with torch.cuda.stream(stream):
        for graph in graphs:
            graph.replay()
            fastest = float("inf")
            for _ in range(_GRAPH_REPLAYS):
                start.record(stream)  # type: ignore[no-untyped-call]
                graph.replay()
                stop.record(stream)  # type: ignore[no-untyped-call]
                stop.synchronize()  # type: ignore[no-untyped-call]
                fastest = min(fastest, start.elapsed_time(stop))  # type: ignore[no-untyped-call]
            best.append(fastest)
    del graphs
    torch.cuda.synchronize()
    return max(best[0] - best[1], 1e-6) / _GRAPH_CALLS


def _as_result(milliseconds: float, quantiles: tuple[float, ...] | list[float] | None) -> list[float] | float:
    return [milliseconds] * len(quantiles) if quantiles is not None else milliseconds


def graph_do_bench(
    fn: Callable[[], object],
    warmup: int = 25,
    rep: int = 100,
    grad_to_none: object = None,
    quantiles: tuple[float, ...] | list[float] | None = None,
    return_mode: str = "mean",
) -> list[float] | float:
    """`triton.testing.do_bench`'s measurement (L2 flushed before every call), timed inside CUDA graphs."""
    cache = triton.runtime.driver.active.get_empty_cache_for_benchmark()
    milliseconds = _graph_milliseconds(cache.zero_, fn, getattr(fn, "__qualname__", "kernel"))
    if milliseconds is None:
        return cast("list[float] | float", triton.testing.do_bench(
            fn, warmup=warmup, rep=rep, grad_to_none=grad_to_none, quantiles=quantiles, return_mode=return_mode))
    return _as_result(milliseconds, quantiles)


def _install_graph_benchmarker() -> None:
    """Make every autotuner that does not name its own benchmarker time through `graph_do_bench`.

    Triton resolves an autotuner's default benchmarker lazily, at its first bench (`Autotuner.do_bench`, a cached
    property over `driver.active.get_benchmarker()`), so replacing the NVIDIA driver's method here is enough for every
    kernel module imported with this one.
    """
    from triton.backends.nvidia.driver import CudaDriver  # noqa: PLC0415

    CudaDriver.get_benchmarker = lambda self: graph_do_bench  # type: ignore[method-assign]  # noqa: ARG005


if _TIMING == "graph":
    _install_graph_benchmarker()


def cold_do_bench(
    fn: object,
    warmup: int = 5,
    rep: int = 20,
    grad_to_none: object = None,
    quantiles: tuple[float, ...] | list[float] | None = (0.5, 0.2, 0.8),
    return_mode: str = "mean",
) -> list[float] | float:
    """Benchmark a kernel while evicting L2 lines via host buffer re-upload.

    Standard `triton.testing.do_bench` uses repeated executions on the same GPU buffer
    with `cache.zero_()`, which fails to evict L2 lines on modern NVIDIA GPUs and
    favors configurations with poor DRAM reuse.

    This benchmarker re-uploads all input/output GPU tensors from pinned host
    memory via PCIe DMA before each timed repetition, ensuring DRAM access latency
    and L2 cache eviction mirror real multi-layer model inference.
    """
    gpu_tensors = _extract_gpu_tensors(fn)
    if not gpu_tensors or not callable(fn):
        return cast(
            "list[float] | float",
            triton.testing.do_bench(
                fn,
                warmup=warmup,
                rep=rep,
                grad_to_none=grad_to_none,
                quantiles=quantiles,
                return_mode=return_mode,
            ),
        )

    if len(_HOST_MIRROR_CACHE) > _MAX_HOST_MIRRORS:
        _HOST_MIRROR_CACHE.clear()

    host_mirrors: list[torch.Tensor] = []
    for t in gpu_tensors:
        key = (t.data_ptr(), tuple(t.shape), t.dtype, tuple(t.stride()))
        entry = _HOST_MIRROR_CACHE.get(key)
        # The allocator reuses addresses: an entry is valid only while the SAME tensor object owns it.
        if entry is None or entry[0]() is not t:
            entry = (weakref.ref(t), t.detach().cpu().pin_memory())
            _HOST_MIRROR_CACHE[key] = entry
        host_mirrors.append(entry[1])

    if _TIMING == "graph":
        def upload() -> None:
            for t, h in zip(gpu_tensors, host_mirrors, strict=True):
                t.copy_(h, non_blocking=True)

        milliseconds = _graph_milliseconds(upload, cast("Callable[[], object]", fn),
                                           getattr(fn, "__qualname__", "kernel"))
        if milliseconds is not None:
            return _as_result(milliseconds, quantiles)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [
        torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        for _ in range(rep)
    ]
    end_events = [
        torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        for _ in range(rep)
    ]

    for i in range(rep):
        for t, h in zip(gpu_tensors, host_mirrors, strict=True):
            t.copy_(h, non_blocking=True)
        torch.cuda.synchronize()

        start_events[i].record()  # type: ignore[no-untyped-call]
        fn()
        end_events[i].record()  # type: ignore[no-untyped-call]

    torch.cuda.synchronize()
    times = [
        s.elapsed_time(e)  # type: ignore[no-untyped-call]
        for s, e in zip(start_events, end_events, strict=True)
    ]
    times.sort()
    return _summarize_timings(times, quantiles, return_mode)
