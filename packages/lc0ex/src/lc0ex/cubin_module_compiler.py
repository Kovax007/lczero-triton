"""Convert hand-compiled CUBIN modules into executable-linker artifacts.

Triton is the vehicle for FP16 fusion glue, but it does not carry FP8 or
persistent kernels on sm_89, so those specializations arrive as CUBINs built by
nvcc from CUTLASS sources. This module is the other half of
`triton_module_compiler`: it takes such a CUBIN and the launch metadata the
compiler cannot infer, and produces the same `KernelArtifact`.
"""

import os
import subprocess
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

from lc0ex.kernel_builder import KernelArtifact
from lc0ex.proto import lc0ex_pb2


def artifact_from_cubin(  # noqa: PLR0913
    cubin: bytes,
    *,
    function: str,
    parameters: Sequence[lc0ex_pb2.ParameterType],
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    dynamic_shared_memory_bytes: int = 0,
) -> KernelArtifact:
    """Wrap one `extern "C" __global__` entry point in a linker artifact.

    Unlike the Triton path there are no trailing scratch parameters: the tuple
    given here is exactly the kernel's signature, in declaration order.
    """
    if not cubin:
        message = "The CUBIN is empty."
        raise ValueError(message)
    if not function:
        message = "A CUBIN artifact needs the mangled-free entry point name."
        raise ValueError(message)
    return KernelArtifact(
        binary_format=lc0ex_pb2.Binary.FORMAT_CUBIN,
        binary_data=cubin,
        function=function,
        parameters=tuple(parameters),
        grid=grid,
        block=block,
        dynamic_shared_memory_bytes=dynamic_shared_memory_bytes,
    )


def compile_cuda(
    source: str,
    *,
    architecture: str,
    include_directories: Sequence[Path | str] = (),
    nvcc: Path | str | None = None,
    extra_arguments: Sequence[str] = (),
) -> bytes:
    """Compile one CUDA translation unit to a CUBIN for *architecture*.

    `architecture` is the same `sm_XX` string the artifact target carries. The
    default nvcc is pinned rather than taken from PATH: sm_89 FP8 needs 12.4 or
    newer, and PATH may point at an older toolkit. The pin is `$CUDA_HOME/bin/nvcc`,
    else /usr/local/cuda-12.9.
    """
    if nvcc is None:
        nvcc = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda-12.9")) / "bin" / "nvcc"
    with TemporaryDirectory(prefix="lc0ex-cuda-") as directory:
        source_path = Path(directory) / "module.cu"
        cubin_path = source_path.with_suffix(".cubin")
        source_path.write_text(source, encoding="utf-8")
        command = [
            str(nvcc),
            "--cubin",
            f"--gpu-architecture={architecture}",
            "-std=c++17",
            "-O3",
            "--expt-relaxed-constexpr",
        ]
        for directory_path in include_directories:
            command.extend(("-I", str(directory_path)))
        command.extend(extra_arguments)
        command.extend((str(source_path), "-o", str(cubin_path)))
        subprocess.run(command, check=True)  # noqa: S603  # nvcc path is caller-pinned.
        return cubin_path.read_bytes()
