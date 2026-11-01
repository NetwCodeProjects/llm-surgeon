"""Command construction for llama.cpp tools at the pinned tag (verified flags only).

Verified against b11889 (d854d98): ``convert_hf_to_gguf.py`` options --outfile/--outtype
(f32,f16,bf16,q8_0,tq1_0,tq2_0,auto)/--mmproj; ``Qwen3_5ForConditionalGeneration`` is registered for
both the text model (QWEN35) and the vision projector (qwen3vl); ``llama-quantize [--imatrix f] in
out TYPE [nthreads]``; ``llama-imatrix -m -f -o [--chunks N] [-ngl N]``. Text and image generation in
stage 09 go through llama-server (adapters/llama_server.py).
"""

from __future__ import annotations

from pathlib import Path


def convert_argv(convert_cmd: list[str], model_dir: Path, outfile: Path, outtype: str) -> list[str]:
    return [*convert_cmd, str(model_dir), "--outfile", str(outfile), "--outtype", outtype]


def mmproj_argv(convert_cmd: list[str], model_dir: Path, outfile: Path, outtype: str) -> list[str]:
    # With --outfile the converter writes exactly that name (verified on b11889); callers pass an mmproj- name.
    return [*convert_cmd, str(model_dir), "--mmproj", "--outfile", str(outfile), "--outtype", outtype]


def imatrix_argv(imatrix_cmd: list[str], model: Path, calib: Path, out: Path, chunks: int | None, gpu_layers: str) -> list[str]:
    argv = [*imatrix_cmd, "-m", str(model), "-f", str(calib), "-o", str(out), "-ngl", gpu_layers]
    if chunks:
        argv += ["--chunks", str(chunks)]
    return argv


def quantize_argv(quantize_cmd: list[str], src: Path, dst: Path, qtype: str, imatrix: Path | None, threads: int) -> list[str]:
    """``threads`` comes from HostProbe.available_cpus (the cgroup quota), never os.cpu_count(): a Runpod B300
    Pod reported 344 host CPUs in a ~36-CPU container."""
    argv = [*quantize_cmd]
    if imatrix is not None:
        argv += ["--imatrix", str(imatrix)]
    return [*argv, str(src), str(dst), qtype.upper(), str(max(1, threads))]


def gguf_name(family: str, tag: str, qtype: str) -> str:
    return f"{family}-{tag}-{qtype.upper()}.gguf"
