# LLM Surgeon (Abliterate and Quantize)

[![Version](https://img.shields.io/badge/version-1.0.0-blue)](pyproject.toml)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20WSL2-lightgrey?logo=linux&logoColor=white)](docs/deployment-ubuntu.md)
[![GPU verified](https://img.shields.io/badge/GPU%20verified-B300%20%7C%20A100-76B900?logo=nvidia&logoColor=white)](docs/limitations.md)
[![Last commit](https://img.shields.io/github/last-commit/NetwCodeProjects/llm-surgeon)](https://github.com/NetwCodeProjects/llm-surgeon/commits/main)

[![Heretic](https://img.shields.io/badge/Heretic-171bb98-8A2BE2)](https://github.com/p-e-w/heretic/commit/171bb98b8b9a41c081631b5e678f7133128973e0)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-b11889-FF8800)](https://github.com/ggml-org/llama.cpp/releases/tag/b11889)
[![vLLM](https://img.shields.io/badge/vLLM-0.32.0-30A2FF)](https://github.com/vllm-project/vllm)
[![llm-compressor](https://img.shields.io/badge/llm--compressor-0.14.0.1-555555)](https://github.com/vllm-project/llm-compressor)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A reproducible, configuration-first CLI platform that takes a user-selected Hugging Face model through
**abliteration** (with the [Heretic](https://github.com/p-e-w/heretic) engine), **lightweight
qualification**, preservation of an immutable **master checkpoint**, and **deployment quantization**
(FP8 via llm-compressor, GGUF via llama.cpp). Every artifact gets a versioned manifest with checksums
and lineage, so downstream consumers (Project B) never need to import this code.

> **Compatibility notice.** This platform does not automatically verify that your model architecture
> is supported by the selected abliteration engine or quantization backend. Verify compatibility with
> the engines' official documentation before execution. Unsupported models or formats may fail during
> model loading, transformation, or export. The platform manages execution, dependencies, logs, and
> artifacts but does not guarantee compatibility with every model.

## What it does

| Stage | ID | Result |
|---|---|---|
| preflight | 00 | OS/GPU/driver/RAM/disk facts, profile detection (Lambda Stack vs Ubuntu), disk & VRAM warnings |
| environment | 01 | Pinned, validated tool environments (Heretic, llm-compressor, vLLM, llama.cpp build) |
| acquire | 02 | Model revision pinned to a commit SHA, sha256-verified download, immutable source manifest |
| published_baselines | 03 | Model-card metrics with provenance — **no** reference-model inference |
| prepare_engine | 04 | Native Heretic config generated and validated against the *installed* Heretic |
| abliterate | 05 | Budgeted, resumable headless Heretic run → candidate checkpoint |
| qualify | 06 | Reload + held-out refusal / benign / integrity checks (optimizer metrics labelled separately) |
| select_master | 07 | Gates → immutable master (actual precision recorded) |
| quantize | 08 | FP8 / GGUF exports, each an independent resumable job (run concurrently by default) |
| verify_exports | 09 | Re-hash, preservation checks, real loader tests (vLLM for FP8, one `llama-server` per GGUF), master vs export comparison |
| report_export | 10 | Manifests, run summary, report, usage examples |

Each stage step by step, with what it writes and what stops it: [docs/run-walkthrough.md](docs/run-walkthrough.md).

Pinned, verified upstream versions (see [docs/implementation-log.md](docs/implementation-log.md)):
Heretic `171bb98b` (master; ARA + headless mode exist only after v1.4.0), llama.cpp `b11889`,
llm-compressor `0.14.0.1`, vLLM `0.32.0`.

## Status

Every stage has run end to end with Qwen3.5-2B on a B300 (Runpod, 2026-10-31), including native FP8 in
vLLM and a CUDA `sm_103` llama.cpp build; an earlier A100 run (Lambda, 2026-10-29) found the environment
issues fixed since. Each commit is accepted locally first: a real run of Qwen3.5-0.8B on a CPU
([docs/acceptance.md](docs/acceptance.md)). The release is one Qwen3.8-27B run on a B300, published only if
its release record says `ready` ([docs/release.md](docs/release.md)). What is and is not verified is listed
in [docs/limitations.md](docs/limitations.md).

## Requirements

- Ubuntu 22.04/24.04 (Lambda Stack image or plain Ubuntu). `--help` and `--dry-run` also work elsewhere.
- Python ≥ 3.10. Either `uv`, or `python3-venv` for the hash-locked bootstrap. PyYAML (`python3-yaml`, preinstalled on most Ubuntu images) for `--dry-run` before bootstrapping.
- NVIDIA GPU + driver for Heretic / FP8 / vLLM (one B200 or B300 is the reference target; native FP8 checks need compute capability ≥ 8.9, an A100 runs a recorded, non-representative fallback). GGUF-only quantization works on CPU.
- git, cmake, a C++ compiler for the llama.cpp build (stage 01 can `apt-get` them with `environment.allow_system_packages: true` + `--yes`).
- Disk: roughly 3–4× the BF16 model size for the full Qwen3.8-27B pipeline (≈ 55 GB source, candidate, GGUF intermediate, exports). Preflight and acquire check real numbers.

## Five-minute quick start

```bash
git clone https://github.com/NetwCodeProjects/llm-surgeon.git && cd llm-surgeon
./main.sh                                                   # welcome screen: banner, detected GPU, where to start
./main.sh --help
./main.sh --config configs/pipeline.yaml --dry-run          # zero side effects: plan, components, disk estimate
./main.sh --setup-only --yes                                # bootstrap .venv + provision pinned tool environments
./main.sh --yes                                             # full pipeline with configs/pipeline.yaml
```

![The welcome screen of ./main.sh: the LLM Surgeon banner, version, the detected GPU, and where to start (review the config, preview with --dry-run, help, run with --yes)](docs/images/start.png)

`./main.sh` without `--yes` asks for confirmation on a terminal and refuses (exit 15) when non-interactive.

### One B200 on Lambda Cloud (Lambda Stack)

```bash
export HF_TOKEN=...                       # optional for public repos; never written to disk by the platform
./main.sh --lambda-stack --dry-run
./main.sh --lambda-stack --setup-only --yes
./main.sh --lambda-stack --yes            # Qwen/Qwen3.8-27B, optimize 50 trials / 180 min, fp8 + q4_k_m
```

The ARA reference (`coder3101/Qwen3.8-27B-heretic-ara`) can only be reproduced **approximately**; see
[examples/qwen-b200.yaml](examples/qwen-b200.yaml) and [docs/heretic-adapter.md](docs/heretic-adapter.md).

### One B300 on Runpod

```bash
export WS=/root/ws                        # the container (root) disk: faster than a network volume, wiped when the Pod stops
./main.sh --output-dir $WS --setup-only --yes
./main.sh --output-dir $WS --yes
aws s3 cp --recursive $WS/exports s3://<bucket>/exports   # copy results out before stopping the Pod
```

The `runpod/pytorch` CUDA 13.2 image works: its nvcc (in `/usr/local/cuda`, off the SSH PATH; the
pipeline finds it) can target the B300's `sm_103`. The release procedure is in [docs/release.md](docs/release.md).

### Publish to Hugging Face

`./main.sh --yes` on a terminal asks before the first stage whether to publish the run when it is ready. On
yes it logs you in if needed (`hf auth login`, from the project's own environment), asks the repository name
and its visibility (private, or public straight away), and, unless `publish.formats` in the config already lists them, lets you pick the formats by bit level
(16-bit master, 8-bit FP8/Q8_0, … 4-bit Q4_K_M/IQ4_XS/…). Login problems and taken names stop the run
in its first minute, not after hours. `--yes` does not answer the question, and without a terminal nothing
is published.

```bash
./main.sh --publish RUN_ID          # a finished run later: the plan, then "Upload now?" on a terminal
```

Only a run whose release record is `ready` is uploaded (`<name>`, `<name>-FP8`, `<name>-GGUF`, as far as
their formats are published), with a generated model card and every file checked remotely; a public upload
that fails the check is switched to private ([docs/release.md](docs/release.md)).

### Plain Ubuntu

```bash
./main.sh --ubuntu --dry-run
./main.sh --ubuntu --setup-only --yes     # reports missing host packages; apt only with allow_system_packages
./main.sh --config examples/ubuntu-quantize-only.yaml --ubuntu --yes
```

## Run, resume, quantize-only

```bash
./main.sh --through 2 --yes                         # stop after acquisition
./main.sh --resume RUN_ID                           # continue; valid receipts are reused
./main.sh --resume RUN_ID --from 5 --through 6 --rerun
./main.sh --resume RUN_ID --fork --quantizations fp8  # config change → new run reusing still-valid work
./main.sh --model-url org/model --quantize-only --quantizations fp8,iq4_xs,iq3_s --yes
./main.sh --model-path /data/model --quantize-only --yes
./main.sh --experiment optimize --max-trials 50 --max-runtime-minutes 120 --yes
./main.sh --status RUN_ID ; ./main.sh --list-runs ; ./main.sh --cleanup
```

More in [docs/cli.md](docs/cli.md) and [examples/resume.md](examples/resume.md).

## Outputs

```text
workspace/
├── cache/{hf,uv,vllm,flashinfer,…}       HF cache (verified, read-only source blobs), uv cache, kernel compile caches
├── tools/{components.json,envs,heretic,llama.cpp}
├── models/source/<org>--<name>/<sha>.json   verified source records (+ link to the snapshot)
├── models/candidates/<run>/<candidate>/     Heretic exports
├── models/selected/<run>/master.json        pointer to the promoted master
├── models/gguf-intermediate/                re-creatable conversion intermediates
├── manifests/<artifact_id>.json             every manifest ever produced
├── assessments/<artifact_id>/               append-only verification, comparison and promotion records
├── exports/{bf16,fp8,gguf}/<name>/          deliverables (+ <name>.manifest.json sidecars)
└── runs/<run>/{logs,receipts,configs,metrics,checkpoints,reports}/
```

`runs/<run>/reports/report.md` summarizes stages, artifacts (integrity, loader test, modalities
actually tested), gates, metric provenance, tool versions and measured time.

## Documentation

[run walkthrough](docs/run-walkthrough.md) · [architecture](docs/architecture.md) · [stages](docs/stages.md) · [cli](docs/cli.md) ·
[configuration](docs/configuration.md) · [Lambda](docs/deployment-lambda.md) · [Ubuntu](docs/deployment-ubuntu.md) ·
[Heretic adapter](docs/heretic-adapter.md) · [study lifecycle](docs/study-lifecycle.md) · [quantization](docs/quantization.md) · [evaluation](docs/evaluation.md) · [assessments](docs/assessments.md) ·
[artifact schema](docs/artifact-schema.md) · [security](docs/security-and-trust.md) · [troubleshooting](docs/troubleshooting.md) ·
[limitations](docs/limitations.md) · [acceptance](docs/acceptance.md) · [release](docs/release.md) · [implementation log](docs/implementation-log.md) · [original specification](docs/PROJECT_A_AGENT_HANDOFF.md)

## Development

```bash
uv sync --frozen            # or ./main.sh --bootstrap --yes, then .venv/bin/pip install -r requirements/dev.txt
.venv/bin/python -m pytest  # CPU-only unit + mocked integration tests
python scripts/generate_schema.py --check
REALTOOLS_WS=/tmp/ws python tests/realtools/provision_cpu.py && REALTOOLS_WS=/tmp/ws pytest -m realtools tests/realtools
pytest -m gpu tests/gpu     # on the B200/B300 release host; other GPUs need GPU_ALLOW_OTHER=1
python3 scripts/acceptance_record.py --workspace WS --run RUN --setup-run A --reuse-run B   # docs/acceptance.md
python3 scripts/acceptance_record.py --kind release --workspace WS --run RUN                 # docs/release.md
```

Project B (coding-capability research) is intentionally **not** part of this repository.
Third-party licenses: [LICENSE-THIRD-PARTY.md](LICENSE-THIRD-PARTY.md) (Heretic is AGPL-3.0).
