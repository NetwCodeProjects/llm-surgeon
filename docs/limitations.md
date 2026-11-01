# Known limitations (v1)

## Verification status

| Path | Verified how | Not verified |
|---|---|---|
| CLI, config, planning, receipts, resume/fork, locks, manifests, partial exports, parallel stage 08, acceptance record | 540+ CPU unit and integration tests (fake engines; `tests/integration/test_acceptance_record.py` builds a record from a mocked acceptance run) | — |
| Heretic adapter (install, introspection, native config, headless run, study export) | Real pinned Heretic (ARA): tiny model on CPU (WSL2); Qwen3.5-2B on an A100 (2026-10-29) and a B300 (2026-10-31), full 00–10 runs (implementation log) | **Qwen3.8-27B** (memory, runtime, ARA quality): the acceptance run (docs/acceptance.md) |
| llama.cpp convert/quantize/verify | Real pinned build: CPU (WSL2), CUDA `sm_80` (A100), CUDA `sm_103` (B300); 2B GGUF exports q8_0/iq4_xs loaded on the B300; `llama-server` output byte-identical to `llama-completion -f` on a real Qwen3.5-0.8B GGUF (CPU) | `llama-server` on a GPU; 27B conversion time and size |
| FP8 (llm-compressor + vLLM) | FP8_BLOCK of Qwen3.5-2B loaded natively by vLLM 0.32.0 on a B300 (representative test); weight-only fallback on the A100 | 27B; `vllm_enforce_eager` start-up time |
| Lambda Stack profile | Unit tests with simulated dpkg/apt markers | A real Lambda Stack image |

No result for the 27B model is claimed until the acceptance record exists. `tests/gpu/` and
docs/acceptance.md contain the commands to obtain it.

## Functional limits

- **No compatibility resolver** (by design): unsupported architectures fail inside the tools; the
  platform reports their errors.
- **Exact reproduction of `coder3101/Qwen3.8-27B-heretic-ara` is impossible** (no reproduce.json,
  unreleased engine build). Approximate reproduction uses published, 4-decimal-rounded parameters and
  pinned defaults for unpublished settings.
- Heretic's merged export **drops `mtp.*` tensors** (multi-token-prediction head). The `qwen3_5` profile sets
  `selection.graft_missing_source_tensors: ['mtp.']`, which restores them verbatim from the source (they were
  not abliterated, so speculative-decoding acceptance may differ slightly). With `[]` the gap is recorded in
  the manifests instead.
- Heretic's merged save re-serializes `tokenizer_config.json`/`generation_config.json` (reported by stage 09).
- Heretic headless mode requires Linux (prompt_toolkit console issue on Windows).
- `trust_remote_code` models cannot run through Heretic non-interactively.
- Only Pareto-optimal trials can be exported headlessly (Heretic's `trial_index` indexes the front);
  a manual `--trial-id` naming a dominated trial is refused. One trial is exported per stage-05 execution;
  choosing another (fork) re-runs export only.
- A study recorded under another tool environment is not continued (`E_STUDY_ENVIRONMENT_CHANGED`); after
  re-provisioning, `--new-run` repeats the search.
- Budget enforcement is wall-clock + SIGINT; a trial in progress at the deadline is discarded by Heretic.
- `baseline_mode: local` and calibrated static FP8 are not implemented.
- FP8 smoke tests exercise text only; image inputs are verified only for GGUF (+mmproj) with a synthetic
  image and a "did it run" criterion (content correctness is recorded, not gated).
- Qualification is a smoke test (≈30 prompts, keyword refusal heuristic) — not a benchmark.
- The master vs export comparison is a 12-prompt smoke comparison (counts with denominators); a delta of one prompt is
  noise, not a measured regression. It compares runtimes as shipped (transformers for the baseline, vLLM or
  llama.cpp for the export), so a delta can come from the runtime as well as from quantization (contract gaps).
- The complete stage 09 comparison ran on a B300 with Qwen3.5-2B (vLLM 0.32.0 and `llama-completion`); the
  `llama-server` path that replaced `llama-completion` was checked on CPU only.
- vLLM's stage 09 check runs with `enforce_eager` by default (no CUDA graphs), unlike a server with vLLM defaults;
  recorded as a contract gap. vLLM and FlashInfer compile caches live in `<cache_root>`; on a new machine the first
  start still compiles kernels (about 215 s on a B300).
- Quantize-only runs with comparison enabled provision the `heretic` environment (transformers) to measure the source.
- Identical bytes registered by two runs share one global manifest; its lineage fields (`run_id`) are those of the latest
  registration. Assessment records are never lost: `workspace/assessments/<artifact_id>/` holds every run's records.
- GGUF fit on a 16 GB RTX A4000 is not guaranteed; tested context is recorded, not promised.
- No automatic reboot, driver install only on the ubuntu profile and only when explicitly allowed.
- Single host, single run per run directory; no distributed queue, HTTP API or uploads (non-goals).
- Stage 08 runs formats concurrently by default; on hosts with little RAM or slow disks set
  `quantization.parallel: false`.
- Resource figures are 15-second samples; host memory is `/proc/meminfo` and, inside a container, covers the
  whole host.
