# Acceptance run (local, CPU)

Before the one GPU run that produces the release ([release.md](release.md)), the same pipeline runs end to
end with real tools on a small model of the same architecture on a CPU: Qwen/Qwen3.5-0.8B
(`Qwen3_5ForConditionalGeneration`, with a vision tower, like Qwen3.8-27B). The run proves every
behaviour of the release path that does not depend on the GPU, and `scripts/acceptance_record.py` turns it
into `acceptance-record.{json,md}` tied to the git commit, config hash, profile hashes and tool versions.
A commit is accepted when the record's verdict is `accepted`; only an accepted commit goes to the GPU.

| Check (record id) | What the run must show | Evidence the script reads |
|---|---|---|
| `clean_setup` | `--setup-only` on an empty workspace installs every component | stage 01 receipt: `component_actions` all `installed` |
| `tool_reuse` | a second `--setup-only` reuses every component | stage 01 receipt: all `reused` |
| `run_complete` | every stage succeeded, no partial exports | `status.json` |
| `full_size_transform` | Heretic transformed the configured model | stage 05 receipt, `metrics/candidate.json`, `--expect-model` |
| `budget_export_only` | the search is stopped by its budget, then export-only adds no trial | study history: a `budget_stopped` completion; the export kept the same trial set |
| `resume_after_interrupt` | an interrupted stage completes after `--resume` | `logs/events.jsonl`, `receipts/history/` |
| `promotion` | the real gates promote the master as `promoted:qualified` | stage 07 receipt, `metrics/selection.json` |
| `export_loading` | the GGUF export loads in `llama-server` and answers text and image prompts | `metrics/verification.json` |
| `comparison` | the master/export comparison is measured and passes the export gates | `metrics/comparison.json` |
| `code_revision` | the run recorded a clean git commit | `metrics/preflight.json` `code` |

What a CPU cannot prove, and where it is proven instead: native FP8 loading in vLLM. The local config sets
`quantization.fp8.verify_backend: none` (the FP8 export is still produced and re-hashed; the record lists
its loader as switched off by configuration). vLLM 0.32.0 loaded a native FP8 export on the B300 on
2026-10-31, and the release record requires it again for the 27B before anything is published.

The CPU run uses `configs/heretic/acceptance-cpu.toml`: `optimize.toml` with every prompt set cut to a
quarter and shorter responses, so a trial takes minutes. That changes how well the search can judge a
trial, not which code runs.

## Commands (WSL2 or Linux, no GPU needed)

```bash
git clone https://github.com/NetwCodeProjects/llm-surgeon && cd llm-surgeon
git checkout <commit to accept> && git status --short      # nothing listed
export WS=$HOME/acceptance-ws CFG=configs/acceptance/local-cpu.yaml   # WS must not exist yet

./main.sh --config $CFG --output-dir $WS --setup-only --yes            # 1. clean setup (note the run id)
./main.sh --config $CFG --output-dir $WS --setup-only --new-run --yes  # 2. reuse
./main.sh --config $CFG --output-dir $WS --new-run --yes               # 3. the run: the search stops at 45 minutes
#    4. when "[08 quantize] starting" appears, press Ctrl-C once (graceful stop, exit 130), then:
./main.sh --config $CFG --output-dir $WS --resume <run id from 3> --yes
python3 scripts/acceptance_record.py --workspace $WS --run <3> --setup-run <1> --reuse-run <2> \
    --expect-model Qwen/Qwen3.5-0.8B                                   # 5. exit 0 only when accepted
```

The record goes to `$WS/runs/<run>/reports/acceptance-record.{json,md}`. Recorded host paths are rebased onto
`--workspace`, so it can be rebuilt from a copy of the workspace.

6. **Publishing** is proven the same way: `--publish <run> --hf-namespace <you> --repo-name
   llm-surgeon-acceptance-test` uploads the 0.8B results to private test repositories and checks every file
   remotely ([release.md](release.md)); delete the test repositories on Hugging Face afterwards. The release
   check (`--kind release`) on the acceptance run reports `export_loading` as failed because its FP8 loader
   is switched off, so this step uses a run of the same config with `--quantizations q4_k_m`.

## Reading the result

- `failed` means the run produced evidence against the check; `unresolved` means the evidence is missing (a
  step was skipped, or a path is absent). Both block acceptance, and both are listed in the record.
- A `failed` `budget_export_only` with "the search finished within its budget" means trials were faster than
  expected: lower `abliteration.max_runtime_minutes` and repeat step 3.
- No eligible trial within the budget (`E_NO_ELIGIBLE_TRIAL`, raised when stage 05 selects a trial) on this
  scaled-down search: resume with a larger budget (`--resume <run> --max-runtime-minutes 90`; the budget is
  operational, so no fork is needed). Expect it now and then: the KL gate (0.1) is calibrated for the full
  evaluation, and the CPU template's 25 prompts make KL noisy. Of the first two runs, one found an eligible
  trial at trial 3, the other none in 39 (closest 0.1004); see the implementation log, 2026-11-01.
