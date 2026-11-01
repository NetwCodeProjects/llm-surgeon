# Acceptance run (v1)

One bounded run of Qwen/Qwen3.8-27B on one B300 (a B200 works the same way) that exercises, on real
hardware, everything the mocked tests can only simulate. `scripts/acceptance_record.py` turns it into
`acceptance-record.json` and `acceptance-record.md`, tied to the git commit, config hash, profile hashes
and tool versions. v1.0.0 is tagged on a record whose verdict is `accepted`.

| Check (record id) | What the run must show | Evidence the script reads |
|---|---|---|
| `clean_setup` | `--setup-only` on an empty workspace installs every component | stage 01 receipt: `component_actions` all `installed` |
| `tool_reuse` | a second `--setup-only` reuses every component | stage 01 receipt: all `reused` |
| `full_size_transform` | the 27B model loads and Heretic transforms it | stage 05 receipt, `metrics/candidate.json`, `--expect-model` |
| `budget_export_only` | the search is stopped by its budget, then export-only adds no trial | study history: a `budget_stopped` completion, export with the same trial set before and after |
| `resume_after_interrupt` | an interrupted stage completes after `--resume` | `logs/events.jsonl` (two starts of one stage, the last succeeded), `receipts/history/` |
| `promotion` | the decision table promotes the master as `promoted:qualified` | stage 07 receipt, `metrics/selection.json` |
| `export_loading` | every export loads and generates in its runtime (FP8 natively) | `metrics/verification.json` |
| `comparison` | the master/export comparison is measured and passes the export gates | `metrics/comparison.json` |
| `code_revision` | the run recorded a clean git commit | `metrics/preflight.json` `code` |

Measured and reported, not judged: time per stage, setup and reuse time, peak GPU memory and utilization,
host memory in use, lowest free disk, the size of every artifact and of each workspace directory.

## Host

- One B300 (or B200) Pod. On Runpod the `runpod/pytorch` CUDA 13.2 image worked (2026-10-31): its nvcc
  13.2 can target `sm_103`; it is in `/usr/local/cuda`, off the PATH of SSH shells, and the pipeline
  finds it there.
- Workspace on the container (root) disk, not a network volume: ≥ 500 GB. The 27B run holds the
  source (≈ 56 GB), master (≈ 56 GB), BF16 GGUF intermediate (≈ 56 GB), FP8 (≈ 30 GB), Q8_0 (≈ 29 GB),
  IQ4_XS (≈ 15 GB), tools (≈ 25 GB) and compile caches. The container disk is wiped when a Runpod Pod
  stops: copy the results out first (step 7).
- `HF_TOKEN` is optional (the model is public).

## Commands

```bash
git clone https://github.com/NetwCodeProjects/llm-surgeon && cd llm-surgeon
git checkout <commit to accept> && git status --short   # tracked files must be unchanged
export WS=/root/ws CFG=configs/acceptance/b300.yaml     # WS must not exist yet (clean setup)

# 1. clean setup; note the run id it prints
./main.sh --config $CFG --output-dir $WS --setup-only --yes
# 2. setup again: every component is reused
./main.sh --config $CFG --output-dir $WS --setup-only --new-run --yes
# 3. the acceptance run: the search stops at max_runtime_minutes (60), export-only follows
./main.sh --config $CFG --output-dir $WS --new-run --yes
# 4. when "[08 quantize] starting" appears, press Ctrl-C once (a graceful stop; exit 130), then resume
./main.sh --config $CFG --output-dir $WS --resume <run id from step 3> --yes
# 5. the record (exit 0 only when accepted)
python3 scripts/acceptance_record.py --workspace $WS --run <step 3> --setup-run <step 1> \
    --reuse-run <step 2> --expect-model Qwen/Qwen3.8-27B
```

The record is written to `$WS/runs/<run>/reports/acceptance-record.{json,md}`. It can be rebuilt later
from a copy of the workspace: recorded host paths are rebased onto `--workspace`.

6. **Batch-size measurement** (after the record, not part of it). Heretic auto-tunes its batch size up to
   `max_batch_size` (template 128). On the B300 with Qwen3.5-2B its own test still doubled throughput
   at the cap (64: 2548 tokens/s, 128: 5007 tokens/s, "Chosen batch size: 128"), so the cap, not the GPU,
   set the batch; the GPU peaked at 76 % and 9 GB. Compare a few trials at two caps; a different cap is a
   different study, so each is a new run:

   ```bash
   for cap in 128 512; do
     ./main.sh --config $CFG --output-dir $WS --new-run --yes --through 5 \
         --max-trials 6 --set abliteration.n_startup_trials=6 --set abliteration.max_batch_size=$cap
   done
   ```

   Compare stage 05 seconds per trial and peak GPU memory in each run's `receipts/05-abliterate.json`
   (`resources`) and Heretic's "Trying batch size" / "Chosen batch size" lines in
   `logs/05-abliterate/heretic.stdout.log`. Record the
   result in the implementation log and, if a larger cap is faster, set it in the `qwen3_5` profile.

7. **Copy out** before stopping the Pod: `$WS/runs`, `$WS/manifests`, `$WS/assessments`, `$WS/exports`
   (and `$WS/cache/vllm`, `$WS/cache/flashinfer` to skip kernel compilation on the next Pod of the same
   GPU type), e.g. with `aws s3 cp --recursive` or Runpod Cloud Sync.

## Reading the result

- `failed` means the run produced evidence against the check; `unresolved` means the evidence is
  missing (a step was skipped, or a path is absent). Both block acceptance, and both are listed at the
  top of the record.
- A `failed` `budget_export_only` with "the search finished within its budget" means trials ran faster
  than expected: lower `abliteration.max_runtime_minutes` and repeat step 3.
- No eligible trial within the budget (`E_NO_ELIGIBLE_TRIAL`, raised when stage 05 selects a trial) is a
  real outcome for this configuration, not a pipeline failure; the record then shows `full_size_transform`
  failed and the later checks unresolved. Continue the same study with a larger budget
  (`--resume <run> --max-runtime-minutes 120`; the budget is operational, so no fork is needed); the
  budget stop and export-only are then exercised by the continued search.

The flagship run (`configs/pipeline.yaml`: 50 trials, 180 minutes) is the first use of the accepted
version, not part of the acceptance.
