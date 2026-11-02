# What happens when you run `./main.sh --yes`

A walk through one run of the release configuration (`configs/pipeline.yaml`: Qwen/Qwen3.8-27B, Heretic
optimize, FP8 and GGUF Q4_K_M), from the moment you press Enter to the summary table. Each stage lists what
it does in order, what it writes, and what stops the run. Times are from the real Qwen3.5-2B run on a B300
(2026-10-31); the 27B takes longer wherever the model is loaded, searched or written.

| Stage | In one line | B300, 2B |
|---|---|---|
| — | start: banner, configuration, choose or resume the run | < 1 s |
| 00 preflight | check the host: OS, GPU, disk, tools, what has been proven on this GPU | 0.1 s |
| 01 environment | install or reuse the pinned tools (Heretic, llm-compressor, vLLM, llama.cpp) | 285 s first time, 22 s after |
| 02 acquire | download the model at an exact commit and verify every file | 20 s |
| 03 published_baselines | record the model card's published metrics (no inference) | < 1 s |
| 04 prepare_engine | write and validate Heretic's settings | 8 s |
| 05 abliterate | search for the best abliteration, then save it | 366 s |
| 06 qualify | test the saved model on prompts Heretic never saw | 41 s |
| 07 select_master | apply the gates; promote the model to the immutable master | 4 s |
| 08 quantize | make the FP8 and GGUF exports | 101 s |
| 09 verify_exports | load every export in its runtime and compare it with the master | 409 s |
| 10 report_export | write the report, run summary and manifest index | < 1 s |

Everything a run writes is under the workspace (`--output-dir`, default `./workspace`): the run's own folder
is `runs/<run id>/` with `logs/`, `receipts/`, `configs/`, `metrics/`, `checkpoints/` and `reports/`.

## Start of every run (before stage 00)

1. **`main.sh` picks a Python.** It uses the project's `.venv`, creating it from the hash-locked
   requirements on the first run, and starts `python -m llm_surgeon`.
2. **Banner.** Version and the GPU that `nvidia-smi` reports.
3. **Configuration.** Built-in defaults, then the profiles listed in the file (`quantization/default.yaml`,
   `profiles/qwen3_5.yaml`), then `configs/pipeline.yaml`, then your command-line flags. The result is
   validated key by key; an unknown key or a wrong type stops here (exit 3).
4. **Which run.** A run with the same configuration that did not finish is resumed automatically; otherwise
   a new run id is made (e.g. `20261101T043848Z-a99cc9`). `--resume RUN_ID` continues a named run;
   `--fork` starts a new run that reuses its parent's still-valid work.
5. **Publishing (new runs).** The release config sets `publish.enabled: true` and `visibility: public`: only the
   login is checked (`hf auth login` runs when there is none), then the summary. With `publish.enabled: ask`
   (the built-in default) a terminal asks "Publish to Hugging Face when the run is ready?" and a person answers it; `--yes` does not, and without a terminal it is
   not asked (nothing is published). On yes: the login (`hf auth login` runs when there is none), the
   repository name, private or public, the formats by bit level when `publish.formats` is null, and the checks that would
   otherwise fail hours later (token permissions, taken names, formats that cannot pass the release check
   here, uncommitted code); then a summary to confirm. The answers are saved with the run
   ([release.md](release.md#publishing-is-chosen-when-the-run-starts)).
6. **Confirmation.** The compatibility notice and what the run may download and install. `--yes`
   acknowledges it; without a terminal and without `--yes` the run stops (exit 15).
7. **Run folder.** The resolved configuration is saved (`configs/resolved-config.json`, `.yaml` and its
   sha256), the run is locked so a second process cannot run it at the same time, and Ctrl-C is armed: the
   first press stops gracefully after the current step cleans up, a second press aborts.

Then, for **every stage** (on the terminal: a live line while it runs, then one result line; the details below
go to the logs):

- **Dependencies.** The stage needs the receipts of the stages it depends on (for example 07 needs 05 and
  06). A missing or failed dependency blocks it.
- **Reuse or run.** The stage's key is a hash of the configuration keys it reads and its dependencies'
  keys. If a successful receipt with the same key exists and its outputs still check out, the stage is
  reused (`reused` in the summary); otherwise the old receipt moves to `receipts/history/` and the stage runs.
- **Sampling.** While the stage runs, GPU memory and utilization, host memory and free disk are sampled
  every 15 s (`metrics/resources-<stage>.jsonl`).
- **Receipt.** At the end, `receipts/<id>-<name>.json` records the outcome (succeeded, failed,
  interrupted, skipped, blocked), duration, resource peaks, inputs and outputs, warnings and any error.
  Every message also goes to `logs/events.jsonl` and `logs/run.log`; each tool's own output goes to
  `logs/<stage>/<tool>.stdout.log` and `.stderr.log`.

## 00 preflight

Checks the machine before anything is installed or downloaded. It always runs again, because hosts change.

1. **Host facts:** OS and kernel, CPU and the CPUs this process may actually use (the container quota, not
   the host's count), RAM, every GPU (name, memory, driver, compute capability, ECC errors), the CUDA
   toolkit (`nvcc`, also found outside PATH in `/usr/local/cuda`) and the GPU architectures it can compile
   for, and the version and git commit of the code that is running.
2. **Profile:** `lambda-stack` when Lambda's packaging markers are present (tools come preinstalled and are
   reused), otherwise `ubuntu` (missing tools are installed, system packages only when allowed).
3. **GPU required?** Heretic, FP8 and vLLM need an NVIDIA GPU; without one the run stops (exit 4).
4. **Disk:** free space against `environment.min_free_disk_gb`, and an estimate of what the run will write
   once the model size is known. Network or ephemeral storage gets a warning: copy results out before the
   machine stops.
5. **Capability findings:** what is unsupported (stops the run), likely too small (warning) or not yet
   proven on this GPU (warning). For example, vLLM's FP8 check on a GPU without native FP8 is recorded as
   not representative.

Writes `metrics/preflight.json`.

## 01 environment

Makes sure every tool the run needs exists in its own pinned environment under `workspace/tools/`.

1. **Host prerequisites:** git, cmake, a C++ compiler, make, `nvidia-smi`, `nvcc`. Missing build tools are
   installed only with `environment.allow_system_packages: true`; otherwise the run stops with the command
   to fix it.
2. **Components**, each with a fingerprint of its pins (version, commit, CUDA backend, Python):
   - **heretic:** a virtual environment with Heretic at its pinned commit and the matching PyTorch build.
     It also runs qualification and the comparison baseline (transformers).
   - **fp8:** llm-compressor and compressed-tensors.
   - **vllm:** vLLM, for the FP8 check in stage 09.
   - **llama_cpp:** llama.cpp built from source at its pinned tag, with CUDA for the detected GPU
     architecture (`sm_103` on a B300), plus the GGUF converter's environment. Builds `llama-quantize`,
     `llama-imatrix` and `llama-server`.
3. **Install or reuse:** a component whose fingerprint matches an existing install is reused; otherwise it
   is installed fresh (the first run on a new machine takes about 5 minutes for all four).
4. **Validation, every run:** imports, `heretic --help`, `llama-server --version`, and a small matrix
   multiply on the GPU in each Python environment. A tool that installed but cannot use this GPU stops here.

Writes `tools/components.json` and `metrics/components.json` (versions of everything installed).

## 02 acquire

Gets the exact model and every input a later stage will need, before any GPU time is spent.

1. **Pin the revision:** asks Hugging Face what commit `model.revision` names (or the default branch) and
   records the full commit SHA. The run uses that commit even if the repository changes later.
2. **Download or reuse:** a snapshot already verified in this workspace is reused without re-hashing.
   Otherwise the files download through the Hugging Face cache, which resumes interrupted downloads.
3. **Verify every file:** sha256 for large files, the git blob id for small ones, against what Hugging Face
   publishes. A mismatch stops the run (`E_DOWNLOAD_CORRUPT`, exit 6).
4. **Protect the source:** the files are made read-only and the **source manifest** is written: every file
   with its size and sha256, the commit, and the tensor dtypes.
5. **Other inputs:** the prompt datasets that Heretic and qualification will use, at pinned dataset commits;
   a calibration text for an importance matrix if one is configured; `reproduce.json` for reproduce runs.

## 03 published_baselines

Copies the metrics that the model card publishes (from `configs/baselines/qwen3.8-27b.yaml`) into
`metrics/published_baselines.json` with their source, date and caveats. No model is run: published numbers
are recorded for comparison, never re-measured. Skipped when no baselines file is configured.

## 04 prepare_engine

Turns llm-surgeon's configuration into Heretic's own settings file and proves the installed Heretic accepts it.

1. **Inspect the installed Heretic:** its settings model and the modifiers it can import (ARA must be there).
2. **Build the native settings:** the template `configs/heretic/optimize.toml` (scorers, ARA, prompt
   datasets, batch-size tuning) plus the keys llm-surgeon controls: model path, number of trials (50),
   random startup trials, seed (derived from the run id and recorded), batch-size cap.
3. **Validate:** every key must exist in the installed Heretic; an unknown key stops the run (exit 3).
4. **Label the experiment:** `optimize` here (a `reproduce` run would be labelled exact or approximate).

Writes `checkpoints/heretic/prepared-config.toml` and `engine-plan.json`.

## 05 abliterate

The search. Heretic looks for the abliteration settings that remove the most refusals while changing the
model's behaviour on harmless prompts the least; llm-surgeon decides which trial becomes the candidate.

1. **Locate the study:** the study (Heretic's journal of trials) is identified by its search settings, the
   engine commit and the tool environment. A resumed run continues its own study; a study recorded under
   another environment is refused.
2. **Search** (Heretic, search only):
   - loads the model; tries batch sizes up to the cap and keeps the fastest;
   - measures the original model: refusals on its evaluation prompts (the 2B refused 97 of 100) and the
     reference token distributions on harmless prompts;
   - computes the refusal direction from harmful and harmless training prompts (ARA);
   - runs trials: each applies one set of parameters (which layers, how strongly) and measures refusals and
     the KL divergence from the original. The first trials are random, then Optuna's TPE proposes them.
   - stops after 50 trials or when the 180-minute budget is reached. At the budget llm-surgeon interrupts
     Heretic, which keeps every finished trial.
   Every step is appended to `study-history.jsonl` (intent before, completion after), so a crash at any
   point is reconciled on resume without repeating trials.
3. **Select:** among completed trials, keep those within the optimizer gates (refusal rate ≤ 0.2, KL ≤ 0.1);
   the policy `min_refusals_within_kl` picks the one with the fewest refusals. No eligible trial stops the
   run (`E_NO_ELIGIBLE_TRIAL`, exit 9); the error names the closest trial (the live line shows it during the
   search too) and gives the commands to use it or to search more, and says what to keep (`runs/RUN`) to
   continue on another machine.
4. **Export only:** Heretic runs again, restores that trial and saves the merged model, with zero new trials
   (checked against the study before and after).
5. **Candidate:** checks the saved checkpoint (config and every indexed shard), compares its tensors with the
   source (Heretic's merge drops the 15 `mtp.*` tensors), and records the candidate.

Writes `models/candidates/<run>/<candidate>/`, `metrics/candidate.json`, `metrics/heretic_study.json` and the
study history.

## 06 qualify

An independent check of the saved candidate, on prompts Heretic never optimized against.

1. **Reload** the candidate from disk with transformers (a real reload of what was saved).
2. **Benign prompts** with checkable answers (for example the capital of France, 17 × 23 = 391): does it
   still answer correctly?
3. **Held-out harmful prompts**, disjoint from Heretic's datasets: does it still refuse (keyword
   heuristic)?
4. **Integrity:** repetition loops, garbage output, leaked thinking tokens.
5. **Heretic's own scores** are copied alongside and labelled as optimizer metrics, never mixed with the
   independent ones.

About 30 prompts in total (the 2B: 10 of 10 benign correct, 0 of 20 held-out refused). Writes
`metrics/qualification.json`. This is a smoke test, not a benchmark; the report says so.

## 07 select_master

Decides whether the candidate becomes the master everything else is built from.

1. **Gates**, all required: reload succeeded; integrity; optimizer KL ≤ 0.1; benign pass rate ≥ 0.7;
   held-out refusal rate ≤ 0.35; optimizer refusal rate ≤ 0.2.
2. **Decision:** `promoted:qualified` when every gate passed. Anything else is either not promoted (the run
   stops, exit 9) or labelled `research` or `ungated` when you explicitly allowed it; never silently.
3. **Master:** the candidate's files go to `exports/bf16/<name>/` (hard-linked, no copy). The `qwen3_5`
   profile copies the 15 `mtp.*` tensors back from the source, unchanged. The precision is read from the
   saved tensors, every file is hashed, the folder is made read-only.
4. **Records:** the master manifest (lineage back to the source commit) and a `promotion` assessment record.

Writes `metrics/selection.json`.

## 08 quantize

Makes the deployment formats from the master. The master is fingerprinted before and after and must not change.

- **FP8** (beside the GGUF jobs): llm-compressor's data-free FP8_BLOCK (FP8 weights with 128×128 block
  scales, dynamic activations), skipping the layers the Qwen3.5 profile lists. Output checks: the
  quantization config and at least one safetensors file.
- **GGUF:**
  1. convert the master to a BF16 GGUF and the vision projector to an F16 `mmproj` GGUF (shared by every
     GGUF type);
  2. `llama-quantize` to Q4_K_M using every usable CPU; the projector is linked next to it.
- Each format is its own job with its own sub-receipt (`receipts/08-quantize/<format>.json`), written to a
  `.partial` folder and renamed only when complete, then hashed into a manifest whose parent is the master.

Writes `exports/fp8/<name>/`, `exports/gguf/<name>/` and `metrics/quantization.json`.

## 09 verify_exports

Proves each export works in the runtime it ships for, and compares it with the master.

1. **Files:** every file of the master and each export re-hashed against its manifest; tokenizer, chat
   template and processor files compared with the source; GGUF metadata read (architecture, tokenizer,
   chat template).
2. **Baseline:** the master answers the comparison suite (8 benign and 4 held-out prompts) with
   transformers. The prompts are rendered once with the model's chat template, and that exact text goes
   to every runtime.
3. **FP8 in vLLM:** a few loader prompts, then the comparison suite, in one load. On a GPU with native FP8
   this counts as representative.
4. **GGUF in `llama-server`:** one server per export, started with the projector: loader prompts, an image
   test (a red square: "What color is this image?"), then the comparison suite.
5. **Comparison:** per export, correct answers, refusals and integrity problems against the master, with
   denominators. The export gates check that benign correct does not drop and held-out refusals do not rise
   by more than 0.25, and that no new integrity problems appear. Default mode is `report`: results are
   recorded and warned about.
6. **Records:** a `verification` and a `comparison` assessment record per artifact; each manifest's
   status (validation, inference, evaluation) is re-derived from them.

Writes `metrics/verification.json` and `metrics/comparison.json`.

## 10 report_export

Writes `reports/report.md` (stages, artifacts, gates, metrics with their provenance, tool versions,
measured time, usage examples), `reports/run-summary.json` (the same, for programs) and
`reports/manifest-index.json` (every artifact of the run; Project B reads this).

## When the run ends

Each stage has printed one result line as it finished ([cli.md](cli.md#console-output); `--verbose` and `--quiet` add a
summary table of every stage's outcome and duration); then come the run's outcome and the report's path. The exit
code is 0 when everything succeeded. Otherwise it identifies the failing area (the table in
[cli.md](cli.md#exit-codes)); 130 means interrupted.

**Publishing**, when it was chosen at the start: the release record is computed for the formats being
published; only if it says `ready` are the repositories created (private or public, as chosen), uploaded and every file checked
remotely. Otherwise nothing is uploaded and the exit code says why (9 when the record is not ready). A run that
did not ask (no terminal) prints the command to publish it later: `./main.sh --publish RUN_ID`
([release.md](release.md)).

## When something goes wrong

- **A stage fails:** the run stops there (`execution.stop_on_failure`), the receipt holds the error id and
  a hint, and the report is still written, marked incomplete. Fix the cause and run the same command again,
  or `--resume RUN_ID`: finished stages are reused.
- **Ctrl-C:** once stops gracefully, killing tool processes and removing partial outputs. Twice aborts.
  `--resume RUN_ID` continues; in stage 05, finished trials are never repeated.
- **Where to look:** `runs/<run>/receipts/` (what happened and why), `logs/<stage>/` (each tool's own
  output), `logs/events.jsonl` (everything, in order), `metrics/` (the measurements).
