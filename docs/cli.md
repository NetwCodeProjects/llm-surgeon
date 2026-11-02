# CLI reference (`./main.sh`)

`main.sh` selects a Python (project `.venv` → stdlib for `--help`/`--version`/`--list-stages` →
system python with PyYAML for `--dry-run`/inspection → otherwise bootstraps `.venv` after confirmation)
and runs `python -m llm_surgeon`. `./main.sh --bootstrap --yes` only creates `.venv`.

## Flags

| Flag | Config key | Notes |
|---|---|---|
| *(no arguments)* | — | welcome screen (banner, detected GPU, where to start) and exit 0; works before `.venv` exists. A run needs at least one argument, e.g. `./main.sh --yes` |
| `--config PATH` | — | default `configs/pipeline.yaml`; `none` = built-in defaults |
| `--model-url REF` | `model.url` | repo ID or `https://huggingface.co/<org>/<name>`; normalized, never shell-interpreted; other hosts, queries, `/tree/…` paths rejected |
| `--model-path DIR` | `model.local_path` | exclusive with `--model-url`; provenance recorded as `supplied` |
| `--revision REV` | `model.revision` | branch/tag/commit; always resolved to a commit SHA in stage 02 |
| `--lambda-stack` / `--ubuntu` / `--env X` | `environment.profile` | mutually exclusive |
| `--from S` / `--through S` / `--only S` | — | stage number (`5`, `05`) or name (`abliterate`) |
| `--resume RUN_ID` | — | uses the stored resolved config; overrides that change it are rejected |
| `--fork` | — | with `--resume`: accept changes by creating a new run that reuses still-valid receipts |
| `--new-run` | `execution.resume` | never auto-resume |
| `--rerun` | — | re-execute selected stages even with valid receipts |
| `--dry-run` | `execution.dry_run` | no network, downloads, installs, subprocesses, GPU imports or writes |
| `--setup-only` | — | stages 00–01 for the components the config needs |
| `--quantize-only` | `abliteration.enabled=false` | quantizes the source (or `--model-path`) |
| `--quantizations LIST` | `quantization.formats` | `fp8` and verified GGUF types; unknown names → exit 14 |
| `--experiment {reproduce,optimize}` | `abliteration.experiment` | |
| `--allow-approximate-reproduction` | `abliteration.reproduce.allow_approximate` | |
| `--qualification {off,fast,extended}` | `evaluation.qualification` | |
| `--max-trials N` | `abliteration.max_trials` | |
| `--max-runtime-minutes N` | `abliteration.max_runtime_minutes` | operational (may change on resume) |
| `--trial-id N` | `abliteration.trial_id` | export Heretic trial N instead of the policy choice; resolved and recorded once; gates still apply ([study lifecycle](study-lifecycle.md)) |
| `--pareto-index N` | `abliteration.pareto_trial_index` | compatibility: resolved to a trial identifier at selection time; exclusive with `--trial-id` |
| `--set KEY=VALUE` | any key | repeatable; VALUE is parsed as YAML (`0.08`, `true`, `null`, `[a, b]`); unknown keys and wrong types → exit 3 (`E_CONFIG_INVALID`); a malformed pair, or a key given twice or also set by its dedicated flag → exit 2 (`E_SET_INVALID`); path keys resolve against the current directory. With `--resume`, a changed key needs `--fork` like any override |
| `--yes`, `-y` | `execution.yes` | acknowledges the compatibility notice and project-local downloads/installs; privileged host steps additionally need their `environment.allow_*` flag |
| `--output-dir DIR` | `storage.root` | |
| `--log-level debug\|info\|normal\|warning`, `--verbose`/`-v` (= info), `--quiet`/`-q` (= warning) | `logging.console_level` | what the terminal shows ([console output](#console-output)); the log files always get everything |
| `--print-config`, `--list-stages`, `--list-runs`, `--status RUN_ID` | — | inspection, no execution |
| `--cleanup [--confirm-cleanup TOKEN]` | — | always shows the plan first; deletes only `.partial` dirs, unregistered candidates and GGUF intermediates, and only with the token printed by the dry listing |
| `--publish RUN_ID [--hf-namespace NAME] [--repo-name NAME] [--visibility private\|public] [--confirm-publish TOKEN]` | — | uploads a finished run whose release record is `ready` to Hugging Face repositories ([release](release.md)). The visibility is part of the plan its token covers. Prints the plan, then asks "Upload now?" on a terminal; without one, the upload needs the plan's token and `--yes`. Namespace, name, formats and visibility default to the run's `publish.*` choices (the namespace then to the login). New runs ask at the start instead (`publish.enabled: ask`); `--yes` never answers that question |

Mutually exclusive / invalid combinations (exit 2): `--lambda-stack`+`--ubuntu`; `--model-url`+`--model-path`;
`--setup-only` with `--from/--through/--only`; `--only` with `--from/--through`; `--fork` without `--resume`;
`--resume`+`--new-run`; `--trial-id`+`--pareto-index`; `--revision` with `--model-path`; `--quantize-only` with abliteration-only flags.

Every run and dry run starts with the same banner (version and GPU type; a dry run does not query the
GPU, so it shows `not probed (dry run)`).

## Console output

The default view (`normal`) prints one line per stage with its result, and a live line for the stage that is
running:

```
Run 20261102T052210Z-4bc6eb (new)
[00 preflight]         ok          0.6s  ubuntu · NVIDIA GeForce GTX 1070 Ti (cc 6.1) · 883.0 GB free
[01 environment]       ok         37.4s  3 tools reused
[02 acquire]           ok          1.0s  Qwen/Qwen3.5-0.8B @ 2fc0636 · 1.7 GB verified
[03 published_baselines] skipped        evaluation.published_baselines not set
[04 prepare_engine]    ok         10.2s  optimize · ARA · seed 154518560
[05 abliterate] 29:38 · trial 10/200 (2:41) · ~15m left (budget) · none usable yet (closest: trial 8, 1/25, KL 0.1004)
```

- **The live line** (terminal only) is redrawn once a second: elapsed time and what the stage is doing. In
  stage 05 it shows Heretic's phase (loading the model, finding the batch size, exporting a trial), the trial
  it is on and how long that trial has run, the best trial within the selection gates so far ("usable"), or the closest one when none is,
  and the time left. The estimate is the average time of the finished trials times the trials left, capped
  by what is left of `abliteration.max_runtime_minutes` (then marked "(budget)"); until the first trial
  finishes the line says "estimate when trial N finishes". Other stages show their latest message; the upload shows bytes sent and files
  committed (huggingface_hub's status, every 60 s).
- **Without a terminal** (a file, a pipe, `nohup`) nothing is redrawn: each stage prints a "started" line,
  every finished Heretic trial a line (`[05 abliterate] trial 3: 17/100 refusals, KL 0.003 ✓ usable`), and
  its result line.
- **Warnings and errors** are printed in every view. The stages' other messages go to `logs/run.log` and
  `logs/events.jsonl` only.
- **`--verbose`** (`info`) prints every message as before plus a line per finished trial; `--log-level debug`
  adds each tool's own output; `--quiet` (`warning`) prints warnings, errors and the summary.
- Colour (green ok, yellow WARN, red FAILED) only on a terminal and never with `NO_COLOR` set.
  huggingface_hub's own progress bars are switched off (`HF_HUB_DISABLE_PROGRESS_BARS`); the stage line
  replaces them.

## Precedence

New run: **CLI (dedicated flags and `--set`) > YAML > profiles listed in the YAML > built-in defaults.** Resume: **stored resolved
config + CLI overrides**; the current YAML is ignored. `resolved-config.{json,yaml,sha256}` are written
per run.

## Exit codes

| Code | Meaning | Typical error IDs |
|---:|---|---|
| 0 | success | |
| 1 | internal error (bug) | `E_INTERNAL`, `E_ASSESSMENT_EXISTS`, `E_ASSESSMENT_INVALID` |
| 2 | usage | argparse errors, `E_MODEL_REF_INVALID` (from CLI), `E_SET_INVALID`, `E_UNKNOWN_STAGE`, `E_STAGE_RANGE`, `E_CLEANUP_TOKEN`, `E_PUBLISH_NAME`, `E_PUBLISH_TOKEN`, `E_PUBLISH_FORMAT` (`--publish`) |
| 3 | configuration | `E_CONFIG_INVALID`, `E_CONFIG_YAML`, `E_CONFIG_NOT_FOUND`, `E_MODEL_MISSING`, `E_TOML_INVALID`, `E_HERETIC_UNKNOWN_SETTING`, `E_HERETIC_TEMPLATE_CONTROL_KEY`, `E_REPRODUCE_SOURCE_MISSING`, `E_BASELINES_INVALID`, `E_PATH_ESCAPE`, `E_GATE_REQUIREMENT_MISSING`, `E_TRIAL_CHOICE_CONFLICT`, `E_TRIAL_NOT_FOUND`, `E_TRIAL_NOT_EXPORTABLE`, `E_HF_TOKEN_MISSING`, `E_HF_TOKEN_READ_ONLY`, `E_HF_LOGIN_FAILED`, `E_PUBLISH_NAMESPACE`, `E_PUBLISH_NAME`, `E_PUBLISH_FORMAT`, `E_PUBLISH_IMPOSSIBLE`, `E_PUBLISH_CONFLICT` (the last seven when a run starts) |
| 4 | preflight | `E_ENV_UNSUPPORTED_PLATFORM`, `E_ENV_AMBIGUOUS`, `E_PREFLIGHT_NO_GPU`, `E_DISK_FLOOR`, `E_CAPABILITY_UNSUPPORTED` |
| 5 | environment | `E_HOST_PREREQ`, `E_HOST_ACTION_FAILED`, `E_REBOOT_REQUIRED`, `E_INSTALL_STEP_FAILED`, `E_COMPONENT_VALIDATION_FAILED`, `E_COMPONENT_MISSING`, `E_UV_MISSING`, `E_PIN_MISMATCH`, `E_CONTROL_ENV_INCOMPLETE`, `E_EXECUTABLE_MISSING`, `E_NVCC_MISSING`, `E_NVCC_ARCH_UNSUPPORTED`, `E_HF_LOGIN_UNAVAILABLE` |
| 6 | acquisition | `E_HF_GATED`, `E_HF_NOT_FOUND`, `E_HF_API`, `E_DOWNLOAD_CORRUPT`, `E_DISK_INSUFFICIENT`, `E_LOCAL_MODEL_INVALID`, `E_IMATRIX_DATA_MISSING` |
| 7 | dependency / resume | `E_MISSING_DEPENDENCY`, `E_RESUME_CONFIG_CONFLICT`, `E_RUN_NOT_FOUND`, `E_COMPONENT_NOT_PROVISIONED`, `E_STUDY_ENVIRONMENT_CHANGED` |
| 8 | engine | `E_HERETIC_NO_OUTPUT`, `E_HERETIC_CONFIG`, `E_HERETIC_BUDGET_EXHAUSTED`, `E_HERETIC_TIMEOUT`, `E_HERETIC_INTROSPECT`, `E_HERETIC_STUDY_TOOL`, `E_HERETIC_SEARCH_FAILED`, `E_EXPORT_ADDED_TRIALS`, `E_EXPORT_TRIAL_MISMATCH`, `E_REPRODUCE_FAILED`, `E_QUALIFY_RUNTIME` |
| 9 | gates | `E_GATES_FAILED`, `E_GATES_UNMEASURED`, `E_NO_ELIGIBLE_TRIAL`, `E_MANUAL_TRIAL_INELIGIBLE`, `E_PUBLISH_NOT_READY` |
| 10 | quantization | `E_QUANT_TOOL_FAILED`, `E_FP8_FAILED`, `E_FP8_OUTPUT_INVALID`, `E_GGUF_CONVERT`, `E_MMPROJ_FAILED`, `E_IMATRIX_REQUIRED`, `E_EXPORT_EXISTS` |
| 11 | verification | `E_VERIFY_FAILED`, `E_MASTER_CORRUPT`, `E_INPUT_MUTATED`, `E_MANIFEST_INVALID`, `E_GGUF_CORRUPT`, `E_SAFETENSORS_CORRUPT`, `E_PUBLISH_CONFLICT`, `E_PUBLISH_STAGING`, `E_PUBLISH_FAILED` |
| 12 | locked | `E_LOCK_HELD`, `E_STUDY_LOCKED` |
| 14 | unsupported | `E_UNSUPPORTED_FORMAT`, `E_EXACT_REPRODUCTION_UNAVAILABLE`, `E_REPRODUCE_VERSION`, `E_MODIFIER_UNAVAILABLE`, `E_FP8_CALIBRATION_UNSUPPORTED`, `E_LOCAL_BASELINES_UNSUPPORTED` |
| 15 | confirmation required | `E_CONFIRMATION_REQUIRED`, `E_NOT_CONFIRMED` |
| 20 | partial success | some exports failed/unverified, others valid (`execution.allow_partial_exports`) |
| 130 | interrupted | `E_INTERRUPTED` |

Errors print as `ERROR [E_ID] message` plus a `hint:` line; the same structure is stored in the receipt's `error` field.
