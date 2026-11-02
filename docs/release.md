# Release run: the official abliterated Qwen3.8-27B

One run on one GPU produces the release, and the same run's record decides whether it may be published.
Everything that can be proven without that GPU is proven first by the local acceptance run
([acceptance.md](acceptance.md)); do not start the release run before that record says `accepted` for the
commit you are about to run.

## What gets published

| Repository (private on upload) | Contents | Created when `publish.formats` lists |
|---|---|---|
| `<namespace>/<name>` | the BF16 master (safetensors, tokenizer, processor and chat-template files) | `master` |
| `<namespace>/<name>-FP8` | FP8_BLOCK (compressed-tensors) for vLLM and other FP8 runtimes | `fp8` |
| `<namespace>/<name>-GGUF` | every published GGUF type and the F16 vision projector (`mmproj-…`) for llama.cpp | any GGUF type |

The release config publishes `[master, fp8, q4_k_m]`. `<namespace>` is the account you are logged in as
unless `publish.namespace` names an organization of yours; `<name>` defaults to `Qwen3.8-27B-abliterated`
(`<model>-abliterated`; `publish.repo_name` overrides it). Each repository gets a generated model card
(license and pipeline tag from the source card; results with the kind of each measurement; caveats; file
checksums; usage; how it was made; provenance) and an `llm-surgeon/` folder with the artifact manifests and
the release record. The source model's own `README.md` is replaced by the card. When the master is not
published, the export cards name the source model as their base and say the master was not published.

## Publishing is chosen when the run starts

`./main.sh --yes` on an interactive terminal asks right after the banner, before the first stage, so the run
never waits for you at the end. `--yes` does not answer this question; without a terminal (a script, a
dropped SSH session) nothing is asked and nothing is published.

```
Publish to Hugging Face when the run is ready? [y/N] y
  Not logged in: running `hf auth login` (paste a write token; it is not shown)
  Logged in as brokenmagic (write access ok)
  login valid until 2026-12-01
Repository name [Qwen3.8-27B-abliterated]:

Summary
  Publish      : BF16 master, FP8, GGUF Q4_K_M
  Repositories : brokenmagic/Qwen3.8-27B-abliterated, -FP8, -GGUF   (private, names free)
  Upload       : ~99.2 GB, after the release record says ready
Continue? [Y/n]
```

- **Login.** `$HF_TOKEN` (or the variable `model.hf_token_env` names), else the login `hf auth login` stored.
  With neither, the `hf` command of the control environment (`.venv`, pinned `huggingface_hub`) runs
  `hf auth login`; nothing is installed. A read-only token, or a fine-grained one without `repo.write` for
  the namespace, is refused here. OAuth logins report no token role, so their write access is confirmed when
  the repositories are created; their expiry date is shown, with a warning when it is less than two days away.
- **Formats.** With `publish.formats: null` you pick them by bit level, then by format inside each level
  (Enter accepts the defaults in brackets, which come from the config):

  ```
  Which precisions? (one or more, separated by commas)
    1) 16-bit   BF16 master (the abliterated model itself; transformers)   ~54.0 GB
    2) 8-bit    2 formats
    3) 6-bit    Q6_K (very close to 8-bit quality)                          ~22.1 GB
    4) 5-bit    4 formats
    5) 4-bit    6 formats
    6) 3-bit    7 formats
    7) 2-bit    6 formats
    8) 1-bit    2 formats
  Choice [1,2,5]: 1,2,5

  4-bit: which formats?
    1) Q4_K_M   GGUF         the common default; good quality for the size          ~16.5 GB
    2) Q4_K_S   GGUF         a little smaller, a little lower quality               ~15.5 GB
    3) IQ4_XS   GGUF         smallest 4-bit; best with an importance matrix         ~14.3 GB
    ...
  Choice [1]: 1,3
  ```

  The picked formats are what the run builds (`quantization.formats`) and publishes. Sizes are estimates
  from the model's parameter count (`src/llm_surgeon/formats.py`). A format that could not pass the release
  check here is refused with the reason: FP8 when its loader test is off (`verify_backend: none`), types that
  need an importance matrix while `quantization.gguf.imatrix_dataset` is unset. The ternary types (TQ1_0,
  TQ2_0) are not offered. With `--quantizations` on the command line the picker is skipped. When the picked
  formats differ from the config file, an interrupted run is continued with `--resume RUN_ID` (an identical
  config is what auto-resume matches).
- **Checked before the first stage:** the login and its permissions, that no repository of that name holds
  files or is public (on a terminal you are asked for another name), and what would make the release record
  fail regardless of the run: qualification off, a selection profile other than `qualified`, the comparison
  off, uncommitted code changes. Edits under `configs/` do not count as uncommitted code: the run stores its
  resolved config, and its profiles and templates are hashed.
- **Stored with the run.** The answers become the run's `publish.*` settings (operational keys: they change no
  stage key). `--resume` keeps them and checks that the login still works. `publish.enabled: true` publishes
  without asking (also without a terminal, with the config's or the default choices); `false` never asks.

At the end of a run that opted in, the release record is computed and, only if it says `ready`, the
repositories are created private and uploaded. Otherwise nothing is uploaded and the run exits with the
reason (`E_PUBLISH_NOT_READY`, exit 9); the stages' own results are unaffected.

## The release record

Publishing computes it and refuses to upload unless its verdict is `ready`. Only the exports being published
are judged; an export that is built but not published does not block the others:

| Check | Requirement |
|---|---|
| `run_complete` | every stage succeeded; no partial exports (`execution.allow_partial_exports: false`) |
| `full_size_transform` | stage 05 transformed the configured model |
| `promotion` | the master is `promoted:qualified`: every required gate passed |
| `export_loading` | every export loaded and generated in its runtime, FP8 natively (`representative`) |
| `comparison` | each export passed its comparison with the master |
| `code_revision` | the run recorded a git commit without uncommitted changes (edits under `configs/` excepted) |

It is written to `runs/<run>/reports/release-record.{json,md}`.

## Host

- One B300 or B200 Pod. The `runpod/pytorch` CUDA 13.2 image works (its nvcc in `/usr/local/cuda` targets
  `sm_103`; the pipeline finds it off the PATH).
- Workspace on the container (root) disk, ≥ 500 GB: source, master, BF16 GGUF intermediate, FP8, Q4_K_M,
  tools and compile caches. The container disk is wiped when a Runpod Pod stops: copy out first (step 5).
- A Hugging Face login with write access: answered at the start of step 2 (`hf auth login` runs if there
  is none), or `HF_TOKEN` exported in the shell. The token reaches the upload process through its
  environment; this project never writes it to disk (`hf auth login` keeps it in Hugging Face's own token file).
- Run step 2 inside `tmux`, so a dropped SSH connection does not end it.

## Commands

```bash
git clone https://github.com/NetwCodeProjects/llm-surgeon && cd llm-surgeon
git checkout <the accepted commit> && git status --short      # nothing listed
export WS=/root/ws

./main.sh --output-dir $WS --setup-only --yes                 # 1. tools (clean install on a new Pod)
tmux new -s release
./main.sh --output-dir $WS --yes                              # 2. answer y: the release run, published when ready
python3 scripts/acceptance_record.py --kind release --workspace $WS --run <run id>   # optional: the record alone
```

Step 2 is 50 Heretic trials within 180 minutes, then qualification, promotion, FP8 and Q4_K_M exports,
loading in vLLM (native FP8) and `llama-server` (text and image), and the comparison with the master; then
the upload.

The upload re-hashes every staged file against its manifest, uploads with `huggingface_hub`'s resumable
`upload_large_folder`, then lists each repository and compares every file: the sha256 of large files, the
git blob id of small ones. It refuses a repository that is public or already holds other files. Each
published artifact gets a `publication` assessment record (repository, revision, result);
`reports/publication.json` lists every repository.

**Publishing a finished run later** (answered no, no terminal, or an interrupted upload):

```bash
./main.sh --output-dir $WS --publish <run id>       # the plan; on a terminal it then asks "Upload now?"
./main.sh --output-dir $WS --publish <run id> --confirm-publish <token from the plan> --yes   # without a terminal
```

The namespace, name and formats are the run's `publish.*` choices (`--hf-namespace`, `--repo-name` override
them); the namespace falls back to the login. Running it again continues an interrupted upload.

3. **Copy out** before stopping the Pod: `$WS/runs`, `$WS/manifests`, `$WS/assessments`, `$WS/exports`.

4. **Review on Hugging Face** (repositories are private): the model cards, the file lists, then make them
   public in each repository's settings. The pipeline never does that.

## If the record is not ready

The upload is refused with the failing checks (`E_PUBLISH_NOT_READY`, exit 9). Nothing is published; the
record names the evidence. A failed gate on the 27B is a property of that run (for example no eligible
trial, or an export that lost answers in the comparison), not something to override for a release.
