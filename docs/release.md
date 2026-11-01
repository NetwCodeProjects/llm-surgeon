# Release run: the official abliterated Qwen3.8-27B

One run on one GPU produces the release, and the same run's record decides whether it may be published.
Everything that can be proven without that GPU is proven first by the local acceptance run
([acceptance.md](acceptance.md)); do not start the release run before that record says `accepted` for the
commit you are about to run.

## What gets published

| Repository (private on upload) | Contents |
|---|---|
| `<namespace>/<name>` | the BF16 master (safetensors, tokenizer, processor and chat-template files) |
| `<namespace>/<name>-FP8` | FP8_BLOCK (compressed-tensors) for vLLM and other FP8 runtimes |
| `<namespace>/<name>-GGUF` | Q4_K_M and the F16 vision projector (`mmproj-…`) for llama.cpp |

`<name>` defaults to `Qwen3.8-27B-abliterated` (`--repo-name` overrides it). Each repository gets a generated
model card (license and pipeline tag from the source card; results with the kind of each measurement;
caveats; file checksums; usage; how it was made; provenance) and an `llm-surgeon/` folder with the
artifact manifests and the release record. The source model's own `README.md` is replaced by the card.

## The release record

`--publish` computes it and refuses to upload unless its verdict is `ready`:

| Check | Requirement |
|---|---|
| `run_complete` | every stage succeeded; no partial exports (`execution.allow_partial_exports: false`) |
| `full_size_transform` | stage 05 transformed the configured model |
| `promotion` | the master is `promoted:qualified`: every required gate passed |
| `export_loading` | every export loaded and generated in its runtime, FP8 natively (`representative`) |
| `comparison` | each export passed its comparison with the master |
| `code_revision` | the run recorded a git commit without uncommitted changes |

It is written to `runs/<run>/reports/release-record.{json,md}`.

## Host

- One B300 or B200 Pod. The `runpod/pytorch` CUDA 13.2 image works (its nvcc in `/usr/local/cuda` targets
  `sm_103`; the pipeline finds it off the PATH).
- Workspace on the container (root) disk, ≥ 500 GB: source, master, BF16 GGUF intermediate, FP8, Q4_K_M,
  tools and compile caches. The container disk is wiped when a Runpod Pod stops: copy out first (step 5).
- `HF_TOKEN`: a Hugging Face token with write access, exported in the shell only for step 4. It is passed to
  the upload process through its environment and never written to disk.

## Commands

```bash
git clone https://github.com/NetwCodeProjects/llm-surgeon && cd llm-surgeon
git checkout <the accepted commit> && git status --short      # nothing listed
export WS=/root/ws

./main.sh --output-dir $WS --setup-only --yes                 # 1. tools (clean install on a new Pod)
./main.sh --output-dir $WS --yes                              # 2. the release run (configs/pipeline.yaml)
python3 scripts/acceptance_record.py --kind release --workspace $WS --run <run id>   # 3. optional: the record alone
./main.sh --output-dir $WS --publish <run id> --hf-namespace <you>                  # 4a. the plan; nothing uploaded
HF_TOKEN=... ./main.sh --output-dir $WS --publish <run id> --hf-namespace <you> \
    --confirm-publish <token from 4a> --yes                                         # 4b. upload and check
```

Step 2 is 50 Heretic trials within 180 minutes, then qualification, promotion, FP8 and Q4_K_M exports,
loading in vLLM (native FP8) and `llama-server` (text and image), and the comparison with the master.

Step 4b re-hashes every staged file against its manifest, uploads with `huggingface_hub`'s resumable
`upload_large_folder` (running the same command again continues an interrupted upload), then lists each
repository and compares every file: the sha256 of large files, the git blob id of small ones. It refuses a
repository that is public or already holds other files. Each published artifact gets a `publication`
assessment record (repository, revision, result); `reports/publication.json` lists all three.

5. **Copy out** before stopping the Pod: `$WS/runs`, `$WS/manifests`, `$WS/assessments`, `$WS/exports`.

6. **Review on Hugging Face** (repositories are private): the model cards, the file lists, then make them
   public in each repository's settings. The pipeline never does that.

## If the record is not ready

The upload is refused with the failing checks (`E_PUBLISH_NOT_READY`, exit 9). Nothing is published; the
record names the evidence. A failed gate on the 27B is a property of that run (for example no eligible
trial, or an export that lost answers in the comparison), not something to override for a release.
