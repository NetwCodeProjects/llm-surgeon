"""Upload staged folders to Hugging Face repositories and check every file (runs in the heretic env).

Reads a job written by llm_surgeon.publish: {private, repos: [{repo_id, folder, files: [{path, size_bytes,
sha256, git_sha1}]}]}. Per repository: refuse one that holds other content; create it with the job's
visibility, or set an existing (empty, or holding this upload's files) repository to it; upload with
``HfApi.upload_large_folder`` (resumable: an interrupted run continues where it stopped); then list the
repository and compare every planned file (sha256 for LFS/Xet files, the git blob id for small files). A public
repository whose check fails is switched to private, so nobody downloads a broken upload.
Writes {"repos": [{repo_id, verified, revision, private, mismatches, error}]}. Verified against
huggingface_hub 1.33.0 (signatures of upload_large_folder, list_repo_tree, create_repo, update_repo_settings,
model_info).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


def remote_files(api, repo_id: str) -> dict:
    return {e.path: e for e in api.list_repo_tree(repo_id, recursive=True, expand=True, repo_type="model") if hasattr(e, "blob_id")}


def matches(entry, planned: dict) -> bool:
    if entry.size != planned["size_bytes"]:
        return False
    if entry.lfs is not None:
        return entry.lfs.sha256 == planned["sha256"]
    return entry.blob_id == planned["git_sha1"]


def publish_repo(api, repo: dict, private: bool = True) -> dict:
    from huggingface_hub.errors import RepositoryNotFoundError

    repo_id, planned = repo["repo_id"], {f["path"]: f for f in repo["files"]}
    try:
        remote = remote_files(api, repo_id)
        exists = True
    except RepositoryNotFoundError:
        remote, exists = {}, False
    if exists:
        foreign = sorted(p for p, e in remote.items() if p != ".gitattributes" and (p not in planned or not matches(e, planned[p])))
        if foreign:
            return {"repo_id": repo_id, "verified": False, "error": f"the repository already holds other content: {foreign[:10]}"}
        if bool(api.model_info(repo_id).private) != private:
            api.update_repo_settings(repo_id, private=private, repo_type="model")
    else:
        api.create_repo(repo_id, private=private, repo_type="model", exist_ok=True)
    api.upload_large_folder(repo_id=repo_id, folder_path=repo["folder"], repo_type="model", private=private)
    remote = remote_files(api, repo_id)
    mismatches = sorted(p for p, f in planned.items() if p not in remote or not matches(remote[p], f))
    info = api.model_info(repo_id)
    result = {
        "repo_id": repo_id,
        "verified": not mismatches and bool(info.private) == private,
        "revision": info.sha,
        "private": info.private,
        "mismatches": mismatches,
    }
    if not result["verified"] and not info.private:
        api.update_repo_settings(repo_id, private=True, repo_type="model")
        result.update(private=True, error="the remote check failed, so the repository was switched to private")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    job = json.loads(Path(args.job).read_text(encoding="utf-8"))
    result: dict = {"repos": []}
    try:
        import huggingface_hub
        from huggingface_hub import HfApi

        result["huggingface_hub"] = huggingface_hub.__version__
        api = HfApi(token=os.environ["HF_TOKEN"])
        for repo in job["repos"]:
            try:
                result["repos"].append(publish_repo(api, repo, bool(job.get("private", True))))
            except Exception as exc:  # noqa: BLE001
                result["repos"].append({"repo_id": repo["repo_id"], "verified": False, "error": f"{type(exc).__name__}: {exc}"})
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()[-4000:]
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if result["repos"] and all(r.get("verified") for r in result["repos"]) else 1


if __name__ == "__main__":
    sys.exit(main())
