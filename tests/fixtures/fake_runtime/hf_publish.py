"""Fake Hugging Face upload: copies each staged folder into $FAKE_COUNTER_DIR/hub/<namespace>/<name>/.

Mirrors scripts/runtime/hf_publish.py: refuses a repository that holds other content, uploads, then checks
every planned file by hash. FAKE_HF_CORRUPT=<path in repo> corrupts that file after the upload (the remote
check must catch it). Never touches the network.
"""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

job = json.loads(Path(sys.argv[sys.argv.index("--job") + 1]).read_text(encoding="utf-8"))
out = Path(sys.argv[sys.argv.index("--out") + 1])
hub = Path(os.environ["FAKE_COUNTER_DIR"]) / "hub"
assert os.environ.get("HF_TOKEN"), "the token reaches the upload process through the environment"
result = {"repos": []}
for repo in job["repos"]:
    target = hub / repo["repo_id"]
    planned = {f["path"]: f for f in repo["files"]}
    existing = {p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()} if target.exists() else set()
    foreign = sorted(p for p in existing if p not in planned)
    if foreign:
        result["repos"].append({"repo_id": repo["repo_id"], "verified": False, "error": f"the repository already holds other content: {foreign}"})
        continue
    for f in repo["files"]:
        (target / f["path"]).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(repo["folder"]) / f["path"], target / f["path"])
    corrupt = os.environ.get("FAKE_HF_CORRUPT")
    if corrupt and (target / corrupt).exists():
        (target / corrupt).write_bytes(b"corrupted")
    mismatches = sorted(p for p, f in planned.items() if hashlib.sha256((target / p).read_bytes()).hexdigest() != f["sha256"])
    result["repos"].append({"repo_id": repo["repo_id"], "verified": not mismatches, "revision": "f" * 40, "private": True, "mismatches": mismatches})
out.write_text(json.dumps(result), encoding="utf-8")
sys.exit(0 if all(r["verified"] for r in result["repos"]) else 1)
