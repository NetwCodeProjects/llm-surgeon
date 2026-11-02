"""Upload a ready run's master and exports to Hugging Face repositories (docs/release.md).

Two ways in: a run that opted in at its start (llm_surgeon.publish_setup) uploads when it finishes, and
``--publish RUN`` uploads a finished run later. Never without the owner's consent: the run must pass its
release record (llm_surgeon.release_record), and ``--publish`` first prints the plan; it uploads after a "y"
on an interactive terminal, or with the plan's confirmation token and ``--yes``. Visibility is the owner's
choice (``publish.visibility``, private unless chosen otherwise) and is part of the plan the token covers;
a public upload whose remote check fails is switched to private. Files are
re-hashed against their manifests right before the upload and checked against the repository afterwards
(sha256 of LFS/Xet files, git blob id of small files). Each published artifact gets a ``publication``
assessment record.

Repositories: ``<namespace>/<name>`` (BF16 master), ``<name>-FP8``, ``<name>-GGUF`` (every GGUF type and
the vision projector), each only when one of its formats is published (``publish.formats``). ``<name>``
defaults to ``<source model name>-abliterated``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import threading
from pathlib import Path
from typing import Any

import yaml

from . import model_card
from . import release_record as R
from .artifacts import assessments as A
from .artifacts.manifest import ArtifactRegistry
from .console import Console
from .environment.components import ComponentManager, find_uv
from .errors import ExitCode, PipelineError
from .fsutil import atomic_write_json, file_inventory, git_blob_sha1, read_json, remove_tree, sha256_obj
from .stages.stage08_quantize import configured_export_roles
from .subprocesses import CommandRunner, child_env
from .telemetry import EventLogger, Redactor, utc_now

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
SUFFIX = {"master": "", "fp8": "-FP8", "gguf": "-GGUF"}


def repo_kind(role: str) -> str | None:
    if role == "master":
        return "master"
    if role == "export:fp8":
        return "fp8"
    if role.startswith("export:gguf:"):
        return "gguf"
    return None


def card_front_matter(readme: Path) -> dict[str, Any]:
    try:
        text = readme.read_text(encoding="utf-8")
    except OSError:
        return {}
    if not text.startswith("---"):
        return {}
    try:
        data = yaml.safe_load(text.split("---", 2)[1])
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def manifest_of(view: R.RunView, entry: dict[str, Any]) -> dict[str, Any]:
    path = view.local(entry.get("manifest"))
    if not (path and path.is_file()):
        path = view.workspace / "manifests" / f"{entry['artifact_id']}.json"
    return read_json(path)


def gather_facts(view: R.RunView, artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """What the model cards say, read from the run's own files."""
    source = artifacts["source"]["manifest"]
    front = card_front_matter(Path(view.local(source["artifact"]["uri"]) or source["artifact"]["uri"]) / "README.md")
    candidate = view.metric("candidate.json") or {}
    summary, selection = candidate.get("summary") or {}, candidate.get("selection") or {}
    qual = view.metric("qualification.json") or {}
    ind = qual.get("independent") or {}
    graded = ind.get("benign_graded") or 0
    responses = (ind.get("benign_prompts") or 0) + (ind.get("heldout_refusal_prompts") or 0)
    verification = view.metric("verification.json") or {}
    preflight = view.metric("preflight.json") or {}
    status = read_json(view.dir / "status.json")
    components = view.metric("components.json") or {}
    master_ext = artifacts["master"]["manifest"].get("extensions") or {}
    return {
        "source": {
            "repo": source["source_model"],
            "revision": source["source_revision"],
            "license": front.get("license"),
            "license_name": front.get("license_name"),
            "pipeline_tag": front.get("pipeline_tag"),
        },
        "optimizer": {
            "scores": [
                {"name": s["name"], "baseline": (s.get("baseline") or {}).get("md_display"), "score": (s.get("score") or {}).get("md_display")}
                for s in summary.get("optimizer_scores") or []
            ],
            "trials_complete": summary.get("trials_complete"),
            "trial": summary.get("exported_trial_index"),
            "engine_commit": (candidate.get("engine") or {}).get("commit") or "",
            "modifiers": candidate.get("modifiers") or [],
            "policy": selection.get("policy"),
            "constraints": selection.get("constraints") or {},
        },
        "qualification": {
            "heldout": f"{ind.get('heldout_refusal_count')}/{ind.get('heldout_refusal_prompts')}",
            "benign": f"{round((ind.get('benign_pass_rate') or 0) * graded)}/{graded}",
            "integrity": f"{len(ind.get('integrity_failures') or [])}/{responses}",
            "caveats": qual.get("caveats") or [],
        },
        "promotion": (view.metric("selection.json") or {}).get("outcome"),
        "comparison": view.metric("comparison.json") or {"baseline": {}, "exports": {}},
        "verification": {role: item.get("smoke") or {} for role, item in verification.items()},
        "grafted": len((master_ext.get("grafted_tensors") or {}).get("grafted") or []),
        "provenance": {
            "run_id": view.run_id,
            "version": (preflight.get("code") or {}).get("version"),
            "git_commit": (preflight.get("code") or {}).get("git_commit"),
            "config_hash": status.get("config_hash"),
            "tools": {name: c.get("versions") or {} for name, c in components.items()},
            "gpus": [g.get("name") for g in preflight.get("gpus") or []],
        },
    }


def stable_record(record: dict[str, Any]) -> dict[str, Any]:
    """The release record as uploaded: without the generation time and the workspace's directory sizes, which
    change between the plan and the confirmed upload (the token must stay the same)."""
    measurements = {k: v for k, v in record["measurements"].items() if k != "workspace_bytes"}
    return {**{k: v for k, v in record.items() if k != "generated_utc"}, "measurements": measurements}


def published_formats(config: dict[str, Any], formats: list[str] | None = None) -> list[str]:
    """``formats``, else the run's ``publish.formats``, else the master and every format the run built."""
    quant = config.get("quantization") or {}
    built = list(quant.get("formats") or []) if quant.get("enabled") else []
    return list(formats or (config.get("publish") or {}).get("formats") or ["master", *built])


def build_plan(
    workspace: Path, run_id: str, namespace: str, name: str | None, formats: list[str] | None = None, visibility: str = "private"
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The upload plan (what the confirmation token covers) and the generated files to stage."""
    for label, value in (("--hf-namespace", namespace), ("--repo-name", name)):
        if value is not None and not NAME_RE.match(value):
            raise PipelineError(ExitCode.USAGE, "E_PUBLISH_NAME", f"{label} {value!r} is not a valid Hugging Face name")
    run_config = R.RunView(workspace, run_id).config
    quant = run_config.get("quantization") or {}
    chosen = published_formats(run_config, formats)
    built = configured_export_roles(quant.get("formats")) if quant.get("enabled") else set()
    wanted = configured_export_roles([f for f in chosen if f != "master"])
    if wanted - built:
        raise PipelineError(ExitCode.USAGE, "E_PUBLISH_FORMAT", f"Run {run_id} did not build {sorted(wanted - built)}", hint=f"It built {sorted(built)}.")
    record = R.build_record(workspace, run_id, kind="release", roles=wanted)
    record_path = R.write_record(record, workspace / "runs" / run_id / "reports")
    if record["verdict"] != "ready":
        raise PipelineError(
            ExitCode.GATE,
            "E_PUBLISH_NOT_READY",
            f"Run {run_id} is not ready to publish: failed {record['failed']}, unresolved {record['unresolved']}",
            hint=f"See {record_path}. Only a complete, qualified run whose exports all loaded and compared cleanly is published.",
        )
    view = R.RunView(workspace, run_id)
    index = read_json(view.dir / "artifacts.json")["artifacts"]
    artifacts = {role: {**entry, "manifest": manifest_of(view, entry)} for role, entry in index.items()}
    facts = gather_facts(view, artifacts)
    base = name or f"{facts['source']['repo'].split('/')[-1]}-abliterated"
    base_repo = f"{namespace}/{base}"
    repos: dict[str, dict[str, Any]] = {}
    for role in sorted(artifacts):
        kind = repo_kind(role)
        if kind is None or (kind == "master" and "master" not in chosen) or (kind != "master" and role not in wanted):
            continue  # not chosen, or an export this run's configuration does not produce (inherited by a fork)
        repo = repos.setdefault(kind, {"repo_id": f"{base_repo}{SUFFIX[kind]}", "kind": kind, "roles": [], "artifact_ids": [], "files": {}})
        manifest = artifacts[role]["manifest"]
        repo["roles"].append(role)
        repo["artifact_ids"].append(manifest["artifact_id"])
        root = view.local(manifest["artifact"]["uri"])
        for f in manifest["artifact"]["files"]:
            if f["relative_path"] == "README.md":
                continue  # the source model's card; the generated card replaces it (the manifest copy still lists it)
            known = repo["files"].get(f["relative_path"])
            if known and known["sha256"] != f["sha256"]:
                raise PipelineError(ExitCode.VERIFY, "E_PUBLISH_CONFLICT", f"{f['relative_path']} differs between artifacts of {repo['repo_id']}")
            repo["files"][f["relative_path"]] = {
                "path": f["relative_path"],
                "size_bytes": f["size_bytes"],
                "sha256": f["sha256"],
                "source": str(root / f["relative_path"]),
            }
    generated: dict[str, dict[str, bytes]] = {}
    for kind, repo in repos.items():
        files = sorted(repo["files"].values(), key=lambda f: f["path"])
        extra = {
            f"llm-surgeon/manifest-{artifacts[r]['manifest']['artifact_id']}.json": json.dumps(artifacts[r]["manifest"], indent=2, sort_keys=True).encode()
            for r in repo["roles"]
        }
        extra["llm-surgeon/release-record.json"] = json.dumps(stable_record(record), indent=2, sort_keys=True).encode()
        card = model_card.render(kind, repo["repo_id"], base_repo if "master" in repos else None, repo["roles"], files, facts).encode()
        extra["README.md"] = card
        generated[repo["repo_id"]] = extra
        for path, data in extra.items():
            files.append({"path": path, "size_bytes": len(data), "sha256": sha256_bytes(data), "source": None})
        repo["files"] = sorted(files, key=lambda f: f["path"])
    plan = {"run_id": run_id, "private": visibility != "public", "repos": [repos[k] for k in ("master", "fp8", "gguf") if k in repos]}
    return plan, generated


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def plan_token(plan: dict[str, Any]) -> str:
    return sha256_obj(
        {"run": plan["run_id"], "private": plan["private"], "repos": [[r["repo_id"], [[f["path"], f["sha256"]] for f in r["files"]]] for r in plan["repos"]]}
    )[:16]


def print_plan(plan: dict[str, Any], token: str, namespace: str, name: str | None, workspace: Path, stream, command: bool = True) -> None:
    seen = "PRIVATE (only you)" if plan["private"] else "PUBLIC (anyone can download the files as they upload)"
    stream.write(f"Publish plan for run {plan['run_id']} (nothing uploaded). Every repository: {seen}.\n")
    for repo in plan["repos"]:
        total = sum(f["size_bytes"] for f in repo["files"])
        stream.write(f"\n  https://huggingface.co/{repo['repo_id']}  ({', '.join(repo['roles'])}; {len(repo['files'])} files, {model_card.human(total)})\n")
        for f in repo["files"]:
            stream.write(f"    {f['path']:<60} {model_card.human(f['size_bytes']):>10}\n")
    if not command:
        return
    flag = (f" --repo-name {name}" if name else "") + ("" if plan["private"] else " --visibility public")
    stream.write(
        f"\nTo upload exactly this plan (logged in, or with $HF_TOKEN set): ./main.sh --output-dir {workspace} --publish {plan['run_id']} "
        f"--hf-namespace {namespace}{flag} --confirm-publish {token} --yes\n"
    )


def stage(workspace: Path, plan: dict[str, Any], generated: dict[str, dict[str, bytes]]) -> dict[str, Path]:
    """One folder per repository: hard links to the artifact files plus the generated files, re-hashed."""
    folders = {}
    for repo in plan["repos"]:
        folder = workspace / "publish" / plan["run_id"] / repo["repo_id"].replace("/", "--")
        for f in repo["files"]:
            target = folder / f["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if f["source"] is None:
                target.write_bytes(generated[repo["repo_id"]][f["path"]])
                continue
            if target.exists() and target.stat().st_size == f["size_bytes"]:
                continue  # staged by an earlier, interrupted attempt; re-hashed below
            target.unlink(missing_ok=True)
            try:
                os.link(f["source"], target)
            except OSError:
                shutil.copy2(f["source"], target)
        actual = {e.relative_path: e for e in file_inventory(folder, exclude_names=(".cache",), workers=8)}
        wanted = {f["path"]: f for f in repo["files"]}
        extra = sorted(set(actual) - set(wanted))
        bad = [p for p, f in wanted.items() if p not in actual or actual[p].sha256 != f["sha256"] or actual[p].size_bytes != f["size_bytes"]]
        if bad or extra:
            raise PipelineError(ExitCode.VERIFY, "E_PUBLISH_STAGING", f"{repo['repo_id']}: staged files do not match the plan: {bad[:5]} {extra[:5]}")
        for f in repo["files"]:
            f["git_sha1"] = git_blob_sha1(folder / f["path"])
        folders[repo["repo_id"]] = folder
    return folders


def heretic_tools(workspace: Path, scripts_dir: Path, log_dir: Path, probe):
    runner = CommandRunner(Redactor(), threading.Event())
    manager = ComponentManager(
        tools_dir=workspace / "tools",
        runner=runner,
        logger=EventLogger(None, Redactor(), "warning"),
        log_dir=log_dir,
        uv_executable=find_uv(None),
        scripts_dir=scripts_dir,
        cache_root=workspace / "cache",
        probe=probe,
    )
    return runner, manager.get("heretic")


def publish(
    workspace: Path,
    run_id: str,
    namespace: str | None,
    name: str | None,
    confirm: str | None,
    yes: bool,
    services,
    stream=sys.stdout,
    visibility: str | None = None,
) -> int:
    """``--publish RUN``: the plan; then the upload after a "y" on a terminal, or with the plan's token and ``--yes``.
    Namespace, name and formats default to the run's ``publish.*`` choices; the namespace then to the login."""
    if not (workspace / "runs" / run_id / "configs" / "resolved-config.json").is_file():
        raise PipelineError(ExitCode.DEPENDENCY, "E_RUN_NOT_FOUND", f"No run {run_id} under {workspace / 'runs'}", hint="List runs with ./main.sh --list-runs.")
    run_config = R.RunView(workspace, run_id).config
    pub = run_config.get("publish") or {}
    token_env = (run_config.get("model") or {}).get("hf_token_env") or "HF_TOKEN"
    account = services.account_factory()
    hf_token = account.token(token_env)
    namespace = namespace or pub.get("namespace")
    if namespace is None:
        who = account.whoami(hf_token) if hf_token else None
        if who is None:
            raise PipelineError(
                ExitCode.CONFIG,
                "E_HF_TOKEN_MISSING",
                "No Hugging Face login to take the namespace from",
                hint="Pass --hf-namespace, or log in: .venv/bin/hf auth login",
            )
        namespace = who["name"]
    name = name or pub.get("repo_name")
    from .publish_setup import stored_visibility  # noqa: PLC0415

    plan, generated = build_plan(workspace, run_id, namespace, name, visibility=visibility or stored_visibility(pub.get("visibility")))
    token = plan_token(plan)
    interactive = confirm is None and services.interactive()
    if confirm is None:
        print_plan(plan, token, namespace, name, workspace, stream, command=not interactive)
        if not interactive:
            return int(ExitCode.OK)
        from .publish_setup import Prompter  # noqa: PLC0415

        stream.write("\n")
        if not Prompter(services, stream).yes_no("Upload now?", False):
            return int(ExitCode.OK)
    elif confirm != token:
        raise PipelineError(
            ExitCode.USAGE,
            "E_PUBLISH_TOKEN",
            "Publish token does not match the current plan",
            hint="Run --publish without --confirm-publish to review the plan.",
        )
    elif not yes:
        raise PipelineError(ExitCode.CONFIRMATION_REQUIRED, "E_CONFIRMATION_REQUIRED", "Uploading needs --yes in addition to the confirmation token")
    if not hf_token:
        raise PipelineError(
            ExitCode.CONFIG,
            "E_HF_TOKEN_MISSING",
            f"No Hugging Face login (${token_env} is not set and there is no stored login)",
            hint="Log in with .venv/bin/hf auth login, or export a write token; it is never written by this project.",
        )
    return upload(workspace, run_id, plan, generated, hf_token, services, stream)


RE_UPLOAD_REPO = re.compile(r"^Uploading (\S+)")
# huggingface_hub 1.33.0 upload_large_folder prints this status every 60 s (LargeUploadStatus.current_report).
RE_UPLOAD_STATUS = re.compile(r"pre-uploaded: (\d+)/(\d+) \(([^/]+)/([^)]+)\).*committed: (\d+)/(\d+)")


class UploadWatch:
    """The console's live line during the upload, from the uploader's output."""

    def __init__(self, console) -> None:
        self.console, self.repo, self.status = console, None, "preparing"

    def on_line(self, _label: str, line: str) -> None:
        if m := RE_UPLOAD_REPO.search(line.strip()):
            self.repo, self.status = m.group(1), "hashing and checking files"
        elif m := RE_UPLOAD_STATUS.search(line):
            self.status = f"{m.group(3)} of {m.group(4)} sent · {m.group(5)}/{m.group(6)} files committed"
        else:
            return
        self.console.activity(f"{self.repo}: {self.status}" if self.repo else self.status)


def upload(workspace: Path, run_id: str, plan: dict[str, Any], generated: dict[str, dict[str, bytes]], hf_token: str, services, stream, console=None) -> int:
    """Stage, upload, check every file remotely and record a ``publication`` assessment per artifact."""
    console = console or Console(sys.stderr)
    console.stage_start("[publish]")
    try:
        return _upload(workspace, run_id, plan, generated, hf_token, services, stream, console)
    finally:
        console.close()


def _upload(workspace: Path, run_id: str, plan, generated, hf_token: str, services, stream, console) -> int:
    console.activity("re-hashing the files")
    folders = stage(workspace, plan, generated)
    log_dir = workspace / "runs" / run_id / "logs" / "publish"
    runner, record = heretic_tools(workspace, services.scripts_dir, log_dir, services.probe_factory(True))
    job = {
        "private": plan["private"],
        "repos": [
            {
                **{k: r[k] for k in ("repo_id", "kind")},
                "folder": str(folders[r["repo_id"]]),
                "files": [{k: f[k] for k in ("path", "size_bytes", "sha256", "git_sha1")} for f in r["files"]],
            }
            for r in plan["repos"]
        ],
    }
    job_path, out_path = log_dir / "publish-job.json", log_dir / "publish-result.json"
    atomic_write_json(job_path, job)
    out_path.unlink(missing_ok=True)
    result = runner.run(
        [*record.script("hf_publish.py"), "--job", str(job_path), "--out", str(out_path)],
        cwd=None,
        env=record.activate(child_env(HF_TOKEN=hf_token, HF_HOME=str(workspace / "cache" / "hf"), PYTHONUNBUFFERED="1")),
        log_dir=log_dir,
        name="hf-publish",
        timeout_seconds=24 * 3600,
        on_line=UploadWatch(console).on_line,
    )
    console.close()
    data = read_json(out_path) if out_path.exists() else {"repos": [], "error": f"no result (exit {result.returncode})"}
    registry = ArtifactRegistry(workspace / "runs" / run_id, workspace)
    report = {"run_id": run_id, "published_utc": utc_now(), "repos": data.get("repos", []), "error": data.get("error")}
    by_repo = {r["repo_id"]: r for r in data.get("repos", [])}
    failures = []
    for repo in plan["repos"]:
        res = by_repo.get(repo["repo_id"]) or {"verified": False, "error": data.get("error") or "not uploaded"}
        ok = bool(res.get("verified"))
        if not ok:
            failures.append(repo["repo_id"])
        for role in repo["roles"]:
            entry = registry.get(role)
            rec = A.new_record(
                artifact_id=entry["artifact_id"],
                role=role,
                run_id=run_id,
                stage_id="publish",
                kind="publication",
                results={
                    "repo_id": repo["repo_id"],
                    "url": f"https://huggingface.co/{repo['repo_id']}",
                    "revision": res.get("revision"),
                    "private": res.get("private"),
                    "files": len(repo["files"]),
                    "mismatches": res.get("mismatches"),
                    "error": res.get("error"),
                },
                checks=[A.check("remote_files_match", "passed" if ok else "failed", res.get("mismatches") or None, "every file's hash matches the manifest")],
            )
            registry.attach(role, rec)
        stream.write(
            f"{'published' if ok else 'FAILED'}: https://huggingface.co/{repo['repo_id']} ({'private' if res.get('private', plan['private']) else 'PUBLIC'}) {res.get('revision') or res.get('error')}\n"
        )
    atomic_write_json(workspace / "runs" / run_id / "reports" / "publication.json", report)
    if failures:
        raise PipelineError(
            ExitCode.VERIFY,
            "E_PUBLISH_FAILED",
            f"Upload or remote verification failed for {failures}",
            hint=f"See {log_dir}; re-running the same command resumes the upload.",
        )
    remove_tree(workspace / "publish" / run_id)
    return int(ExitCode.OK)
