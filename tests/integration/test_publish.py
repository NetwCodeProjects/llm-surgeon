"""--publish with the fake upload runtime (docs/release.md): release gate, plan and token, upload, remote check."""

from __future__ import annotations

import io
import json
import re

import pytest
import yaml

from llm_surgeon.errors import ExitCode, PipelineError
from llm_surgeon.publish import publish


def token_from(out: str) -> str:
    return re.search(r"--confirm-publish ([0-9a-f]{16})", out).group(1)


def ready_run(env) -> str:
    env.provision()
    code, out = env.run("--quantizations", "fp8,q4_k_m")
    assert code == 0, out
    return env.latest_run().name


def test_plan_lists_three_private_repositories_and_uploads_nothing(env, monkeypatch):
    run = ready_run(env)
    code, out = env.run("--publish", run, "--hf-namespace", "tester")
    assert code == 0, out
    for repo in ("tester/Qwen3.8-27B-abliterated ", "tester/Qwen3.8-27B-abliterated-FP8 ", "tester/Qwen3.8-27B-abliterated-GGUF "):
        assert f"https://huggingface.co/{repo}" in out
    assert "PRIVATE" in out and "README.md" in out and "mmproj-" in out
    assert not (env.counters / "hub").exists()
    assert (env.workspace / "runs" / run / "reports" / "release-record.md").exists()


def test_publish_uploads_checks_and_records(env, monkeypatch):
    run = ready_run(env)
    code, out = env.run("--publish", run, "--hf-namespace", "tester")
    token = token_from(out)
    with pytest.raises(PipelineError) as no_yes:  # env.run always passes --yes, so call the function directly
        publish(env.workspace, run, "tester", None, token, False, env.services, io.StringIO())
    assert no_yes.value.code == ExitCode.CONFIRMATION_REQUIRED
    monkeypatch.delenv("HF_TOKEN", raising=False)
    code, out = env.run("--publish", run, "--hf-namespace", "tester", "--confirm-publish", token, "--yes")
    assert code == ExitCode.CONFIG, out
    monkeypatch.setenv("HF_TOKEN", "hf_fake_write_token")
    code, out = env.run("--publish", run, "--hf-namespace", "tester", "--confirm-publish", "0" * 16, "--yes")
    assert code == ExitCode.USAGE, out
    code, out = env.run("--publish", run, "--hf-namespace", "tester", "--confirm-publish", token, "--yes")
    assert code == 0, out
    hub = env.counters / "hub" / "tester"
    master, fp8, gguf = (hub / n for n in ("Qwen3.8-27B-abliterated", "Qwen3.8-27B-abliterated-FP8", "Qwen3.8-27B-abliterated-GGUF"))
    assert (master / "config.json").exists() and (fp8 / "config.json").exists()
    assert sorted(p.name for p in gguf.glob("*.gguf")) == ["mmproj-qwen3.8-27b-f16.gguf", next(p.name for p in gguf.glob("*Q4_K_M.gguf"))]
    card = (master / "README.md").read_text(encoding="utf-8")
    meta = yaml.safe_load(card.split("---")[1])
    assert meta["license"] == "apache-2.0" and meta["base_model"] == ["Qwen/Qwen3.8-27B"] and "abliterated" in meta["tags"]
    assert meta["pipeline_tag"] == "image-text-to-text"
    gmeta = yaml.safe_load((gguf / "README.md").read_text(encoding="utf-8").split("---")[1])
    assert gmeta["base_model"] == ["tester/Qwen3.8-27B-abliterated"] and gmeta["base_model_relation"] == "quantized"
    assert "Refusals on Heretic's evaluation prompts" in card and "deliberately reduced" in card and "c0dec0dec0de" in card
    assert len(list((master / "llm-surgeon").glob("manifest-art-*.json"))) == 1 and (master / "llm-surgeon" / "release-record.json").exists()
    manifest = json.loads(
        next(
            (env.workspace / "manifests").glob(
                f"{json.loads((master / 'llm-surgeon' / next(p.name for p in (master / 'llm-surgeon').glob('manifest-*'))).read_text())['artifact_id']}.json"
            )
        ).read_text()
    )
    kinds = [a["kind"] for a in manifest["assessments"]]
    assert kinds[-1] == "publication"
    report = json.loads((env.workspace / "runs" / run / "reports" / "publication.json").read_text())
    assert [r["verified"] for r in report["repos"]] == [True, True, True]
    assert not (env.workspace / "publish" / run).exists(), "staging removed after a verified upload"


def test_a_corrupted_upload_is_caught(env, monkeypatch):
    run = ready_run(env)
    monkeypatch.setenv("HF_TOKEN", "hf_fake_write_token")
    monkeypatch.setenv("FAKE_HF_CORRUPT", "config.json")
    code, out = env.run("--publish", run, "--hf-namespace", "tester")
    code, out = env.run("--publish", run, "--hf-namespace", "tester", "--confirm-publish", token_from(out), "--yes")
    assert code == ExitCode.VERIFY, out
    report = json.loads((env.workspace / "runs" / run / "reports" / "publication.json").read_text())
    assert {r["repo_id"]: r["mismatches"] for r in report["repos"]}["tester/Qwen3.8-27B-abliterated"] == ["config.json"]


def test_a_run_that_is_not_ready_is_refused(env, monkeypatch):
    monkeypatch.setenv("FAKE_DEGRADE", "q4_k_m")
    run = ready_run(env)
    code, out = env.run("--publish", run, "--hf-namespace", "tester")
    assert code == ExitCode.GATE, out
    record = json.loads((env.workspace / "runs" / run / "reports" / "release-record.json").read_text())
    assert record["verdict"] == "not ready" and record["failed"] == ["comparison"]


def test_publish_flags_need_each_other(env):
    for args in (("--hf-namespace", "tester"), ("--publish", "r1")):
        with pytest.raises(SystemExit) as usage:  # argparse usage error
            env.run(*args)
        assert usage.value.code == ExitCode.USAGE
