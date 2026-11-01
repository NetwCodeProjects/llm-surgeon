"""Performance paths with the fake runtimes: llama-server in stage 09, parallel stage 08, caches, batch keys."""

from __future__ import annotations

import json
from pathlib import Path

from llm_surgeon.errors import ExitCode


def comparison(run: Path) -> dict:
    return json.loads((run / "metrics" / "comparison.json").read_text())


def test_each_gguf_export_loads_once_in_llama_server(env):
    env.provision()
    code, out = env.run()
    assert code == 0, out
    run = env.latest_run()
    smoke_prompts = len(env.config().get("quantization.smoke_prompts"))
    assert env.count("llama-server") == 2, "one server per GGUF export (iq4_xs, iq3_s)"
    assert env.count("llama-completion") == 0
    assert env.count("llama-server-request") == 2 * (smoke_prompts + 12), "loader prompts + 8 benign + 4 held-out per export"
    assert all("threads=32" in line for line in (env.counters / "llama-server.log").read_text().splitlines()), "-t from available_cpus"
    doc = comparison(run)
    for role in ("export:gguf:iq4_xs", "export:gguf:iq3_s", "export:fp8"):
        record = json.loads((env.workspace / "assessments" / doc["exports"][role]["artifact_id"] / f"{doc['exports'][role]['assessment_id']}.json").read_text())
        assert record["results"]["prompt_tokens"] == {"compared": 12, "mismatches": []}, role
    verification = json.loads((run / "metrics" / "verification.json").read_text())
    gguf = verification["export:gguf:iq4_xs"]["smoke"]
    assert gguf["runtime"] == "llama-server" and gguf["passed"] and gguf["load_seconds"] is not None
    assert gguf["modalities"] == ["text", "image"], "the image test still runs with llama-mtmd-cli"


def test_a_gguf_server_that_cannot_load_fails_only_that_export(env, monkeypatch):
    monkeypatch.setenv("FAKE_SERVER_FAIL", "1")
    env.provision()
    code, out = env.run()
    assert code == ExitCode.PARTIAL, out
    receipt = env.receipt(env.latest_run(), "09")
    assert receipt["diagnostics"]["inference_failures"] == ["export:gguf:iq3_s", "export:gguf:iq4_xs"]
    doc = comparison(env.latest_run())
    assert doc["exports"]["export:gguf:iq4_xs"]["outcome"] == "failed" and doc["exports"]["export:fp8"]["outcome"] == "passed"
    loaded = doc["exports"]["export:gguf:iq4_xs"]["checks"][0]
    assert (loaded["check"], loaded["result"], loaded["reason"]) == ("export_loaded", "failed", "did not load")


def test_formats_quantize_concurrently_with_split_threads(env):
    env.provision()
    code, out = env.run()
    assert code == 0, out
    receipt = env.receipt(env.latest_run(), "08")
    assert receipt["diagnostics"]["parallel"] == {"enabled": True, "jobs": 3, "llama_quantize_threads": 16}
    argv_lines = [p.read_text().splitlines()[0] for p in (env.latest_run() / "logs" / "08-quantize").glob("gguf-quantize-*.stdout.log")]
    assert len(argv_lines) == 2 and all(line.endswith(" 16") for line in argv_lines), argv_lines


def test_sequential_quantization_uses_every_cpu(env):
    env.provision()
    code, out = env.run("--set", "quantization.parallel=false")
    assert code == 0, out
    receipt = env.receipt(env.latest_run(), "08")
    assert receipt["diagnostics"]["parallel"] == {"enabled": False, "jobs": 3, "llama_quantize_threads": 32}
    assert sorted(receipt["diagnostics"]["results"]) == ["fp8", "iq3_s", "iq4_xs"]


def test_qwen3_5_profile_restores_mtp_tensors_by_default(env):
    assert env.config().get("selection.graft_missing_source_tensors") == ["mtp."]
    env.provision()
    code, out = env.run("--quantizations", "fp8")
    assert code == 0, out
    master = json.loads(Path(env.artifacts(env.latest_run())["master"]["manifest"]).read_text())
    assert master["extensions"]["tensor_diff_vs_source"]["missing_count"] == 0
    code, out = env.run("--quantizations", "fp8", "--new-run", "--set", "selection.graft_missing_source_tensors=[]")
    assert code == 0, out
    master = json.loads(Path(env.artifacts(env.latest_run())["master"]["manifest"]).read_text())
    assert master["extensions"]["tensor_diff_vs_source"]["missing_prefixes"] == {"mtp": 2}, "opting out records the gap"


def test_vllm_caches_and_batch_sizes_reach_the_tools(env):
    env.provision()
    code, out = env.run("--set", "abliteration.max_batch_size=512", "--set", "evaluation.batch_size=32")
    assert code == 0, out
    run = env.latest_run()
    vllm = json.loads((run / "logs" / "09-verify_exports" / "vllm-smoke.json").read_text())
    assert vllm["enforce_eager"] is True
    assert Path(vllm["vllm_cache_root"]).resolve() == (env.workspace / "cache" / "vllm").resolve()
    assert "max_batch_size = 512" in (run / "checkpoints" / "heretic" / "prepared-config.toml").read_text()
    spec = json.loads((run / "logs" / "06-qualify" / "qualify-spec.json").read_text())
    assert spec["batch_size"] == 32
    baseline = json.loads((run / "logs" / "09-verify_exports" / "comparison-baseline-spec.json").read_text())
    assert baseline["batch_size"] == 32


def test_an_interrupt_during_parallel_quantization_stops_the_stage_and_resumes(env, monkeypatch):
    from llm_surgeon.errors import Interrupted
    from llm_surgeon.stages.stage08_quantize import QuantizeStage

    def interrupted(self, ctx, inp):
        ctx.run.interrupt_event.set()  # what the signal handler does on Ctrl-C
        raise Interrupted()

    env.provision()
    monkeypatch.setattr(QuantizeStage, "_fp8", interrupted)
    code, out = env.run()
    assert code == ExitCode.INTERRUPTED, out
    run = env.latest_run()
    assert env.receipt(run, "08")["outcome"] == "interrupted"
    sub = run / "receipts" / "08-quantize" / "fp8.json"
    assert not sub.exists(), "an interrupt is not a failed format"
    monkeypatch.undo()
    code, out = env.run("--resume", run.name)
    assert code == 0, out
    assert json.loads(sub.read_text())["outcome"] == "succeeded"
