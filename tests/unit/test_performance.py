"""Performance changes from the B300 run: llama-server, thread counts, parallel hashing, batch-size keys."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest
from conftest import FAKE_BIN

from llm_surgeon.adapters import heretic as H
from llm_surgeon.adapters import llama_cpp as LC
from llm_surgeon.adapters import llama_server as LS
from llm_surgeon.artifacts.manifest import ArtifactRegistry
from llm_surgeon.errors import Interrupted
from llm_surgeon.evaluation.comparison import prompt_token_check
from llm_surgeon.fsutil import file_inventory
from llm_surgeon.subprocesses import CommandRunner, child_env
from llm_surgeon.telemetry import Redactor


def test_quantize_threads_come_from_the_caller():
    argv = LC.quantize_argv(["llama-quantize"], Path("in.gguf"), Path("out.gguf"), "q8_0", None, 18)
    assert argv[-2:] == ["Q8_0", "18"]
    assert LC.quantize_argv(["q"], Path("a"), Path("b"), "iq4_xs", Path("m.gguf"), 0)[-1] == "1"


def test_server_argv_and_request_match_the_verified_interface():
    argv = LS.server_argv(["llama-server"], Path("m.gguf"), 4096, "all", 8123, 36)
    assert argv == [
        "llama-server", "-m", "m.gguf", "-c", "4096", "-ngl", "all", "-t", "36", "-np", "1",
        "--host", "127.0.0.1", "--port", "8123", "--no-webui",
    ]  # fmt: skip
    assert LS.completion_request("p", 64) == {"prompt": "p", "n_predict": 64, "temperature": 0, "seed": 0, "cache_prompt": False}
    # Real response fields at d854d98 (docs/implementation-log.md, Qwen3.5-0.8B GGUF in WSL)
    body = {"content": " Paris", "stop_type": "eos", "tokens_evaluated": 24, "tokens_predicted": 2, "truncated": False}
    assert LS.completion_result(body) == {"response": " Paris", "finish_reason": "stop", "prompt_tokens": 24, "new_tokens": 2, "truncated": False}
    assert LS.completion_result({**body, "stop_type": "limit"})["finish_reason"] == "length"


def server(tmp_path, event=None, **env):
    runner = CommandRunner(Redactor(), event or threading.Event())
    return LS.LlamaServer(
        runner,
        [sys.executable, str(FAKE_BIN / "fake_llama.py"), "server"],
        tmp_path / "model.gguf",
        ctx=2048,
        gpu_layers="0",
        threads=4,
        env=child_env(**env),
        log_dir=tmp_path / "logs",
        name="llama-server-test",
        timeout_seconds=60,
        interrupt_event=event or threading.Event(),
    )


def test_one_server_serves_every_prompt_and_stops(tmp_path):
    srv = server(tmp_path)
    with srv:
        first = srv.complete("<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n", 16)
        second = srv.complete("raw smoke prompt", 16)
        assert srv.process.alive()
    assert first["finish_reason"] == "stop" and first["prompt_tokens"] == 8  # the fake counts whitespace-separated words
    assert "Paris" in second["response"]
    assert srv.load_seconds is not None and not srv.process.alive()
    assert (tmp_path / "logs" / "llama-server-test.stderr.log").read_text(encoding="utf-8").count("listening") == 1


def test_a_server_that_exits_while_loading_is_a_failure(tmp_path):
    srv = server(tmp_path, FAKE_SERVER_FAIL="1")
    with pytest.raises(LS.LlamaServerError, match="exited with 1 before the model loaded"):
        with srv:
            pass
    assert "failed to load model" in str(srv.result.tail)


def test_a_lost_port_race_is_retried_on_a_new_port(tmp_path):
    srv = server(tmp_path, FAKE_SERVER_BIND_FAIL_ONCE=str(tmp_path / "bind-failed"))
    with srv:
        assert "Paris" in srv.complete("raw smoke prompt", 8)["response"]
    stderr = (tmp_path / "logs" / "llama-server-test.stderr.log").read_text(encoding="utf-8")
    assert stderr.count("couldn't bind HTTP server socket") == 1 and stderr.count("listening") == 1


def test_an_interrupt_stops_the_server(tmp_path):
    event = threading.Event()
    srv = server(tmp_path, event)
    with pytest.raises(Interrupted):
        with srv:
            event.set()
            srv.complete("x", 4)
    assert not srv.process.alive()


def test_parallel_hashing_matches_sequential(tmp_path):
    for i in range(6):
        (tmp_path / f"shard-{i}.bin").write_bytes(bytes([i]) * (100_000 + i))
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "x.json").write_text("{}")
    assert file_inventory(tmp_path, workers=8) == file_inventory(tmp_path)


def test_concurrent_registrations_keep_every_role(tmp_path, monkeypatch):
    registry = ArtifactRegistry(tmp_path / "run", tmp_path / "ws")
    monkeypatch.setattr(registry, "_write", lambda manifest, role, path: manifest)
    monkeypatch.setattr("llm_surgeon.artifacts.manifest.check_manifest", lambda manifest: None)
    roles = [f"export:gguf:q{i}" for i in range(12)]
    threads = [
        threading.Thread(target=registry.register, args=({"artifact_id": f"art-{i:024x}"}, role, tmp_path / role.replace(":", "-"), "08"))
        for i, role in enumerate(roles)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(registry.roles()) == sorted(roles)


def test_prompt_token_check_reports_mismatches():
    suite = [{"id": "a", "prompt_tokens": 10}, {"id": "b", "prompt_tokens": 12}, {"id": "c", "prompt_tokens": None}]
    rows = [{"id": "a", "prompt_tokens": 10}, {"id": "b", "prompt_tokens": 13}, {"id": "c", "prompt_tokens": 5}]
    assert prompt_token_check(suite, rows) == {"compared": 2, "mismatches": [{"id": "b", "baseline": 12, "export": 13}]}
    assert prompt_token_check(suite, None) is None and prompt_token_check(suite, [{"id": "a"}]) is None


def test_max_batch_size_overrides_the_template_only_when_set():
    template = {"max_batch_size": 128, "n_startup_trials": 30}
    kwargs = dict(model_path="m", n_trials=12, n_startup_trials=None, seed=1, trial_index=0, save_directory="s", checkpoint_action="restart")
    assert H.build_optimize_config(template, **kwargs)["max_batch_size"] == 128
    native = H.build_optimize_config(template, max_batch_size=512, **kwargs)
    assert native["max_batch_size"] == 512 and template["max_batch_size"] == 128
    assert H.study_hash(native, "c") != H.study_hash(H.build_optimize_config(template, **kwargs), "c"), "a new batch cap is a new study"
