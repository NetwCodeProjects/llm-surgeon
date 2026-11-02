"""Publishing chosen at the start of a run (docs/release.md): the questions, the picker by bit level, the login
and name checks before the first stage, and the upload at the end. Fake hub, fake upload runtime."""

from __future__ import annotations

import io
import json
import re
import subprocess
from datetime import datetime, timezone

import pytest
import yaml

from llm_surgeon import configspec
from llm_surgeon import formats as F
from llm_surgeon.environment.detect import HostProbe
from llm_surgeon.errors import ExitCode, PipelineError
from llm_surgeon.publish_setup import Prompter, access, expiry_note, pick_formats


class Answers:
    """Scripted answers; fails on a question the test did not expect."""

    def __init__(self, *answers: str) -> None:
        self.answers, self.prompts = list(answers), []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        assert self.answers, f"unexpected question {prompt!r}"
        return self.answers.pop(0)


def prompter(*answers: str):
    stream, answer = io.StringIO(), Answers(*answers)
    return Prompter(type("S", (), {"ask": staticmethod(answer)})(), stream), stream, answer


def interactive(env, *answers: str) -> Answers:
    answer = Answers(*answers)
    env.services.interactive = lambda: True
    env.services.ask = answer
    return answer


def stored_config(env) -> dict:
    return json.loads((env.latest_run() / "configs" / "resolved-config.json").read_text())["config"]


# ------------------------------------------------------------------------------------------ the picker
def test_the_catalogue_covers_every_buildable_format_except_the_ternary_types():
    gguf = {f.name for f in F.CATALOGUE if f.container == "GGUF"}
    assert gguf == set(configspec.GGUF_TYPES) - {"tq1_0", "tq2_0"}
    assert [len(F.level(b)) for b in F.LEVELS] == [1, 2, 1, 4, 6, 7, 6, 2]  # 16, 8, 6, 5, 4, 3, 2, 1 bit


def test_pick_by_bit_level_then_by_format():
    p, stream, answer = prompter("1,2,5", "1", "1,3")
    chosen = pick_formats(p, ["master", "fp8", "q4_k_m"], 27_000_000_000, {})
    assert chosen == ["master", "fp8", "q4_k_m", "iq4_xs"]
    assert answer.prompts == ["Choice [1,2,5]: ", "Choice [1]: ", "Choice [1]: "]
    out = stream.getvalue()
    assert "16-bit   BF16 master" in out and "~54.0 GB" in out and "4-bit    6 formats" in out
    assert "Q4_K_M   GGUF" in out and "~16.5 GB" in out and "the common default" in out


def test_enter_accepts_the_defaults_and_a_new_level_defaults_to_its_usual_format():
    p, _stream, _ = prompter("", "", "")
    assert pick_formats(p, ["master", "fp8", "q4_k_m"], None, {}) == ["master", "fp8", "q4_k_m"]
    p, _stream, _ = prompter("4", "")
    assert pick_formats(p, ["master"], None, {}) == ["q5_k_m"]


def test_a_format_that_cannot_be_published_here_is_refused_and_asked_again():
    p, stream, answer = prompter("2,7", "1", "2", "3", "2")
    blocked = {"fp8": "its loader test is off here", "iq2_s": "needs an importance matrix: set quantization.gguf.imatrix_dataset"}
    assert pick_formats(p, ["fp8"], None, blocked) == ["q8_0", "iq2_m"]
    assert answer.prompts[1] == "Choice [2]: ", "the blocked default moves to the next format"
    out = stream.getvalue()
    assert "1 cannot be published here: its loader test is off here" in out
    assert "3 cannot be published here: needs an importance matrix" in out
    assert "IQ2_S    GGUF         needs an importance matrix: set quantization.gguf.imatrix_dataset" in out


def test_invalid_numbers_are_asked_again():
    p, stream, _ = prompter("9", "x", "1")
    assert pick_formats(p, ["master"], None, {}) == ["master"]
    assert stream.getvalue().count("Enter numbers from 1 to 8") == 2


# ------------------------------------------------------------------------------------------ the login
def who(role=None, scoped=None, orgs=(), expires=None):
    token = {"role": role} if role else {}
    if scoped is not None:
        token["fineGrained"] = {"scoped": scoped}
    auth = {"type": "access_token" if role else "oauth", "accessToken": token}
    if expires:
        auth["expiresAt"] = expires
    return {"name": "tester", "orgs": [{"name": o} for o in orgs], "auth": auth}


def test_write_access_by_token_kind():
    assert access(who("write"), "tester") == "write access ok"
    assert access(who(), "tester") == "write access is confirmed when the repositories are created"  # OAuth
    assert access(who("fineGrained", [{"entity": {"name": "team"}, "permissions": ["repo.write"]}], orgs=("team",)), "team") == "write access ok"
    for bad, namespace, error in (
        (who("read"), "tester", "E_HF_TOKEN_READ_ONLY"),
        (who("fineGrained", [{"entity": {"name": "tester"}, "permissions": ["repo.content.read"]}]), "tester", "E_HF_TOKEN_READ_ONLY"),
        (who("write"), "someone-else", "E_PUBLISH_NAMESPACE"),
    ):
        with pytest.raises(PipelineError) as exc:
            access(bad, namespace)
        assert exc.value.error_id == error


def test_a_login_that_expires_soon_is_flagged():
    now = datetime(2026, 11, 1, tzinfo=timezone.utc)
    assert expiry_note(who(expires="2026-12-01T04:52:19.000Z"), now) == "login valid until 2026-12-01"
    assert expiry_note(who(expires="2026-11-02T00:00:00.000Z"), now).startswith("WARNING: this login expires 2026-11-02")


# ------------------------------------------------------------------------------------------ whole runs
def test_asked_at_the_start_logged_in_picked_and_uploaded_at_the_end(env, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    env.provision(["--quantizations", "fp8,q4_k_m"])  # the components the picked formats need
    answer = interactive(env, "y", "", "", "1,2,5", "1", "1", "")  # yes, name, visibility (private), the picker, continue
    code, out = env.run()
    assert code == 0, out
    assert answer.prompts[:2] == ["Publish to Hugging Face when the run is ready? [y/N] ", "Repository name [Qwen3.8-27B-abliterated]: "]
    assert env.account.logins == 1, "no token: `hf auth login` ran once"
    assert "Logged in as tester (write access ok)" in out
    assert "Publish      : BF16 master, FP8, GGUF Q4_K_M" in out
    assert "Repositories : tester/Qwen3.8-27B-abliterated, -FP8, -GGUF   (private, names free)" in out
    assert "These formats differ from the config file" in out
    assert out.index("Summary") < out.index("Run ")  # the questions come before the first stage
    config = stored_config(env)
    assert config["publish"] == {
        "enabled": True,
        "namespace": "tester",
        "repo_name": "Qwen3.8-27B-abliterated",
        "formats": ["master", "fp8", "q4_k_m"],
        "visibility": "private",
    }
    assert "Private      : counts toward your Hugging Face private storage" in out
    assert config["quantization"]["formats"] == ["fp8", "q4_k_m"]
    hub = env.counters / "hub" / "tester"
    assert sorted(p.name for p in hub.iterdir()) == ["Qwen3.8-27B-abliterated", "Qwen3.8-27B-abliterated-FP8", "Qwen3.8-27B-abliterated-GGUF"]
    report = json.loads((env.latest_run() / "reports" / "publication.json").read_text())
    assert [r["verified"] for r in report["repos"]] == [True, True, True]
    assert "published: https://huggingface.co/tester/Qwen3.8-27B-abliterated-GGUF (private)" in out


def test_answering_no_runs_without_publishing(env, monkeypatch):
    env.provision()
    interactive(env, "n")
    code, out = env.run()
    assert code == 0, out
    assert stored_config(env)["publish"]["enabled"] is False
    assert not (env.counters / "hub").exists() and "Not published" not in out


def test_without_a_terminal_nothing_is_asked_and_the_command_is_printed(env):
    env.provision()
    code, out = env.run()
    assert code == 0, out
    assert "Not published (not asked: no interactive terminal). To publish this run: ./main.sh --output-dir " in out
    assert f"--publish {env.latest_run().name}" in out


def test_problems_that_would_show_after_hours_stop_the_run_before_the_first_stage(env, monkeypatch):
    env.provision()
    monkeypatch.delenv("HF_TOKEN", raising=False)
    code, out = env.run("--set", "publish.enabled=true")
    assert code == ExitCode.CONFIG and not (env.workspace / "runs").exists(), out
    monkeypatch.setenv("HF_TOKEN", "hf_fake_write_token")
    taken = env.counters / "hub" / "tester" / "Qwen3.8-27B-abliterated-FP8"
    taken.mkdir(parents=True)
    (taken / "README.md").write_text("someone else's model")
    code, out = env.run("--set", "publish.enabled=true")
    assert code == ExitCode.CONFIG and not (env.workspace / "runs").exists(), out
    code, out = env.run("--set", "publish.enabled=true", "--set", "publish.repo_name=other", "--set", "quantization.fp8.verify_backend=none")
    assert code == ExitCode.CONFIG and not (env.workspace / "runs").exists(), out


def test_a_taken_name_is_asked_again_on_a_terminal(env, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_fake_write_token")
    taken = env.counters / "hub" / "tester" / "Qwen3.8-27B-abliterated"
    taken.mkdir(parents=True)
    (taken / "config.json").write_text("{}")
    env.provision()
    # yes, the default name, private, the picker's defaults (levels, 8-, 4- and 3-bit), a free name for the taken one, then no
    answer = interactive(env, "y", "", "", "", "", "", "", "mine", "n")
    code, out = env.run()
    assert code == ExitCode.CONFIRMATION_REQUIRED, out
    assert "tester/Qwen3.8-27B-abliterated already exists with files. Choose another name." in out
    assert answer.prompts[1] == answer.prompts[-2] == "Repository name [Qwen3.8-27B-abliterated]: "
    assert answer.prompts[-1] == "Continue? [Y/n] "
    assert "tester/mine, -FP8, -GGUF" in out


def test_a_resumed_run_publishes_with_the_login_as_namespace(env, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_fake_write_token")
    env.provision()
    code, out = env.run("--set", "publish.enabled=true", "--through", "4")
    assert code == 0 and "Publishing: skipped (this invocation stops before the last stage)" in out, out
    run = env.latest_run().name
    code, out = env.run("--resume", run)  # the stored choice, not the config file's 'ask'
    assert code == 0, out
    assert "Publishing: on, as chosen when the run started" in out
    assert len(list((env.counters / "hub" / "tester").iterdir())) == 3


def visibility(env) -> dict:
    return json.loads((env.counters / "hub" / "visibility.json").read_text())


def test_public_is_chosen_at_the_start_and_uploaded_directly_as_public(env, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_fake_write_token")
    env.provision()
    answer = interactive(env, "y", "", "2", "", "", "", "", "")  # yes, name, public, picker defaults, continue
    code, out = env.run()
    assert code == 0, out
    assert "Visibility once uploaded:" in out and answer.prompts[2] == "Choice [1]: "
    assert "(public, names free)" in out and "PUBLIC       : anyone can see and download" in out
    assert stored_config(env)["publish"]["visibility"] == "public"
    assert set(visibility(env).values()) == {"public"} and len(visibility(env)) == 3
    assert "published: https://huggingface.co/tester/Qwen3.8-27B-abliterated (PUBLIC)" in out


def test_without_a_terminal_ask_means_private(env, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_fake_write_token")
    env.provision()
    code, out = env.run("--set", "publish.enabled=true")
    assert code == 0, out
    assert set(visibility(env).values()) == {"private"}


def test_the_plan_token_covers_the_visibility_and_a_failed_public_upload_is_made_private(env, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_fake_write_token")
    env.provision(["--quantizations", "fp8,q4_k_m"])
    code, out = env.run("--quantizations", "fp8,q4_k_m")
    run = env.latest_run().name
    code, private_plan = env.run("--publish", run)
    code, public_plan = env.run("--publish", run, "--visibility", "public")
    assert "Every repository: PRIVATE (only you)" in private_plan and "Every repository: PUBLIC" in public_plan
    private_token, public_token = (re.search(r"--confirm-publish ([0-9a-f]{16})", o).group(1) for o in (private_plan, public_plan))
    assert private_token != public_token and "--visibility public --confirm-publish" in public_plan
    code, out = env.run("--publish", run, "--visibility", "public", "--confirm-publish", private_token, "--yes")
    assert code == ExitCode.USAGE, "a token reviewed for a private upload does not upload publicly"
    monkeypatch.setenv("FAKE_HF_CORRUPT", "config.json")
    code, out = env.run("--publish", run, "--visibility", "public", "--confirm-publish", public_token, "--yes")
    assert code == ExitCode.VERIFY, out
    assert visibility(env)["tester/Qwen3.8-27B-abliterated"] == "private" and visibility(env)["tester/Qwen3.8-27B-abliterated-GGUF"] == "public"


def test_publishing_only_a_gguf_type(env, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_fake_write_token")
    env.provision()
    code, out = env.run("--set", "publish.enabled=true", "--set", "publish.formats=[iq4_xs]")
    assert code == 0, out
    assert "Also built   : BF16 master (abliteration produces it; not published)" in out
    hub = env.counters / "hub" / "tester"
    assert [p.name for p in hub.iterdir()] == ["Qwen3.8-27B-abliterated-GGUF"]
    gguf = hub / "Qwen3.8-27B-abliterated-GGUF"
    assert sorted(p.name for p in gguf.glob("*.gguf") if not p.name.startswith("mmproj")) == [next(p.name for p in gguf.glob("*IQ4_XS.gguf"))]
    meta = yaml.safe_load((gguf / "README.md").read_text(encoding="utf-8").split("---")[1])
    assert meta["base_model"] == ["Qwen/Qwen3.8-27B"] and "base_model_relation" not in meta
    assert "BF16 master not published" in (gguf / "README.md").read_text(encoding="utf-8")


def test_the_release_check_judges_only_the_published_exports(env, monkeypatch):
    monkeypatch.setenv("FAKE_DEGRADE", "iq3_s")
    env.provision()
    code, out = env.run("--set", "publish.enabled=false", "--set", "publish.formats=[master, fp8]")
    assert code == 0, out
    run = env.latest_run().name
    code, out = env.run("--publish", run, "--hf-namespace", "tester")
    assert code == 0, out
    assert "Qwen3.8-27B-abliterated-FP8 " in out and "GGUF" not in out
    record = json.loads((env.workspace / "runs" / run / "reports" / "release-record.json").read_text())
    assert record["verdict"] == "ready"
    assert [r["outcome"] for r in json.loads((env.workspace / "runs" / run / "metrics" / "comparison.json").read_text())["exports"].values()].count(
        "failed"
    ) == 1, "the unpublished IQ3_S export did fail its comparison"


def test_publish_later_asks_before_uploading_on_a_terminal(env, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_fake_write_token")
    env.provision(["--quantizations", "fp8,q4_k_m"])
    code, out = env.run("--quantizations", "fp8,q4_k_m")
    assert code == 0, out
    run = env.latest_run().name
    interactive(env, "n")
    code, out = env.run("--publish", run)  # the namespace comes from the login
    assert code == 0 and "tester/Qwen3.8-27B-abliterated" in out and "--confirm-publish" not in out
    assert not (env.counters / "hub").exists()
    interactive(env, "y")
    code, out = env.run("--publish", run)
    assert code == 0, out
    assert len(list((env.counters / "hub" / "tester").iterdir())) == 3


def test_publish_formats_must_be_built(env):
    code, out = env.run("--set", "publish.formats=[master, q8_0]", "--dry-run")
    assert code == ExitCode.CONFIG


# ------------------------------------------------------------------------------------------ the code revision
def test_config_edits_do_not_make_the_checkout_dirty(tmp_path):
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "pipeline.yaml").write_text("a: 1\n")
    (tmp_path / "main.py").write_text("print(1)\n")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "init")
    probe = HostProbe(allow_exec=True)
    (tmp_path / "configs" / "pipeline.yaml").write_text("a: 2\n")
    assert probe.code_revision(tmp_path)["git_dirty"] is False
    (tmp_path / "main.py").write_text("print(2)\n")
    assert probe.code_revision(tmp_path)["git_dirty"] is True
