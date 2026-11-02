"""A search with no trial within the gates names the closest one and the commands that use it or search
more, for the run's own workspace; they work: a fork that raises the missed gate just enough re-runs
selection and export only, and promotes; a saved run directory continues its study on a new machine."""

from __future__ import annotations

import json
import re
import shlex
import shutil
from pathlib import Path

from llm_surgeon.errors import ExitCode
from llm_surgeon.stages.stage05_abliterate import no_eligible_hint


class Config(dict):
    def get(self, key, default=None):
        return super().get(key, default)


GATES = {"selection.max_optimizer_refusal_rate": 0.2, "selection.max_optimizer_kl": 0.1}


def printed(out: str, label: str) -> list[str]:
    """The arguments of a hint line's command, after ./main.sh (the note in parentheses dropped)."""
    match = re.search(rf"{label}: +\./main\.sh (.+)", out)
    assert match, out
    return shlex.split(match.group(1).split("   (")[0])


def test_the_hint_raises_only_the_gate_the_closest_trial_misses():
    ws = Path("/root/ws")
    config = Config({**GATES, "abliteration.max_trials": 50, "abliteration.max_runtime_minutes": 180})
    doc = {"closest": [{"index": 28, "pareto_position": 2, "refusal_rate": 0.04, "kl": 0.1004}], "trials": [{}] * 50}
    hint = no_eligible_hint(doc, "R1", ws, config)
    assert printed(hint, "Use closest") == ["--output-dir", str(ws), "--resume", "R1", "--fork", "--set", "selection.max_optimizer_kl=0.101"]
    assert "trial 28's value" in hint and "refusal_rate" not in hint
    assert printed(hint, "Search more")[2:] == ["--resume", "R1", "--fork", "--max-trials", "100"], "all trials ran: more trials"
    assert f"Keep study:   copy {ws}/runs/R1 out" in hint
    off_front = {"closest": [{"index": 5, "pareto_position": None, "refusal_rate": 0.3, "kl": 0.2}], "trials": [{}] * 41}
    hint = no_eligible_hint(off_front, "R1", ws, config)
    assert "Use closest" not in hint, "only Pareto-front trials can be exported"
    assert printed(hint, "Search more")[2:] == ["--resume", "R1", "--max-runtime-minutes", "360"], "the time limit stopped it: more time"


def test_following_the_hint_promotes_the_closest_trial_without_new_trials(env, monkeypatch):
    monkeypatch.setenv("FAKE_KL_STEP", "0.051")  # every trial misses a gate
    env.provision()
    code, out = env.run()
    assert code == ExitCode.GATE and "E_NO_ELIGIBLE_TRIAL" in out, out
    command = printed(out, "Use closest")
    assert command[:2] == ["--output-dir", str(env.workspace)], command
    parent = env.latest_run().name
    code, out = env.run(*command[2:])
    assert code == 0, out
    fork = env.latest_run()
    assert fork.name != parent
    candidate = json.loads((fork / "metrics" / "candidate.json").read_text())
    assert candidate["summary"]["trials_complete"] == 50, "zero new trials"
    assert json.loads((fork / "metrics" / "selection.json").read_text())["outcome"] == "promoted:qualified"


def test_a_saved_run_directory_continues_its_study_on_a_new_machine(env, monkeypatch):
    """The 'Keep study' line: only runs/<run> survives (a stopped Pod), restored to the same workspace path on
    a fresh machine; the printed 'Search more' command continues the same study, and only the new trials run."""
    monkeypatch.setenv("FAKE_KL_STEP", "0.051")  # every trial misses a gate; all 50 run
    env.provision()
    code, out = env.run()
    assert code == ExitCode.GATE and "E_NO_ELIGIBLE_TRIAL" in out, out
    command = printed(out, "Search more")
    assert command[:2] == ["--output-dir", str(env.workspace)] and "--fork" in command and "--max-trials" in command, command
    run = env.latest_run()
    before = json.loads((run / "metrics" / "selection-candidates.json").read_text())["trials"]

    saved = env.tmp / "saved"
    shutil.copytree(run, saved)
    shutil.rmtree(env.workspace)
    shutil.copytree(saved, env.workspace / "runs" / run.name)
    env.provision()  # a fresh install of the same pinned tools

    code, out = env.run(*command[2:])
    assert code == ExitCode.GATE and "E_NO_ELIGIBLE_TRIAL" in out, out  # still none within the gates, now with 100 trials
    fork = env.latest_run()
    after = json.loads((fork / "metrics" / "selection-candidates.json").read_text())["trials"]
    assert len(after) == 100 and [t["number"] for t in after][:50] == [t["number"] for t in before], "the old trials are kept"
    entries = [json.loads(line) for line in next((fork / "checkpoints" / "heretic").glob("study-*/study-history.jsonl")).read_text().splitlines()]
    added = [(e["action"], e.get("trials_added", e.get("trials_at_copy"))) for e in entries if e["phase"] == "completion"]
    assert ("fork-continue", 50) in added and ("continue", 50) in added and not any(a == "new" for a, _ in added), added
