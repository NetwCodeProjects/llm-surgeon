"""A search with no trial within the gates names the closest one and the command that uses it; that command
works: a fork that raises the missed gate just enough re-runs selection and export only, and promotes."""

from __future__ import annotations

import json
import re

from llm_surgeon.errors import ExitCode
from llm_surgeon.stages.stage05_abliterate import no_eligible_hint


def test_the_hint_raises_only_the_gate_the_closest_trial_misses():
    doc = {"closest": [{"index": 28, "pareto_position": 2, "refusal_rate": 0.04, "kl": 0.1004}]}
    hint = no_eligible_hint(doc, "R1", 0.2, 0.1)
    assert "to use trial 28 (the closest)" in hint
    assert "./main.sh --resume R1 --fork --set selection.max_optimizer_kl=0.101 " in hint and "refusal_rate" not in hint
    off_front = {"closest": [{"index": 5, "pareto_position": None, "refusal_rate": 0.3, "kl": 0.2}]}
    assert "to use trial" not in no_eligible_hint(off_front, "R1", 0.2, 0.1), "only Pareto-front trials can be exported"
    assert "--max-runtime-minutes" in no_eligible_hint(off_front, "R1", 0.2, 0.1)


def test_following_the_hint_promotes_the_closest_trial_without_new_trials(env, monkeypatch):
    monkeypatch.setenv("FAKE_KL_STEP", "0.051")  # every trial misses a gate
    env.provision()
    code, out = env.run()
    assert code == ExitCode.GATE and "E_NO_ELIGIBLE_TRIAL" in out, out
    match = re.search(r"to use trial (\d+) \(the closest\), raise the gate it misses just enough: \./main\.sh (--resume \S+ --fork (?:--set \S+ )+)", out)
    assert match, out
    parent = env.latest_run().name
    code, out = env.run(*match.group(2).split())
    assert code == 0, out
    fork = env.latest_run()
    assert fork.name != parent
    candidate = json.loads((fork / "metrics" / "candidate.json").read_text())
    assert candidate["summary"]["trials_complete"] == 50, "zero new trials"
    assert json.loads((fork / "metrics" / "selection.json").read_text())["outcome"] == "promoted:qualified"
