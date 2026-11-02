"""The default console view on a whole (fake) run: one result line per stage, no stage chatter, no
huggingface_hub progress bars; --verbose brings the messages back."""

from __future__ import annotations

import os
import re


def test_default_view_is_one_line_per_stage(env, capsys):
    env.provision()
    env.base_args.remove("--log-level")
    env.base_args.remove("warning")
    code, _ = env.run()
    assert code == 0
    err = capsys.readouterr().err
    stage_lines = [line for line in err.splitlines() if re.match(r"^\[\d\d \w+\]\s+(ok|reused|skipped|FAILED)\b", line)]
    assert [line.split("]")[0][1:3] for line in stage_lines] == ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    assert re.search(r"^Run \S+ \(new\)$", err, re.M), "the run line stays in the default view"
    by_stage = {line.split("]")[0][1:3]: line for line in stage_lines}
    assert "Qwen/Qwen3.8-27B @ 1d4bf0f" in by_stage["02"] and "verified" in by_stage["02"]
    assert "optimize · ARA · seed" in by_stage["04"]
    assert "trials · trial" in by_stage["05"] and "selected" in by_stage["05"]
    assert "promoted (every gate passed)" in by_stage["07"]
    assert "FP8, GGUF IQ4_XS, GGUF IQ3_S" in by_stage["08"]
    assert "exports loaded" in by_stage["09"]
    assert "Component " not in err and "Resolved " not in err and "Engine plan ready" not in err, "stage messages stay in the log"
    assert "[05 abliterate] trial 3: 17/100 refusals, KL 0.003 ✓ usable" in err, "without a terminal, each finished trial is a line"
    assert os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS") == "1"
    run_log = (env.latest_run() / "logs" / "run.log").read_text(encoding="utf-8")
    assert "Engine plan ready" in run_log and "[05 abliterate] succeeded" in run_log


def test_verbose_view_prints_the_messages(env, capsys):
    env.provision()
    env.base_args.remove("--log-level")
    env.base_args.remove("warning")
    code, _ = env.run("--verbose")
    assert code == 0
    err = capsys.readouterr().err
    assert "Engine plan ready" in err and "[05 abliterate] succeeded in" in err
