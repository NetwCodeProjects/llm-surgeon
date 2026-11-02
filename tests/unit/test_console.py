"""Console output (docs/cli.md#console-output): the trimmed default view, the live line, the search progress."""

from __future__ import annotations

import io
import re

from llm_surgeon.adapters.heretic import TrialWatch
from llm_surgeon.console import Console, duration

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


def lines(console_stream) -> list[str]:
    return [ANSI.sub("", line).strip() for line in console_stream.getvalue().replace("\r", "\n").splitlines() if ANSI.sub("", line).strip()]


def test_normal_view_hides_messages_and_prints_one_result_line_per_stage():
    out = io.StringIO()
    console = Console(out, "normal", tty=False)
    console.stage_start("[01 environment]")
    console.message("info", "Component fp8: reused (b0536a03c1a8)")
    console.message("info", "[01 environment] succeeded in 37.4s", lifecycle=True)
    console.message("warning", "Optional host tool missing: nvcc")
    console.stage_end("[01 environment]", "succeeded", 37.4, "3 tools reused")
    assert lines(out) == ["[01 environment] started", "WARN Optional host tool missing: nvcc", "[01 environment]       ok         37.4s  3 tools reused"]


def test_verbose_view_prints_every_message_and_no_result_lines():
    out = io.StringIO()
    console = Console(out, "info", tty=False)
    console.stage_start("[01 environment]")
    console.message("info", "[01 environment] starting", lifecycle=True)
    console.message("info", "Component fp8: reused (b0536a03c1a8)")
    console.milestone("trial 1: 3/25 refusals, KL 0.094 ✓ usable")
    console.stage_end("[01 environment]", "succeeded", 37.4, "3 tools reused")
    assert lines(out) == ["[01 environment] starting", "Component fp8: reused (b0536a03c1a8)", "trial 1: 3/25 refusals, KL 0.094 ✓ usable"]


def test_quiet_view_prints_warnings_and_errors_only():
    out = io.StringIO()
    console = Console(out, "warning", tty=False)
    console.stage_start("[05 abliterate]")
    console.message("info", "Searching")
    console.milestone("trial 1: 3/25 refusals, KL 0.094")
    console.message("warning", "Budget of 45 min reached")
    console.stage_end("[05 abliterate]", "failed", 2700, "E_NO_ELIGIBLE_TRIAL")
    assert lines(out) == ["WARN Budget of 45 min reached"]


def test_on_a_terminal_the_live_line_is_redrawn_and_cleared_before_other_lines():
    out = Tty()
    console = Console(out, "normal", tick=3600)
    console.stage_start("[05 abliterate]")
    console.activity("trial 3/200")
    console.message("warning", "something")
    console.stage_end("[05 abliterate]", "succeeded", 125, "39 trials")
    raw = out.getvalue()
    assert "\r\x1b[2K[05 abliterate] 0:00 · trial 3/200" in raw
    assert raw.index("trial 3/200") < raw.index("WARN") and "\x1b[33mWARN\x1b[0m something\n" in raw
    assert raw.rstrip().endswith("[05 abliterate]        \x1b[32mok      \x1b[0m   2m05s  39 trials")
    assert "started" not in raw, "the live line replaces the start line on a terminal"


def test_no_color_and_pipes_get_plain_text(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    out = Tty()
    Console(out, "normal").message("warning", "x")
    assert out.getvalue() == "WARN x\n"


def test_durations():
    assert [duration(s) for s in (0.6, 37.44, 125, 3725)] == ["0.6s", "37.4s", "2m05s", "1h02m"]


# ------------------------------------------------------------------------------------------ the search
class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def watch(budget=None):
    clock, reports, milestones = Clock(), [], []
    w = TrialWatch(reports.append, milestones.append, max_refusal_rate=0.2, max_kl=0.1, budget_seconds=budget, clock=clock)
    return w, clock, reports, milestones


def trial(w, clock, n, of, refusals, kl, seconds):
    w.on_line("stdout", f"Running trial {n} of {of}...")
    clock.now += seconds
    w.on_line("stdout", f"  * Refusals: {refusals}/25")
    w.on_line("stdout", f"  * KL divergence: {kl}")


def test_trial_progress_best_usable_and_estimate():
    w, clock, reports, milestones = watch()
    w.on_line("stdout", "Loading model Qwen/Qwen3.5-0.8B...")
    w.on_line("stdout", "  * Refusals: 24/25")  # the baseline before trial 1 is not a trial
    assert reports == ["loading the model"]
    w.on_line("stdout", "Running trial 1 of 10...")
    assert w.status() == "trial 1/10 · time left after the first trial"
    clock.now = 240
    w.on_line("stdout", "  * Refusals: 6/25")
    w.on_line("stdout", "  * KL divergence: 0.2100")
    assert milestones == ["trial 1: 6/25 refusals, KL 0.210"]
    assert w.status() == "trial 1/10 · ~36m left · none usable yet (closest: 6/25, KL 0.210)"
    trial(w, clock, 2, 10, 3, 0.094, 240)
    trial(w, clock, 3, 10, 1, 0.104, 240)
    assert milestones[1] == "trial 2: 3/25 refusals, KL 0.094 ✓ usable"
    assert w.status() == "trial 3/10 · ~28m left · best usable: trial 2 (3/25, KL 0.094)"


def test_the_budget_caps_the_estimate():
    w, clock, _, _ = watch(budget=45 * 60)
    trial(w, clock, 1, 200, 6, 0.21, 240)
    assert w.eta() == "~41m left (budget)"


def test_export_phases_are_reported():
    w, _, reports, _ = watch()
    w.on_line("stdout", "Restoring model from trial 6...")
    w.on_line("stdout", "Saving merged model...")
    assert reports == ["exporting trial 6", "saving the merged model"]
