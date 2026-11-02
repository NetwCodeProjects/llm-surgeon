"""One-line results per stage for the normal console view (llm_surgeon.console).

Read from what each stage already writes (its receipt and metrics files), so a reused stage shows the same
line as the run that produced it. Never raises: a missing fact makes a shorter line, not a failed run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import formats as F


def stage_label(stage) -> str:
    return f"[{stage.id} {stage.name}]"


def human(n: float | None) -> str:
    if n is None:
        return "?"
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1000 or unit == "TB":
            return f"{value:.0f} {unit}" if unit in ("B", "KB") else f"{value:.1f} {unit}"
        value /= 1000
    return f"{value:.1f} TB"


def _metric(run, name: str) -> Any:
    path = Path(run.run_dir) / "metrics" / name
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _count(pair: tuple[Any, Any]) -> str:
    return f"{pair[0]}/{pair[1]}" if None not in pair else "?"


def preflight(run, receipt) -> str:
    doc = _metric(run, "preflight.json") or {}
    diag = receipt.get("diagnostics") or {}
    gpus = diag.get("gpus") or doc.get("gpus") or []
    gpu = f"{gpus[0].get('name')} (cc {gpus[0].get('compute_capability')})" + (f" ×{len(gpus)}" if len(gpus) > 1 else "") if gpus else "no GPU"
    free = ((doc.get("storage") or {}).get("workspace") or {}).get("free_bytes")
    return " · ".join(p for p in (diag.get("profile") or (doc.get("profile") or {}).get("resolved"), gpu, f"{human(free)} free" if free else None) if p)


def environment(run, receipt) -> str:
    actions = (receipt.get("diagnostics") or {}).get("component_actions") or {}
    tally: dict[str, int] = {}
    for action in actions.values():
        tally[action] = tally.get(action, 0) + 1
    if not tally:
        return "no tools needed"
    if len(tally) == 1:
        ((action, n),) = tally.items()
        return f"{n} tool{'s' if n != 1 else ''} {action}"
    return "tools: " + ", ".join(f"{n} {a}" for a, n in sorted(tally.items()))


def acquire(run, receipt) -> str:
    diag = receipt.get("diagnostics") or {}
    repo, rev = diag.get("source_repo"), diag.get("source_revision")
    where = f"{repo} @ {rev[:7]}" if repo and rev else (diag.get("source_path") or "")
    size = diag.get("source_total_size")
    return f"{where} · {human(size)} verified" if size else where


def prepare(run, receipt) -> str:
    diag = receipt.get("diagnostics") or {}
    modifiers = ", ".join(m.split(".")[-1] for m in diag.get("modifiers") or [])
    return " · ".join(p for p in (diag.get("label"), modifiers, f"seed {diag['seed']}" if diag.get("seed") is not None else None) if p)


def abliterate(run, receipt) -> str:
    summary = (_metric(run, "candidate.json") or {}).get("summary") or {}
    if summary.get("exported_trial_index") is None:
        return "reproduced from the given parameters" if summary.get("trials_total") == 0 else ""
    refusals = summary.get("optimizer_refusal_rate")
    kl = summary.get("optimizer_kl")
    facts = ", ".join(p for p in (f"refusals {refusals:.0%}" if refusals is not None else None, f"KL {kl:.3f}" if kl is not None else None) if p)
    return f"{summary.get('trials_complete')} trials · trial {summary['exported_trial_index']} selected" + (f" ({facts})" if facts else "")


def qualify(run, receipt) -> str:
    ind = (_metric(run, "qualification.json") or {}).get("independent") or {}
    if not ind:
        return ""
    graded = ind.get("benign_graded") or 0
    benign = round((ind.get("benign_pass_rate") or 0) * graded)
    integrity = "integrity ok" if ind.get("integrity_passed") else f"{len(ind.get('integrity_failures') or [])} integrity problems"
    return f"held-out refused {_count((ind.get('heldout_refusal_count'), ind.get('heldout_refusal_prompts')))} · benign {benign}/{graded} · {integrity}"


def select_master(run, receipt) -> str:
    outcome = (_metric(run, "selection.json") or {}).get("outcome") or (receipt.get("diagnostics") or {}).get("promotion")
    return {"promoted:qualified": "promoted (every gate passed)", "not_promoted": "not promoted"}.get(outcome, outcome or "")


def quantize(run, receipt) -> str:
    diag = receipt.get("diagnostics") or {}
    built = [F.describe(f) for f in (diag.get("results") or {})]
    failed = [F.describe(f) for f in (diag.get("failed") or {})]
    return ", ".join(built) + (f" · FAILED: {', '.join(failed)}" if failed else "") if built or failed else ""


def verify(run, receipt) -> str:
    verification = _metric(run, "verification.json") or {}
    exports = {r: (i.get("smoke") or {}) for r, i in verification.items() if r.startswith("export:")}
    if not exports:
        return ""
    loaded = sum(1 for s in exports.values() if s.get("passed"))
    off = sum(1 for s in exports.values() if not s.get("ran") and s.get("reason") == "verify_backend=none")
    parts = [f"{loaded}/{len(exports)} exports loaded" + (f" ({off} loader off)" if off else "")]
    comparison = (_metric(run, "comparison.json") or {}).get("exports") or {}
    if comparison:
        passed = sum(1 for e in comparison.values() if e.get("outcome") == "passed")
        parts.append(f"comparison {passed}/{len(comparison)} passed")
    return " · ".join(parts)


def report(run, receipt) -> str:
    return "report written"


SUMMARIES = {
    "00": preflight,
    "01": environment,
    "02": acquire,
    "04": prepare,
    "05": abliterate,
    "06": qualify,
    "07": select_master,
    "08": quantize,
    "09": verify,
    "10": report,
}


def stage_summary(run, stage_id: str, receipt: dict[str, Any] | None) -> str:
    fn = SUMMARIES.get(stage_id)
    if fn is None or not receipt:
        return ""
    try:
        return fn(run, receipt)
    except Exception:  # noqa: BLE001 - a summary line must never fail a run
        return ""
