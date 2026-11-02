"""Publishing choices at the start of a run, and the upload at its end (docs/release.md).

With ``publish.enabled: ask`` a new run on an interactive terminal asks, right after the banner, whether to
publish. On yes: the login (``hf auth login`` from ``.venv`` when there is no token), the repository name, its
visibility (private or public) and, when ``publish.formats`` is null, the formats, picked by bit level. ``--yes`` never answers these questions,
and without a terminal nothing is published. The answers are stored in the run's resolved configuration
(``publish.*`` keys are operational, so they never change a stage key); a resumed run uses them.

Whatever would only fail hours later stops the run here instead: no token or a read-only one, a repository
name that is taken, a format whose release check cannot pass on this host, uncommitted code. At the end, a run
that opted in is uploaded, with the chosen visibility, only when its release record says ready.
"""

from __future__ import annotations

import math
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from . import configspec
from . import formats as F
from .config import PROJECT_ROOT, ResolvedConfig, deep_merge, finalize, overrides_to_tree
from .errors import ExitCode, PipelineError
from .model_card import human

NAME_RE = re.compile(rf"^{configspec.HF_NAME_PATTERN}$")
SUFFIX = {"master": "", "fp8": "-FP8", "gguf": "-GGUF"}
EXPIRY_WARNING = timedelta(days=2)


class Prompter:
    """Questions on the run's output stream, answers from stdin (or ``services.ask`` in tests)."""

    def __init__(self, services, stream) -> None:
        self.answer = services.ask
        self.stream = stream

    def say(self, text: str = "") -> None:
        self.stream.write(text + "\n")
        self.stream.flush()

    def ask(self, prompt: str) -> str:
        if self.answer is not None:
            return self.answer(prompt).strip()
        self.stream.write(prompt)
        self.stream.flush()
        line = sys.stdin.readline()
        if not line:
            raise PipelineError(ExitCode.CONFIRMATION_REQUIRED, "E_NOT_CONFIRMED", "No answer (end of input)")
        return line.strip()

    def yes_no(self, question: str, default: bool) -> bool:
        while True:
            answer = self.ask(f"{question} {'[Y/n]' if default else '[y/N]'} ").lower()
            if not answer:
                return default
            if answer in ("y", "yes"):
                return True
            if answer in ("n", "no"):
                return False
            self.say("  Please answer y or n.")


# ------------------------------------------------------------------------------------------ what can be published
def not_applicable(config: ResolvedConfig, stage_ids: list[str]) -> str | None:
    if not config.get("abliteration.enabled"):
        return "abliteration is off, so there is no abliterated master to publish"
    if "10" not in stage_ids:
        return "this invocation stops before the last stage"
    return None


def format_blockers(config: ResolvedConfig) -> dict[str, str]:
    """Formats whose release check cannot pass with this configuration, and why."""
    out = {}
    if config.get("quantization.fp8.verify_backend") == "none":
        out["fp8"] = "its loader test is off here (quantization.fp8.verify_backend: none)"
    if not config.get("quantization.gguf.imatrix_dataset"):
        for name in configspec.GGUF_TYPES_REQUIRING_IMATRIX:
            out[name] = "needs an importance matrix: set quantization.gguf.imatrix_dataset"
    return out


def run_blockers(config: ResolvedConfig, chosen: list[str], code: dict[str, Any]) -> list[str]:
    """Run-wide reasons the release record could not say ready."""
    out = []
    if config.get("evaluation.qualification") == "off":
        out.append("qualification is off (evaluation.qualification), so the master cannot be promoted as qualified")
    elif config.get("selection.profile") != "qualified":
        out.append(f"selection.profile is {config.get('selection.profile')!r}; publishing needs a master promoted as qualified")
    if any(f != "master" for f in chosen) and not config.get("evaluation.comparison.enabled"):
        out.append("the master/export comparison is off (evaluation.comparison.enabled)")
    if not code.get("git_commit"):
        out.append("this is not a git checkout; the release record needs the commit the run used")
    elif code.get("git_dirty"):
        out.append("the checkout has uncommitted code changes (git status); the release record needs a clean commit (configs/ do not count)")
    return out


def stored_visibility(value: str | None) -> str:
    """The stored choice; 'ask' that was never answered (no terminal) is private."""
    return "public" if value == "public" else "private"


def repo_ids(namespace: str, name: str, chosen: list[str]) -> list[str]:
    base = f"{namespace}/{name}"
    kinds = []
    if "master" in chosen:
        kinds.append("master")
    if "fp8" in chosen:
        kinds.append("fp8")
    if any(f not in ("master", "fp8") for f in chosen):
        kinds.append("gguf")
    return [base + SUFFIX[k] for k in kinds]


def default_repo_name(config: ResolvedConfig) -> str:
    url, path = config.get("model.url"), config.get("model.local_path")
    return f"{url.split('/')[-1] if url else Path(path).name}-abliterated"


def parameter_count(config: ResolvedConfig, account, token: str | None) -> int | None:
    if config.get("model.url"):
        return account.parameter_count(config.get("model.url"), config.get("model.revision"), token)
    try:
        from .artifacts.safetensors_meta import tensor_map  # noqa: PLC0415

        return sum(math.prod(t["shape"]) for t in tensor_map(config.get("model.local_path")).values()) or None
    except Exception:  # noqa: BLE001 - only used for size estimates
        return None


# ------------------------------------------------------------------------------------------ login
def login(p: Prompter, account, token_env: str, interactive: bool) -> tuple[str, dict[str, Any]]:
    token = account.token(token_env)
    who = account.whoami(token) if token else None
    if who is not None:
        return token, who
    if not interactive:
        raise PipelineError(
            ExitCode.CONFIG,
            "E_HF_TOKEN_MISSING",
            "No Hugging Face login" if not token else "The Hugging Face token was not accepted",
            hint=f"Run .venv/bin/hf auth login, or export a write token in ${token_env}; or set publish.enabled: false.",
        )
    p.say(
        "  Not logged in: running `hf auth login` (paste a write token; it is not shown)"
        if not token
        else "  The token was not accepted: running `hf auth login`"
    )
    account.login()
    token = account.token(token_env)
    who = account.whoami(token) if token else None
    if who is None:
        raise PipelineError(
            ExitCode.CONFIG,
            "E_HF_LOGIN_FAILED",
            "Still not logged in to Hugging Face",
            hint=f"A token in ${token_env} takes precedence over the stored login; unset it or replace it.",
        )
    return token, who


def access(who: dict[str, Any], namespace: str) -> str:
    """Raise when the login cannot create repositories in ``namespace``; otherwise say what is known."""
    names = [who.get("name")] + [o.get("name") for o in who.get("orgs") or []]
    if namespace not in names:
        raise PipelineError(
            ExitCode.CONFIG,
            "E_PUBLISH_NAMESPACE",
            f"Logged in as {who.get('name')}; {namespace!r} is not this account or one of its organizations ({', '.join(n for n in names if n)})",
            hint="Set publish.namespace (or --hf-namespace) to your account or an organization you belong to.",
        )
    token = (who.get("auth") or {}).get("accessToken") or {}
    role = token.get("role")
    if role == "read":
        raise PipelineError(
            ExitCode.CONFIG, "E_HF_TOKEN_READ_ONLY", "The Hugging Face token is read-only", hint="Log in with a write token: .venv/bin/hf auth login"
        )
    if role == "fineGrained":
        scoped = (token.get("fineGrained") or {}).get("scoped") or []
        if not any((s.get("entity") or {}).get("name") == namespace and "repo.write" in (s.get("permissions") or []) for s in scoped):
            raise PipelineError(
                ExitCode.CONFIG,
                "E_HF_TOKEN_READ_ONLY",
                f"The fine-grained token has no repo.write permission for {namespace}",
                hint="Add 'Write access to contents/settings of all repos' for that namespace, or log in with a write token.",
            )
        return "write access ok"
    if role == "write":
        return "write access ok"
    return "write access is confirmed when the repositories are created"  # OAuth logins report no role


def expiry_note(who: dict[str, Any], now: datetime | None = None) -> str | None:
    text = (who.get("auth") or {}).get("expiresAt")
    if not text:
        return None
    try:
        when = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    now = now or datetime.now(timezone.utc)
    if when - now < EXPIRY_WARNING:
        return f"WARNING: this login expires {when:%Y-%m-%d %H:%M} UTC; the upload at the end of the run needs it (log in again with a longer-lived token)"
    return f"login valid until {when:%Y-%m-%d}"


# ------------------------------------------------------------------------------------------ the picker
def parse_numbers(text: str, count: int) -> list[int] | None:
    try:
        numbers = [int(part) for part in text.replace(" ", "").split(",") if part]
    except ValueError:
        return None
    if not numbers or any(n < 1 or n > count for n in numbers):
        return None
    return list(dict.fromkeys(numbers))


def size(name: str, parameters: int | None) -> str:
    estimate = F.estimate_bytes(name, parameters)
    return f"~{human(estimate)}" if estimate else ""


def pick(p: Prompter, title: str, rows: list[str], default: list[int], allowed=lambda i: None) -> list[int]:
    p.say("")
    p.say(title)
    for i, row in enumerate(rows, 1):
        p.say(f"  {i}) {row}")
    while True:
        answer = p.ask(f"Choice [{','.join(map(str, default))}]: ")
        numbers = default if not answer else parse_numbers(answer, len(rows))
        if not numbers:
            p.say(f"  Enter numbers from 1 to {len(rows)}, separated by commas.")
            continue
        refused = [(n, allowed(n)) for n in numbers if allowed(n)]
        if refused:
            for n, reason in refused:
                p.say(f"  {n} cannot be published here: {reason}")
            continue
        return numbers


def pick_formats(p: Prompter, defaults: list[str], parameters: int | None, blocked: dict[str, str]) -> list[str]:
    """Bit levels first, then the formats inside each chosen level (a level with one format needs no second question)."""
    level_rows = []
    for bits in F.LEVELS:
        entries = F.level(bits)
        if len(entries) == 1:
            level_rows.append(f"{bits}-bit".ljust(9) + f"{entries[0].label} ({entries[0].note})".ljust(52) + size(entries[0].name, parameters))
        else:
            level_rows.append(f"{bits}-bit".ljust(9) + f"{len(entries)} formats")
    default_levels = [i for i, bits in enumerate(F.LEVELS, 1) if any(f.name in defaults for f in F.level(bits))] or [1]
    chosen: list[str] = []
    for index in pick(p, "Which precisions? (one or more, separated by commas)", level_rows, default_levels):
        entries = F.level(F.LEVELS[index - 1])
        if len(entries) == 1:
            if entries[0].name in blocked:
                p.say(f"  {entries[0].label} cannot be published here: {blocked[entries[0].name]}")
                continue
            chosen.append(entries[0].name)
            continue
        rows = []
        for fmt in entries:
            note = blocked.get(fmt.name) or ("needs an importance matrix" if fmt.needs_imatrix else fmt.note)
            rows.append(f"{fmt.label:<9}{fmt.container:<13}{note:<52}{size(fmt.name, parameters)}")
        allowed = [i for i, f in enumerate(entries, 1) if f.name not in blocked]
        default = [i for i, f in enumerate(entries, 1) if f.name in defaults and f.name not in blocked] or allowed[:1]
        if not default:
            p.say(f"  No {entries[0].bits}-bit format can be published here.")
            continue
        picked = pick(p, f"{entries[0].bits}-bit: which formats?", rows, default, lambda n, entries=entries: blocked.get(entries[n - 1].name))
        chosen += [entries[n - 1].name for n in picked]
    return F.ordered(chosen)


# ------------------------------------------------------------------------------------------ start of the run
def ask_name(p: Prompter, current: str) -> str:
    while True:
        name = p.ask(f"Repository name [{current}]: ") or current
        if NAME_RE.match(name):
            return name
        p.say(f"  {name!r} is not a valid Hugging Face repository name (letters, digits, '.', '_', '-').")


def ask_visibility(p: Prompter) -> str:
    rows = [
        "private  only you can see it; make it public later on Hugging Face (counts toward your private storage)",
        "public   anyone can see and download it as soon as the upload starts",
    ]
    while True:
        picked = pick(p, "Visibility once uploaded:", rows, [1])
        if len(picked) == 1:
            return ("private", "public")[picked[0] - 1]
        p.say("  Choose one.")


def with_choices(config: ResolvedConfig, values: dict[str, Any]) -> ResolvedConfig:
    return finalize(deep_merge(config.data, overrides_to_tree(values)), [*config.sources, "publish choices"], profiles=config.profiles)


def at_start(config: ResolvedConfig, *, run_id: str, how: str, stage_ids: list[str], cli_formats: bool, services, stream) -> ResolvedConfig:
    """Ask (or check) the publishing choices before the first stage; returns the configuration to run with."""
    mode = config.get("publish.enabled")
    if mode is False:
        return config
    p = Prompter(services, stream)
    reason = not_applicable(config, stage_ids)
    interactive = services.interactive()
    token_env = config.get("model.hf_token_env") or "HF_TOKEN"
    if how != "new":
        if mode is True and reason is None:  # answered when the run started; make sure the login still works
            _token, who = login(p, services.account_factory(), token_env, interactive)
            namespace = config.get("publish.namespace") or who["name"]
            access(who, namespace)
            name = config.get("publish.repo_name") or default_repo_name(config)
            repos = repo_ids(namespace, name, config.get("publish.formats") or ["master"])
            p.say(f"Publishing: on, as chosen when the run started ({', '.join(repos)})")
        return config
    if reason is not None:
        if mode is True:
            p.say(f"Publishing: skipped ({reason})")
        return config
    asked = mode == "ask"
    if asked:
        if not interactive:
            return config  # the end of the run prints the publish command
        if not p.yes_no("Publish to Hugging Face when the run is ready?", False):
            return with_choices(config, {"publish.enabled": False})
    account = services.account_factory()
    token, who = login(p, account, token_env, interactive)
    namespace = config.get("publish.namespace") or who["name"]
    p.say(f"  Logged in as {who['name']} ({access(who, namespace)})")
    note = expiry_note(who)
    if note:
        p.say(f"  {note}")
    ask_here = interactive and asked
    name = config.get("publish.repo_name") or default_repo_name(config)
    if ask_here:
        name = ask_name(p, name)
    elif not NAME_RE.match(name):
        raise PipelineError(ExitCode.CONFIG, "E_PUBLISH_NAME", f"{name!r} is not a valid Hugging Face repository name", hint="Set publish.repo_name.")
    visibility = config.get("publish.visibility")
    if visibility == "ask":
        visibility = ask_visibility(p) if ask_here else "private"
    code = services.probe_factory(True).code_revision(PROJECT_ROOT)
    quant = config.get("quantization.formats") if config.get("quantization.enabled") else []
    blocked = format_blockers(config)
    parameters = parameter_count(config, account, token)
    overrides: dict[str, Any] = {}
    chosen = config.get("publish.formats")
    if chosen is None:
        if ask_here and not cli_formats:
            chosen = pick_formats(p, ["master", *quant], parameters, blocked)
            if not chosen:
                raise PipelineError(ExitCode.CONFIRMATION_REQUIRED, "E_NOT_CONFIRMED", "No format chosen to publish")
            exports = [f for f in chosen if f != "master"]
            if exports != quant:
                overrides.update({"quantization.formats": exports, "quantization.enabled": bool(exports)})
        else:
            chosen = ["master", *quant]
    problems = [f"{F.describe(f)} {blocked[f]}" for f in chosen if f in blocked] + run_blockers(config, chosen, code)
    if problems:
        raise PipelineError(
            ExitCode.CONFIG,
            "E_PUBLISH_IMPOSSIBLE",
            "This run could not be published: " + "; ".join(problems),
            hint="Change the configuration, or answer no to publishing (publish.enabled: false).",
        )
    while True:
        taken = [repo for repo in repo_ids(namespace, name, chosen) if account.repo_state(repo, token) == "content"]
        if not taken:
            break
        message = "; ".join(f"{repo} already exists with files" for repo in taken)
        if not ask_here:
            raise PipelineError(ExitCode.CONFIG, "E_PUBLISH_CONFLICT", message, hint="Choose another name (publish.repo_name).")
        p.say(f"  {message}. Choose another name.")
        name = ask_name(p, name)
    estimate = sum(F.estimate_bytes(f, parameters) or 0 for f in chosen)
    repos = repo_ids(namespace, name, chosen)
    p.say("")
    p.say("Summary")
    p.say(f"  Publish      : {', '.join(F.describe(f) for f in chosen)}")
    if "master" not in chosen:
        p.say("  Also built   : BF16 master (abliteration produces it; not published)")
    base = f"{namespace}/{name}"
    shown = [base, *(r[len(base) :] for r in repos[1:])] if repos[0] == base else repos
    p.say(f"  Repositories : {', '.join(shown)}   ({visibility}, names free)")
    p.say(f"  Upload       : {f'~{human(estimate)}, ' if estimate else ''}after the release record says ready")
    if visibility == "public":
        p.say("  PUBLIC       : anyone can see and download the files from the moment the upload starts")
    else:
        p.say("  Private      : counts toward your Hugging Face private storage; make it public later in each repository's settings")
    if "quantization.formats" in overrides:
        p.say(f"  These formats differ from the config file: after an interruption continue with ./main.sh --resume {run_id}")
    if ask_here and not p.yes_no("Continue?", True):
        raise PipelineError(ExitCode.CONFIRMATION_REQUIRED, "E_NOT_CONFIRMED", "Run cancelled at the publishing summary")
    overrides.update(
        {"publish.enabled": True, "publish.namespace": namespace, "publish.repo_name": name, "publish.formats": chosen, "publish.visibility": visibility}
    )
    return with_choices(config, overrides)


# ------------------------------------------------------------------------------------------ end of the run
def at_end(run, stage_ids: list[str], services, stream) -> int | None:
    """Upload a finished run that opted in; returns an exit code when publishing was attempted, else None."""
    config = run.config
    mode = config.get("publish.enabled")
    if mode is False or not_applicable(config, stage_ids):
        return None
    p = Prompter(services, stream)
    command = f"./main.sh --output-dir {run.workspace} --publish {run.run_id}"
    if mode == "ask":
        p.say(f"\nNot published (not asked: no interactive terminal). To publish this run: {command}")
        return None
    from .publish import build_plan, upload  # noqa: PLC0415

    p.say(f"\nPublishing ({stored_visibility(config.get('publish.visibility'))} repositories)")
    token_env = config.get("model.hf_token_env") or "HF_TOKEN"
    try:
        account = services.account_factory()
        token = account.token(token_env)
        who = account.whoami(token) if token else None
        if who is None:
            raise PipelineError(ExitCode.CONFIG, "E_HF_TOKEN_MISSING", "The Hugging Face login is gone", hint=f"Log in again, then: {command}")
        namespace = config.get("publish.namespace") or who["name"]  # null when publish.enabled: true came from the config
        plan, generated = build_plan(
            run.workspace,
            run.run_id,
            namespace,
            config.get("publish.repo_name"),
            config.get("publish.formats"),
            stored_visibility(config.get("publish.visibility")),
        )
        p.say(f"  Release record: ready; uploading {sum(len(r['files']) for r in plan['repos'])} files")
        return upload(run.workspace, run.run_id, plan, generated, token, services, stream)
    except PipelineError as exc:
        p.say(f"Not published: {exc}")
        p.say(f"  After fixing it: {command}")
        return int(exc.code)
