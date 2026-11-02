# Contributing

## Setup

```text
Install:  uv sync --frozen                      (or ./main.sh --bootstrap --yes, then .venv/bin/pip install -r requirements/dev.txt)
```

## Checks before a commit

```text
Tests:    .venv/bin/python -m pytest                       CPU, fake engines, ~15 min; -m 'not gpu and not realtools' by default
Lint:     .venv/bin/ruff check src tests scripts && .venv/bin/ruff format src tests scripts
Schema:   python scripts/generate_schema.py --check         after a change to src/llm_surgeon/configspec.py (without --check: regenerate)
```

Run the tests of the area you changed while working; the full suite before a broad change is committed.

## Real tools

```text
CPU:      REALTOOLS_WS=/tmp/ws python tests/realtools/provision_cpu.py && REALTOOLS_WS=/tmp/ws pytest -m realtools tests/realtools
GPU:      pytest -m gpu tests/gpu                           B200/B300; other GPUs need GPU_ALLOW_OTHER=1
```

A commit that changes what runs is accepted with a real CPU run before any GPU run: [docs/acceptance.md](docs/acceptance.md).

## Conventions

- Files are LF (`.gitattributes`); generated files too.
- Facts about pinned tools (flags, output formats, file layouts) are verified against the pinned version and
  say so in a comment or docstring.
- Findings, decisions and measurements go into [docs/implementation-log.md](docs/implementation-log.md) as a
  dated entry; reference docs describe the current behavior only.

## Writing

Maximum clarity with minimum wording, for readers who know the tools.

- Explain the action, not the name: `Read: config.yaml`.
- Keep related actions in one aligned block.
- Give each fact one home. README: purpose, requirements, common commands, outputs, limitations. Startup
  screen: hardware context and next actions. Reference docs: settings, mechanisms, edge cases, troubleshooting.
- Keep requirements, consequences, compatibility limits and uncertainty when they affect a decision.
- Structure over words; the layout must read without color.

Every line should provide useful context, support a decision, or enable an action.

## Scope

Project B (coding-capability research) is not part of this repository; it reads this project's manifests
(`runs/<run>/reports/manifest-index.json`), never its code.
