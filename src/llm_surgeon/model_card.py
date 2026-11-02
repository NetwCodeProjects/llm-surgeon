"""Hugging Face model cards for a published run (docs/release.md).

Every number on a card comes from a file the run wrote (candidate, qualification, comparison and
verification metrics, manifests); the card says which kind of measurement each one is. Pure functions:
``facts`` is gathered by ``llm_surgeon.publish``.
"""

from __future__ import annotations

from typing import Any

import yaml

PROJECT_URL = "https://github.com/NetwCodeProjects/llm-surgeon"
HERETIC_URL = "https://github.com/p-e-w/heretic"

NOTICE = (
    "This model's refusal behaviour was deliberately reduced. It can produce content the original model declines. "
    "You are responsible for how you use it, and for complying with the license and with applicable law."
)


def human(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1000 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1000
    return f"{value:.1f} TB"


def pair(counts: dict[str, int] | None) -> str:
    return f"{counts['count']}/{counts['of']}" if counts else "n/a"


def front_matter(kind: str, facts: dict[str, Any], base_repo: str | None) -> str:
    """``base_repo``: the published master, or None when only exports are published (then the source model is
    the base, with no relation: the export is abliterated and quantized, not a plain quantization of it)."""
    source = facts["source"]
    meta: dict[str, Any] = {"license": source.get("license") or "other"}
    if source.get("license_name"):
        meta["license_name"] = source["license_name"]
    if kind == "master":
        meta["base_model"] = [source["repo"]]
        meta["library_name"] = "transformers"
        tags = ["abliterated", "heretic", "llm-surgeon"]
    else:
        meta["base_model"] = [base_repo or source["repo"]]
        if base_repo:
            meta["base_model_relation"] = "quantized"
        if kind == "fp8":
            meta["library_name"] = "transformers"
            tags = ["abliterated", "heretic", "llm-surgeon", "fp8", "compressed-tensors", "vllm"]
        else:
            tags = ["abliterated", "heretic", "llm-surgeon", "gguf", "llama.cpp"]
    if source.get("pipeline_tag"):
        meta["pipeline_tag"] = source["pipeline_tag"]
    meta["tags"] = tags
    return "---\n" + yaml.safe_dump(meta, sort_keys=False, allow_unicode=True) + "---\n"


def results_section(kind: str, roles: list[str], facts: dict[str, Any]) -> list[str]:
    opt, qual = facts["optimizer"], facts["qualification"]
    lines = ["## Results", ""]
    if kind == "master":
        lines += [
            "| Measure | Original | This model | Kind of measurement |",
            "|---|---|---|---|",
            *[f"| {s['name']} on Heretic's evaluation prompts | {s['baseline']} | {s['score']} | Heretic's in-loop optimizer score |" for s in opt["scores"]],
            f"| Refusals on held-out harmful prompts | n/a | {qual['heldout']} | independent check (keyword heuristic) |",
            f"| Benign prompts answered correctly | n/a | {qual['benign']} | independent check |",
            f"| Responses with integrity problems | n/a | {qual['integrity']} | independent check |",
            "",
        ]
    else:
        base = facts["comparison"]["baseline"]
        lines += [
            "The same rendered prompts on the BF16 master (transformers) and on this export (its own runtime), greedy decoding:",
            "",
            "| Export | Runtime | Benign correct | Held-out refused | Integrity problems | Comparison |",
            "|---|---|---|---|---|---|",
            f"| BF16 master | transformers | {pair(base.get('benign_correct'))} | {pair(base.get('heldout_refused'))} | {pair(base.get('integrity_issues'))} | baseline |",
        ]
        for role in roles:
            row = facts["comparison"]["exports"].get(role) or {}
            loader = facts["verification"].get(role) or {}
            lines.append(
                f"| {role.split(':')[-1].upper()} | {loader.get('backend')} | {pair(row.get('benign_correct'))} | {pair(row.get('heldout_refused'))} | "
                f"{pair(row.get('integrity_issues'))} | {row.get('outcome')} |"
            )
        lines += [""]
        for role in roles:
            loader = facts["verification"].get(role) or {}
            lines.append(
                f"- {role.split(':')[-1].upper()}: loaded and generated with {loader.get('backend_revision')} "
                f"(modalities tested: {', '.join(loader.get('modalities') or []) or 'none'}; context tested: {loader.get('tested_context')})."
            )
        lines += [""]
    lines += ["Caveats:", "", *[f"- {c}" for c in qual["caveats"]], ""]
    return lines


def usage_section(kind: str, repo_id: str, files: list[dict[str, Any]]) -> list[str]:
    if kind == "master":
        return [
            "## Usage",
            "",
            "```python",
            "from transformers import AutoModelForImageTextToText, AutoProcessor",
            "",
            f'model = AutoModelForImageTextToText.from_pretrained("{repo_id}", dtype="auto", device_map="auto")',
            f'processor = AutoProcessor.from_pretrained("{repo_id}")',
            "```",
            "",
        ]
    if kind == "fp8":
        return ["## Usage", "", "```bash", f"vllm serve {repo_id}", "```", "", "Native FP8 needs compute capability 8.9 or newer (Ada, Hopper, Blackwell).", ""]
    model = next((f["path"] for f in files if f["path"].endswith(".gguf") and not f["path"].startswith("mmproj")), "<model>.gguf")
    mmproj = next((f["path"] for f in files if f["path"].startswith("mmproj")), None)
    return [
        "## Usage",
        "",
        "```bash",
        f"llama-server -hf {repo_id} -c 8192" if mmproj else f"llama-server -m {model} -c 8192",
        "```",
        "",
        f"Text: `{model}`." + (f" Images: add the projector `{mmproj}` (`--mmproj`)." if mmproj else ""),
        "",
    ]


def render(kind: str, repo_id: str, base_repo: str | None, roles: list[str], files: list[dict[str, Any]], facts: dict[str, Any]) -> str:
    source, prov, opt = facts["source"], facts["provenance"], facts["optimizer"]
    source_link = f"[{source['repo']}](https://huggingface.co/{source['repo']}) (revision `{source['revision'][:12]}`)"
    if kind == "master":
        title, intro = (
            repo_id.split("/")[-1],
            f"Abliterated version of {source_link}: refusal behaviour reduced with [Heretic]({HERETIC_URL}) "
            f"({', '.join(m.split('.')[-1] for m in opt['modifiers']) or 'directional ablation'}), in BF16.",
        )
    else:
        label = "FP8 (FP8_BLOCK, compressed-tensors)" if kind == "fp8" else "GGUF"
        master = (
            f"[{base_repo}](https://huggingface.co/{base_repo}), the abliterated {source_link}"
            if base_repo
            else f"the abliterated {source_link} (BF16 master not published)"
        )
        title, intro = repo_id.split("/")[-1], f"{label} quantization of {master}."
    lines = [
        front_matter(kind, facts, base_repo),
        f"# {title}",
        "",
        intro,
        f"Produced and verified with [llm-surgeon]({PROJECT_URL}) {prov['version']} (commit `{(prov['git_commit'] or 'unknown')[:12]}`), run `{prov['run_id']}`.",
        "",
        f"> {NOTICE}",
        "",
        *results_section(kind, roles, facts),
        "## Files",
        "",
        "| File | Size | sha256 |",
        "|---|---:|---|",
        *[f"| `{f['path']}` | {human(f['size_bytes'])} | `{f['sha256']}` |" for f in files if f["path"] != "README.md"],
        "",
        *usage_section(kind, repo_id, files),
        "## How it was made",
        "",
        f"1. Source pinned to commit `{source['revision']}` and verified file by file (sha256).",
        f"2. Heretic `{opt['engine_commit'][:12]}` searched {opt['trials_complete']} trials; trial {opt['trial']} was selected by the "
        f"policy `{opt['policy']}` (refusal rate ≤ {opt['constraints'].get('max_optimizer_refusal_rate')}, KL ≤ {opt['constraints'].get('max_optimizer_kl')}).",
        f"3. Independent qualification on held-out prompts; promotion `{facts['promotion']}` (every required gate passed).",
    ]
    if facts["grafted"]:
        lines.append(
            f"4. Heretic's merged export drops {facts['grafted']} tensors (`mtp.*`, the multi-token-prediction head); they were copied "
            "unchanged from the source, so speculative decoding still works (its acceptance rate may differ slightly)."
        )
    lines += [
        f"{5 if facts['grafted'] else 4}. Exports re-hashed, loaded in their runtimes and compared with the master on the same prompts.",
        "",
        "## Provenance",
        "",
        f"- llm-surgeon {prov['version']}, commit `{prov['git_commit']}`; config hash `{prov['config_hash']}`",
        *[f"- {name}: {', '.join(f'{k} {v}' for k, v in sorted(versions.items()) if isinstance(v, str))}" for name, versions in sorted(prov["tools"].items())],
        f"- GPU: {', '.join(prov['gpus']) or 'n/a'}",
        "- `llm-surgeon/` holds the artifact manifests (checksums and lineage) and the run's release record.",
        "",
    ]
    return "\n".join(lines)
