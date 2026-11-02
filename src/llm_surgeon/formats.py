"""What can be built and published, grouped by bit level for the start-of-run picker (docs/release.md).

``bpw`` is the approximate bits per weight, used only for the size estimates the picker shows (labelled "~").
GGUF values follow llama.cpp's own size table for its quantization types; FP8 keeps the embeddings, the output
layer and the vision tower in BF16, so it lands above 8. The ternary types (TQ1_0, TQ2_0) are left out of the
picker: they suit models trained as ternary (e.g. BitNet). quantization.formats still accepts them.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import configspec


@dataclass(frozen=True)
class Format:
    name: str
    bits: int
    label: str
    container: str
    note: str
    bpw: float

    @property
    def needs_imatrix(self) -> bool:
        return self.name in configspec.GGUF_TYPES_REQUIRING_IMATRIX


MASTER = Format("master", 16, "BF16 master", "safetensors", "the abliterated model itself; transformers", 16.0)

# Within a level, the first entry is the usual choice (the default when the level is picked).
CATALOGUE: tuple[Format, ...] = (
    MASTER,
    Format("fp8", 8, "FP8", "safetensors", "vLLM on GPUs with FP8 (Ada, Hopper, Blackwell)", 8.5),
    Format("q8_0", 8, "Q8_0", "GGUF", "near-lossless; llama.cpp, LM Studio, Ollama", 8.5),
    Format("q6_k", 6, "Q6_K", "GGUF", "very close to 8-bit quality", 6.56),
    Format("q5_k_m", 5, "Q5_K_M", "GGUF", "a step above Q4_K_M in quality", 5.69),
    Format("q5_k_s", 5, "Q5_K_S", "GGUF", "a little smaller than Q5_K_M", 5.54),
    Format("q5_0", 5, "Q5_0", "GGUF", "legacy", 5.5),
    Format("q5_1", 5, "Q5_1", "GGUF", "legacy", 6.0),
    Format("q4_k_m", 4, "Q4_K_M", "GGUF", "the common default; good quality for the size", 4.89),
    Format("q4_k_s", 4, "Q4_K_S", "GGUF", "a little smaller, a little lower quality", 4.58),
    Format("iq4_xs", 4, "IQ4_XS", "GGUF", "smallest 4-bit; best with an importance matrix", 4.25),
    Format("iq4_nl", 4, "IQ4_NL", "GGUF", "non-linear 4-bit", 4.5),
    Format("q4_0", 4, "Q4_0", "GGUF", "legacy; fast on some CPUs and ARM", 4.5),
    Format("q4_1", 4, "Q4_1", "GGUF", "legacy", 5.0),
    Format("q3_k_m", 3, "Q3_K_M", "GGUF", "the usual 3-bit choice", 3.91),
    Format("q3_k_l", 3, "Q3_K_L", "GGUF", "larger, better 3-bit", 4.27),
    Format("q3_k_s", 3, "Q3_K_S", "GGUF", "smallest K-quant 3-bit", 3.5),
    Format("iq3_m", 3, "IQ3_M", "GGUF", "", 3.66),
    Format("iq3_s", 3, "IQ3_S", "GGUF", "", 3.44),
    Format("iq3_xs", 3, "IQ3_XS", "GGUF", "", 3.3),
    Format("iq3_xxs", 3, "IQ3_XXS", "GGUF", "", 3.06),
    Format("q2_k", 2, "Q2_K", "GGUF", "", 3.0),
    Format("iq2_m", 2, "IQ2_M", "GGUF", "", 2.7),
    Format("iq2_s", 2, "IQ2_S", "GGUF", "", 2.5),
    Format("iq2_xs", 2, "IQ2_XS", "GGUF", "", 2.31),
    Format("iq2_xxs", 2, "IQ2_XXS", "GGUF", "", 2.06),
    Format("q2_k_s", 2, "Q2_K_S", "GGUF", "", 2.8),
    Format("iq1_m", 1, "IQ1_M", "GGUF", "", 1.75),
    Format("iq1_s", 1, "IQ1_S", "GGUF", "", 1.56),
)
BY_NAME = {f.name: f for f in CATALOGUE}
LEVELS: tuple[int, ...] = tuple(sorted({f.bits for f in CATALOGUE}, reverse=True))


def level(bits: int) -> list[Format]:
    return [f for f in CATALOGUE if f.bits == bits]


def describe(name: str) -> str:
    """How a format is named in messages: 'BF16 master', 'FP8', 'GGUF Q4_K_M'."""
    fmt = BY_NAME.get(name)
    if fmt is None:
        return f"GGUF {name.upper()}"
    return fmt.label if fmt.container == "safetensors" else f"GGUF {fmt.label}"


def estimate_bytes(name: str, parameters: int | None) -> int | None:
    fmt = BY_NAME.get(name)
    if fmt is None or not parameters:
        return None
    return int(parameters * fmt.bpw / 8)


def ordered(names: list[str]) -> list[str]:
    """Catalogue order (master, then by bit level, highest first); unknown names keep their order at the end."""
    rank = {f.name: i for i, f in enumerate(CATALOGUE)}
    return sorted(dict.fromkeys(names), key=lambda n: rank.get(n, len(rank)))
