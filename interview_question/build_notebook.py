"""Combine the 14 transformer-component solution notebooks into a single notebook.

Reads each `*-{name}.ipynb` (the solved version, not `*-Question.ipynb`) under
`torch/transformer/NN-Name/` and concatenates them with section dividers.
"""

from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).parent.parent
SRC_ROOT = REPO / "torch" / "transformer"
OUT = REPO / "transformer_combined.ipynb"

# (folder, display title). Order matches numeric prefixes.
COMPONENTS = [
    ("01-Scaled-Dot-Product-Attention",     "01 · Scaled Dot-Product Attention"),
    ("02-Single-Head-Attention",            "02 · Single-Head Attention"),
    ("03-Self-Attention",                   "03 · Self-Attention"),
    ("04-Multi-Head-Attention",             "04 · Multi-Head Attention (MHA)"),
    ("05-Multi-Query-Attention",            "05 · Multi-Query Attention (MQA)"),
    ("06-Grouped-Query-Attention",          "06 · Grouped-Query Attention (GQA)"),
    ("07-Layer-Norm",                       "07 · LayerNorm"),
    ("08-RMS-Norm",                         "08 · RMSNorm"),
    ("09-Residual-Connection",              "09 · Residual Connection"),
    ("10-Feed-Forward-Network",             "10 · Feed-Forward Network"),
    ("11-Sinusoidal-Positional-Embedding",  "11 · Sinusoidal Positional Embedding"),
    ("12-Rotary-Positional-Embedding",      "12 · Rotary Positional Embedding (RoPE)"),
    ("13-KV-Cache",                         "13 · KV Cache"),
    ("14-Transformer-Decoder-Block",        "14 · Transformer Decoder Block"),
]


def find_solution_notebook(folder: Path) -> Path:
    """Pick the .ipynb that does NOT end in '-Question.ipynb'."""
    candidates = [p for p in folder.glob("*.ipynb") if not p.name.endswith("-Question.ipynb")]
    if not candidates:
        raise FileNotFoundError(f"No solution notebook in {folder}")
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple solution notebooks in {folder}: {candidates}")
    return candidates[0]


def make_md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def clean_cell(cell: dict) -> dict:
    """Normalize a cell: strip outputs/execution_count, drop kernel-specific metadata."""
    base = {
        "cell_type": cell["cell_type"],
        "metadata": {},
        "source": cell.get("source", []),
    }
    if cell["cell_type"] == "code":
        base["execution_count"] = None
        base["outputs"] = []
    return base


def build_notebook() -> dict:
    cells: list[dict] = []
    cells.append(
        make_md_cell(
            "# 🧠 Transformer Components — Combined Notebook\n"
            "\n"
            "All 14 transformer-component solutions from `torch/transformer/` in one notebook.\n"
            "\n"
            "**Sections (in order):**\n"
            "\n"
            + "\n".join(f"{i+1}. {title}" for i, (_, title) in enumerate(COMPONENTS))
            + "\n\n"
            "> 💡 Each section is self-contained. Run the cells top-to-bottom within a section. "
            "Re-running across sections will redefine classes (that's fine).\n"
        )
    )

    for folder_name, title in COMPONENTS:
        folder = SRC_ROOT / folder_name
        nb_path = find_solution_notebook(folder)
        with nb_path.open() as f:
            nb = json.load(f)

        # Section header
        cells.append(
            make_md_cell(
                f"\n---\n\n"
                f"# {title}\n\n"
                f"*Source: `torch/transformer/{folder_name}/{nb_path.name}`*\n"
                f"\n---\n"
            )
        )
        # Append the source notebook's cells (cleaned)
        for cell in nb.get("cells", []):
            cells.append(clean_cell(cell))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11",
            },
            "title": "Transformer Components — Combined",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    nb = build_notebook()
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    n_md = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    print(f"Wrote {OUT}")
    print(f"  Total cells: {len(nb['cells'])}  (markdown: {n_md}, code: {n_code})")
    size_kb = OUT.stat().st_size / 1024
    print(f"  Size: {size_kb:.1f} KB")
