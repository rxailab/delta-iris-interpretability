"""Comparison figure: how much the delta trick sharpens code monosemanticity.

Reads the three mono_<kind>.json files (Δ-IRIS delta tokenizer, frame-only
ablation, original IRIS), and plots, per achievement, the best detector code's
P(achievement | code) under each tokenizer (grouped horizontal bars, sorted by
the Δ-IRIS value). Δ-IRIS's delta codes are far stronger per-event detectors;
the frame-only ablation tracks literal IRIS, isolating the gap to the delta
trick rather than codebook/token-count/architecture differences.

Output: figures/tokenizer_compare.pdf (+ .png)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path("/mmfs1/scratch/hpc/11/xiar3/expwm-runs/analysis-21531393/mono_compare")
OUT = Path("/mmfs1/storage/users/xiar3/exp/ExpWM/paper/figures")
OUT.mkdir(exist_ok=True)

KINDS = [("deltairis", "$\\Delta$-IRIS (delta + action)", "#1B4F8A"),
         ("frameonly", "frame-only ablation",            "#E07B00"),
         ("iris",      "original IRIS",                   "#7C828B")]

D = {k: json.loads((SRC / f"mono_{k}.json").read_text()) for k, _, _ in KINDS}

def bestp(d):
    return {b["achievement"]: b["p_a_given_c"] for b in d["per_achievement"]}
bp = {k: bestp(D[k]) for k, _, _ in KINDS}
achs = sorted(bp["deltairis"], key=lambda a: bp["deltairis"][a])  # ascending -> top of barh is largest

fig, ax = plt.subplots(figsize=(7.0, 5.6))
y = np.arange(len(achs)); h = 0.26
for j, (k, label, col) in enumerate(KINDS):
    vals = [bp[k].get(a, 0.0) for a in achs]
    ax.barh(y + (1 - j) * h, vals, height=h, label=label, color=col,
            edgecolor="black", linewidth=0.4)
ax.set_yticks(y); ax.set_yticklabels([a.replace("_", " ") for a in achs], fontsize=8.5)
ax.set_xlabel("best detector $P(\\mathrm{achievement}\\mid\\mathrm{code})$", fontsize=10)
ax.set_xlim(0, 1.0)
ax.legend(loc="lower right", fontsize=9.5, frameon=True)
ax.grid(axis="x", alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig(OUT / "tokenizer_compare.pdf", facecolor="white", bbox_inches="tight")
fig.savefig(OUT / "tokenizer_compare.png", dpi=200, facecolor="white", bbox_inches="tight")
print("wrote", OUT / "tokenizer_compare.pdf")
