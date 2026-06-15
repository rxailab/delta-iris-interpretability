"""Honest single-step causal-ablation figure (paper Fig. 5 / label fig:onestep).

Replaces the design-derived dual-scale version. Addresses review points:
  * shared x-axis for gap 0 and gap 3 (no x9 magnified inset) -> the timing
    collapse is read at the same scale, no misleading magnification;
  * complete 2x2 of conditions: {matched, random} direction x {unlock, ordinary}
    moments (the random x ordinary cell was previously omitted);
  * sign made explicit: Delta = baseline - ablated log P of the REALISED codes,
    so positive = ablation lowers their likelihood;
  * 95% bootstrap CIs shown; a filled marker means the matched-unlock CI excludes
    zero (significant), an open marker means it does not;
  * concentration shown honestly: a shaded band marks the few dominant effects,
    and n per condition is annotated (two concepts have n<80);
  * grey controls drawn dark enough to see.

Reads results/ablation_gap0|gap3/causal_effects.json and the rollout base rates.
Outputs figures/ablation_onestep.pdf (+ .png).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RES = Path("/mmfs1/storage/users/xiar3/exp/ExpWM/results")
OUT = Path("/mmfs1/storage/users/xiar3/exp/ExpWM/paper/figures")
SCRATCH = Path("/mmfs1/scratch/hpc/11/xiar3/expwm-runs/analysis-21531393")

def load(p):
    m = {}
    for r in json.loads(Path(p).read_text()):
        m[(r["concept"], r["moment_type"], r["condition"])] = r
    return m

G0 = load(RES / "ablation_gap0" / "causal_effects.json")
G3 = load(RES / "ablation_gap3" / "causal_effects.json")
concepts = sorted({k[0] for k in G0}, key=lambda c: G0[(c, "unlock", "feat")]["mean"])  # asc -> top=largest in barh

# conditions: (moment, condition, label, colour, marker, filled)
CONDS = [
    ("unlock",   "feat", "matched dir. · unlock",   "#1B4F8A", "o"),
    ("ordinary", "feat", "matched dir. · ordinary", "#B23A8E", "s"),
    ("unlock",   "rand", "random dir. · unlock",    "#4d5560", "^"),
    ("ordinary", "rand", "random dir. · ordinary",  "#aeb4bd", "D"),
]
LANE = [0.27, 0.09, -0.09, -0.27]   # vertical offsets within a row

fig, axes = plt.subplots(1, 2, figsize=(7.8, 6.2), sharey=True,
                         gridspec_kw={"width_ratios": [1, 1], "wspace": 0.06})
xlo, xhi = -0.75, 5.5
y = np.arange(len(concepts))

# significance of the matched-unlock effect (CI excludes 0)
sig = {c: (G0[(c, "unlock", "feat")]["ci_lo"] > 0 or G0[(c, "unlock", "feat")]["ci_hi"] < 0)
       for c in concepts}
# the "dominant" band: matched-unlock mean >= 1.0 nat
dominant = [i for i, c in enumerate(concepts) if G0[(c, "unlock", "feat")]["mean"] >= 1.0]

for ax, (G, tag, sub) in zip(axes, [(G0, "gap 0", "ablate at step $T$"),
                                    (G3, "gap 3", "ablate 3 steps before $T$")]):
    # shaded band behind dominant rows
    if dominant:
        ax.axhspan(min(dominant) - 0.5, max(dominant) + 0.5, color="#fff4d6", zorder=0)
    ax.axvline(0, color="#444", lw=1.1, zorder=1)
    for ci, c in enumerate(concepts):
        for (mt, cond, _lab, col, mk), off in zip(CONDS, LANE):
            r = G.get((c, mt, cond))
            if r is None:
                continue
            yy = ci + off
            m, lo, hi = r["mean"], r["ci_lo"], r["ci_hi"]
            ax.plot([0, m], [yy, yy], color=col, lw=1.4, zorder=2, solid_capstyle="round")
            ax.plot([lo, hi], [yy, yy], color=col, lw=0.7, alpha=0.7, zorder=2)
            is_mu = (mt == "unlock" and cond == "feat")
            face = col if (not is_mu or sig[c]) else "white"
            ax.plot([m], [yy], marker=mk, ms=4.2, mfc=face, mec=col, mew=1.0, zorder=3)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(-0.6, len(concepts) - 0.4)
    ax.set_title(tag, fontsize=12, fontweight="bold", pad=16)
    ax.text(0.5, 1.004, sub, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.5, style="italic", color="#666")
    ax.grid(axis="x", alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

axes[0].set_yticks(y)
axes[0].set_yticklabels([c.replace("_", " ") for c in concepts], fontsize=8.5)
# bold the significant rows' labels
for tl, c in zip(axes[0].get_yticklabels(), concepts):
    if sig[c]:
        tl.set_fontweight("bold")

# n annotation at far right of the gap-3 panel
for ci, c in enumerate(concepts):
    n = G0[(c, "unlock", "feat")]["n"]
    axes[1].text(xhi * 0.995, ci, f"n={n}", ha="right", va="center", fontsize=6.6,
                 color="#888")

# legend placed in the empty interior of the gap-3 panel (everything there sits at ~0)
handles = [Line2D([0], [0], color=col, marker=mk, ms=5, lw=1.6, label=lab)
           for (mt, cond, lab, col, mk) in CONDS]
axes[1].legend(handles=handles, loc="center", fontsize=8.2, frameon=True,
               framealpha=0.95)

# single shared x-axis label (the per-panel labels collided/truncated)
fig.tight_layout(rect=[0, 0.055, 1, 1])
fig.text(0.5, 0.018,
         "$\\Delta\\log P$(realised codes at $T$) $=$ baseline $-$ ablated   "
         "(positive $\\Rightarrow$ ablation lowers the likelihood of the realised codes)",
         ha="center", va="bottom", fontsize=9.5)
fig.savefig(OUT / "ablation_onestep.pdf", facecolor="white", bbox_inches="tight")
fig.savefig(OUT / "ablation_onestep.png", dpi=200, facecolor="white", bbox_inches="tight")
print("wrote", OUT / "ablation_onestep.pdf")

# print summary for the caption
npos = sum(1 for c in concepts if sig[c] and G0[(c,"unlock","feat")]["mean"] > 0)
nneg = sum(1 for c in concepts if sig[c] and G0[(c,"unlock","feat")]["mean"] < 0)
print(f"significant positive: {npos}; significant negative: {nneg}; dominant(>=1nat): {len(dominant)}")
