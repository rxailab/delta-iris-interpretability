"""Supplementary figure: behavioural cascade test through the Crafter tech tree.

Closed-loop imagination on the frozen agent: burn in to a collect_wood unlock,
then the actor-critic acts for 12 imagined steps while a direction is projected
out of the residual stream (baseline / SAE wood-feature f187 / wood CAV / random).
(A) imagined return by condition; (B) per wood-tree achievement (ordered by tech-
tree depth), probability its detector code is sampled within the horizon.

The lesion erases the wood code (depth 0) but does NOT cascade to downstream
achievements, and imagined return is unchanged: the SAE effect is local to the
next-code distribution, not behaviourally load-bearing.

Reads results/cascade_compare.json (copied from the scratch run). Outputs
figures/cascade.pdf (+ .png).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = json.loads(Path("/mmfs1/storage/users/xiar3/exp/ExpWM/results/cascade_compare.json").read_text())
OUT = Path("/mmfs1/storage/users/xiar3/exp/ExpWM/paper/figures")
conds = [c for c in ["baseline", "sae", "cav", "random"] if c in S["return"]]
cc = {"baseline": "#2ca02c", "sae": "#1B4F8A", "cav": "#B23A8E", "random": "#9aa0a8"}
lbl = {"baseline": "baseline", "sae": "SAE (f187)", "cav": "CAV", "random": "random"}

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 2.1]})

# (A) imagined return
xs = np.arange(len(conds))
ax[0].bar(xs, [S["return"][c]["mean"] for c in conds],
          yerr=[[S["return"][c]["mean"] - S["return"][c]["lo"] for c in conds],
                [S["return"][c]["hi"] - S["return"][c]["mean"] for c in conds]],
          color=[cc[c] for c in conds], capsize=3, edgecolor="black", lw=0.5)
ax[0].set_xticks(xs); ax[0].set_xticklabels([lbl[c] for c in conds], rotation=12)
ax[0].set_ylabel("imagined return (sum of $\\hat r$)")
ax[0].set_title("(A) Imagined return", fontsize=11, fontweight="bold")
ax[0].grid(axis="y", alpha=0.25)

# (B) cascade per wood-tree achievement
rows = sorted([r for r in S["per_ach"] if r["on_wood_tree"]],
              key=lambda r: (r["depth"], r["achievement"]))
y = np.arange(len(rows)); h = 0.26
for j, cond in enumerate([c for c in ["baseline", "sae", "cav"] if c in conds]):
    ax[1].barh(y + (1 - j) * h, [r[cond]["p"] for r in rows], height=h,
               color=cc[cond], edgecolor="black", lw=0.4, label=lbl[cond])
ax[1].set_yticks(y)
ax[1].set_yticklabels([f"d{r['depth']}  {r['achievement'].replace('_',' ')}" for r in rows], fontsize=8.2)
ax[1].invert_yaxis()
ax[1].set_xlabel("P(detector code sampled within 12-step imagination)")
ax[1].set_title("(B) No cascade through the wood tech tree", fontsize=11, fontweight="bold")
ax[1].legend(fontsize=8.5, loc="lower right"); ax[1].grid(axis="x", alpha=0.25); ax[1].set_xlim(0, 1.02)
ax[1].axhline(0.5, color="none")
ax[1].text(0.34, 5.5, "only the wood code (d0) is suppressed;\ndownstream achievements unchanged",
           fontsize=7.6, style="italic", color="#555", ha="left", va="center")
for s in ("top", "right"):
    ax[0].spines[s].set_visible(False); ax[1].spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "cascade.pdf", facecolor="white", bbox_inches="tight")
fig.savefig(OUT / "cascade.png", dpi=200, facecolor="white", bbox_inches="tight")
print("wrote", OUT / "cascade.pdf")
print(f"return baseline {S['return']['baseline']['mean']:.2f} vs sae {S['return']['sae']['mean']:.2f}")
