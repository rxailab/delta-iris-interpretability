"""Figure: multi-direction (INLP-style) subspace ablation (paper Fig. fig:subspace).

Two panels, gap-0 single-step, unlock moments:
  (A) mean Delta log P of the realised codes vs subspace rank r, for the matched
      SAE-feature stack, the matched CAV subspace, and a random subspace. The SAE
      curve climbs with rank (stacking more dictionary directions removes more of
      the causal signal); the CAV curve saturates ~0.5 nat and never approaches
      it; random is inert. This is the multi-direction test the single-direction
      read-write result deferred.
  (B) probe AUROC after projecting out the rank-r CAV subspace and refitting: the
      concept stays perfectly decodable (~1.0) even at rank 16 -- decodability is
      massively redundant, which is exactly why removing CAV directions does not
      bite causally.

Reads results/subspace_ablation_gap0/{subspace_effects,auroc_recovery}.json.
Outputs figures/subspace_ablation.pdf (+ .png).
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path("/mmfs1/scratch/hpc/11/xiar3/expwm-runs/analysis-21531393/subspace_ablation_gap0")
OUT = Path("/mmfs1/storage/users/xiar3/exp/ExpWM/paper/figures")
OUT.mkdir(exist_ok=True)

eff = json.loads((SRC / "subspace_effects.json").read_text())
rec = json.loads((SRC / "auroc_recovery.json").read_text())

BASES = [("sae", "matched SAE-feature stack", "#1B4F8A", "o"),
         ("cav", "matched CAV subspace", "#B23A8E", "s"),
         ("random", "random subspace", "#9aa0a8", "^")]

# Panel A: aggregate mean +/- SEM across concepts, unlock moments
by = defaultdict(list)
for r in eff:
    if r.get("moment_type") != "unlock":
        continue
    by[(r["basis"], r["rank"])].append(r["mean"])
ranks = sorted({k[1] for k in by})

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.7))
ax = axes[0]
for key, label, col, mk in BASES:
    ys = np.array([np.mean(by[(key, r)]) for r in ranks])
    sem = np.array([np.std(by[(key, r)]) / np.sqrt(len(by[(key, r)])) for r in ranks])
    ax.plot(ranks, ys, marker=mk, color=col, lw=1.8, ms=5, label=label)
    ax.fill_between(ranks, ys - sem, ys + sem, color=col, alpha=0.15, lw=0)
ax.axhline(0, color="#444", lw=0.8)
ax.set_xlabel("subspace rank $r$ (directions removed)")
ax.set_ylabel("$\\Delta\\log P$(realised codes)  (nats)")
ax.set_title("(A) Causal effect vs rank", fontsize=11, fontweight="bold")
ax.set_xticks(ranks)
ax.legend(fontsize=8.2, frameon=True, loc="upper left")
ax.grid(alpha=0.25)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# Panel B: AUROC recovery vs rank (CAV subspace projected out, probe refit)
byc = defaultdict(dict)
for r in rec:
    byc[r["concept"]][r["rank"]] = r["auroc"]
rr = sorted({k for d in byc.values() for k in d})
ax = axes[1]
mat = np.array([[byc[c].get(r, np.nan) for r in rr] for c in byc])
mean = np.nanmean(mat, axis=0)
lo, hi = np.nanmin(mat, axis=0), np.nanmax(mat, axis=0)
for c in byc:
    ax.plot(rr, [byc[c].get(r, np.nan) for r in rr], color="#B23A8E", alpha=0.18, lw=1.0)
ax.plot(rr, mean, color="#B23A8E", lw=2.2, marker="s", ms=5, label="mean over concepts")
ax.fill_between(rr, lo, hi, color="#B23A8E", alpha=0.10, lw=0, label="min--max")
ax.set_ylim(0.5, 1.02)
ax.set_xlabel("CAV-subspace rank $r$ removed")
ax.set_ylabel("re-fit probe AUROC")
ax.set_title("(B) Decodability survives removal", fontsize=11, fontweight="bold")
ax.set_xticks(rr)
ax.axhline(0.5, color="#888", lw=0.8, ls=":")
ax.legend(fontsize=8.2, frameon=True, loc="lower left")
ax.grid(alpha=0.25)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

fig.tight_layout()
fig.savefig(OUT / "subspace_ablation.pdf", facecolor="white", bbox_inches="tight")
fig.savefig(OUT / "subspace_ablation.png", dpi=200, facecolor="white", bbox_inches="tight")
print("wrote", OUT / "subspace_ablation.pdf")

# print the numbers the caption/text cite
sae16 = np.mean(by[("sae", 16)]); cav16 = np.mean(by[("cav", 16)])
sae1 = np.mean(by[("sae", 1)]); cav1 = np.mean(by[("cav", 1)])
print(f"SAE r1={sae1:.2f} r16={sae16:.2f} | CAV r1={cav1:.2f} r16={cav16:.2f} | ratio r16={sae16/max(cav16,1e-6):.1f}x")
print(f"AUROC mean r0={mean[0]:.3f} r16={mean[-1]:.3f}")
