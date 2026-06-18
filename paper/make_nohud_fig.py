"""Supplementary figure: with-HUD vs no-HUD partial-observability control.

(A) per cumulative-state concept, input-encoder (frame_emb) held-out AUROC
    with the HUD vs with the HUD zeroed (no-HUD). World-persistent achievements
    stay near 1.0 (read off the world); only genuinely HUD-only concepts drop.
(B) causal-tracing restoration mass on history cells vs the final (unlock) step,
    HUD vs no-HUD: with state hidden, the model reaches ~5x further into history
    (de-Markovianisation), while the final-step mass is unchanged.

Reads the with-HUD analysis-21531393 and no-HUD analysis-nohud-22054506 outputs.
Outputs figures/nohud_compare.pdf (+ .png) and results/nohud_compare.json.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HUD = Path("/mmfs1/scratch/hpc/11/xiar3/expwm-runs/analysis-21531393")
NOH = Path("/mmfs1/scratch/hpc/11/xiar3/expwm-runs/analysis-nohud-22054506")
OUT = Path("/mmfs1/storage/users/xiar3/exp/ExpWM/paper/figures")
RES = Path("/mmfs1/storage/users/xiar3/exp/ExpWM/results/nohud_compare.json")

def probe_idx(p):
    return {(r["rep"], r["label"]): r for r in json.loads(Path(p).read_text())}

H = probe_idx(HUD / "probes_hud" / "probe_metrics.json")
N = probe_idx(NOH / "probes" / "probe_metrics.json")
blocks = ["wm_block_1", "wm_block_2", "wm_block_3"]
au = lambda D, rep, lab: (D.get((rep, lab)) or {}).get("auroc")
bb = lambda D, lab: max([v for v in (au(D, b, lab) for b in blocks) if v is not None] or [None])

cums = sorted({l for (_, l) in N if l.startswith("ach_cum[")})
rows = []
for l in cums:
    hfe, nfe, nbl = au(H, "frame_emb", l), au(N, "frame_emb", l), bb(N, l)
    if None in (hfe, nfe, nbl):
        continue
    rows.append((l[8:-1], hfe, nfe, nbl))

# causal-trace history vs final restoration
def trace_stats(p):
    t = json.loads(Path(p).read_text())
    hist, final = [], []
    for _, v in t.items():
        g = np.array(v["grid"], dtype=float)
        if g.ndim != 3:
            continue
        T = g.shape[-1]
        hist.append(np.nanmean(np.abs(g[..., :T - 1])))
        final.append(np.nanmean(g[..., T - 1]))
    return float(np.mean(hist)), float(np.mean(final))
h_hist, h_fin = trace_stats(HUD / "causal_trace" / "trace_results.json")
n_hist, n_fin = trace_stats(NOH / "causal_trace" / "trace_results.json")

fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.9))

# Panel A: frame_emb AUROC HUD vs no-HUD
axA = ax[0]
hx = [r[1] for r in rows]; ny = [r[2] for r in rows]
drop = [(r[1] - r[2]) > 0.10 for r in rows]
axA.plot([0.5, 1.01], [0.5, 1.01], color="#999", lw=1, ls="--", zorder=1)
axA.scatter([x for x, d in zip(hx, drop) if not d], [y for y, d in zip(ny, drop) if not d],
            s=34, c="#1B4F8A", edgecolor="black", lw=0.4, zorder=3, label="world-persistent")
axA.scatter([x for x, d in zip(hx, drop) if d], [y for y, d in zip(ny, drop) if d],
            s=42, c="#B23A8E", edgecolor="black", lw=0.4, zorder=3, label="HUD-dependent (drop $>$0.10)")
for name, hfe, nfe, _ in rows:
    if (hfe - nfe) > 0.10:
        axA.annotate(name, (hfe, nfe), fontsize=6.0, color="#7a2860",
                     xytext=(2, -4), textcoords="offset points")
axA.set_xlabel("$\\mathsf{frame\\_emb}$ AUROC, with HUD")
axA.set_ylabel("$\\mathsf{frame\\_emb}$ AUROC, no HUD")
axA.set_title("(A) Input-encoder cumulative-state decodability", fontsize=10, fontweight="bold")
axA.set_xlim(0.6, 1.01); axA.set_ylim(0.55, 1.02)
axA.legend(fontsize=7.5, loc="lower right", frameon=True)
axA.grid(alpha=0.25)
for s in ("top", "right"):
    axA.spines[s].set_visible(False)

# Panel B: history vs final restoration
axB = ax[1]
x = np.arange(2); w = 0.36
axB.bar(x - w / 2, [h_hist, h_fin], w, label="with HUD", color="#7C828B", edgecolor="black", lw=0.5)
axB.bar(x + w / 2, [n_hist, n_fin], w, label="no HUD", color="#E07B00", edgecolor="black", lw=0.5)
axB.set_xticks(x); axB.set_xticklabels(["history cells\n(steps $1$--$T{-}1$)", "final (unlock)\nstep $T$"])
axB.set_ylabel("mean causal-tracing restoration")
axB.set_title("(B) De-Markovianisation of next-code prediction", fontsize=10, fontweight="bold")
axB.legend(fontsize=8, frameon=True)
axB.grid(axis="y", alpha=0.25)
for s in ("top", "right"):
    axB.spines[s].set_visible(False)
axB.annotate(f"$\\times${n_hist / h_hist:.1f}", (0, n_hist), fontsize=9, ha="center",
             xytext=(6, 4), textcoords="offset points", color="#E07B00", fontweight="bold")

fig.tight_layout()
fig.savefig(OUT / "nohud_compare.pdf", facecolor="white", bbox_inches="tight")
fig.savefig(OUT / "nohud_compare.png", dpi=200, facecolor="white", bbox_inches="tight")

digest = {
    "behaviour_return": {"hud": 16.20, "nohud": 9.65},
    "frame_emb_cum_auroc_mean": {"hud": float(np.mean(hx)), "nohud": float(np.mean(ny))},
    "n_concepts": len(rows),
    "n_hud_dependent_drop_gt_0.10": int(sum(drop)),
    "hud_dependent_concepts": [r[0] for r in rows if (r[1] - r[2]) > 0.10],
    "block_beats_frame_emb_count": int(sum((r[3] - r[2]) > 0.02 for r in rows)),
    "trace_restoration": {"hud": {"history": h_hist, "final": h_fin},
                          "nohud": {"history": n_hist, "final": n_fin}},
    "per_concept_frame_emb_auroc": [{"concept": r[0], "hud": r[1], "nohud": r[2], "nohud_best_block": r[3]} for r in rows],
}
RES.parent.mkdir(exist_ok=True)
RES.write_text(json.dumps(digest, indent=2))
print("wrote", OUT / "nohud_compare.pdf")
print(f"frame_emb cum AUROC mean: HUD {np.mean(hx):.3f} -> noHUD {np.mean(ny):.3f}; "
      f"HUD-dependent {sum(drop)}/{len(rows)}; trace history x{n_hist/h_hist:.1f}")
