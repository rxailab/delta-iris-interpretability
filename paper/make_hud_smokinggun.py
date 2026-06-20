"""Smoking gun for the 2->3 (write->drive) gap: under the wood-feature lesion the
wood CODE is suppressed, yet the decoded frame still renders wood in the HUD
inventory counter -- so the state is carried by the frame, not the code, which is
why downstream tools (and behaviour) are unaffected.

Uses the already-saved paired decoded frames (baseline vs f187-ablated) from the
filmstrip pipeline; no new model run. The HUD region is the bottom rows (y>=49,
the inventory/vitals strip the no-HUD wrapper zeroes); the world region is y<49.

Outputs figures/hud_smoking_gun.pdf (+ .png) and results/hud_smoking_gun.json.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

NPZ = Path("/mmfs1/scratch/hpc/11/xiar3/expwm-runs/analysis-21531393/filmstrips/filmstrip_collect_wood.npz")
OUT = Path("/mmfs1/storage/users/xiar3/exp/ExpWM/paper/figures")
RES = Path("/mmfs1/storage/users/xiar3/exp/ExpWM/results/hud_smoking_gun.json")
HUD_Y = 49  # rows >= HUD_Y are the inventory/vitals HUD strip

d = np.load(NPZ)
fb = d["frames_baseline"].astype(np.float32)   # (M,T,64,64,3)
fa = d["frames_ablated"].astype(np.float32)
burn = d["burnin_last"].astype(np.float32)      # (M,64,64,3)
hb, ha = d["hit_baseline"], d["hit_ablated"]    # (M,T) wood detector-code sampled
M, T = fb.shape[:2]

def mae(a, b):  # mean abs pixel diff on 0-255 scale
    return float(np.mean(np.abs(a - b)))

# wood-code suppression (instrument power at the upstream node)
p_base = float(hb.mean()); p_abl = float(ha.mean())

# region-wise baseline-vs-ablated difference, averaged over all moments/steps
hud_ba = mae(fb[:, :, HUD_Y:], fa[:, :, HUD_Y:])
world_ba = mae(fb[:, :, :HUD_Y], fa[:, :, :HUD_Y])
# persistence: ablated HUD vs the burn-in HUD (which already shows wood) — does it stay?
burn_hud = burn[:, None, HUD_Y:]
hud_abl_vs_burn = mae(fa[:, :, HUD_Y:], np.repeat(burn_hud, T, axis=1))
world_abl_vs_burn = mae(fa[:, :, :HUD_Y], np.repeat(burn[:, None, :HUD_Y], T, axis=1))

digest = {
    "wood_code_p_baseline": p_base, "wood_code_p_ablated": p_abl,
    "hud_mae_baseline_vs_ablated": hud_ba, "world_mae_baseline_vs_ablated": world_ba,
    "hud_mae_ablated_vs_burnin": hud_abl_vs_burn, "world_mae_ablated_vs_burnin": world_abl_vs_burn,
    "ratio_world_over_hud_baseline_vs_ablated": world_ba / max(hud_ba, 1e-6),
    "n_moments": M, "horizon": T, "hud_rows": f">={HUD_Y}",
}
RES.parent.mkdir(exist_ok=True); RES.write_text(json.dumps(digest, indent=2))
print(json.dumps(digest, indent=2))

# ---- figure: a paired filmstrip for one illustrative moment, HUD boxed ----
# pick the moment where the wood code fires most under baseline and least under ablation
score = hb.sum(1) - ha.sum(1)
m = int(np.argmax(score))
steps = np.linspace(0, T - 1, 8).round().astype(int)
fig, axes = plt.subplots(2, len(steps), figsize=(1.15 * len(steps), 3.0))
for col, t in enumerate(steps):
    for row, (frames, hits, tag) in enumerate([(fb, hb, "baseline"), (fa, ha, "f187 ablated")]):
        ax = axes[row, col]
        ax.imshow(frames[m, t].astype(np.uint8)); ax.set_xticks([]); ax.set_yticks([])
        # box the HUD region
        ax.add_patch(Rectangle((0, HUD_Y), 63, 63 - HUD_Y, fill=False, edgecolor="#E8A33D", lw=1.4))
        if hits[m, t]:
            for s in ax.spines.values(): s.set_edgecolor("#1B4F8A"); s.set_linewidth(2.2)
        if row == 0: ax.set_title(f"t={t}", fontsize=7)
        if col == 0: ax.set_ylabel(tag, fontsize=8)
fig.suptitle("Wood-feature (f187) lesion: the wood code is suppressed, but the HUD inventory "
             "(orange box) still renders wood\n(blue frame = wood detector code sampled). "
             "State is frame-carried, not code-carried.", fontsize=8.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT / "hud_smoking_gun.pdf", facecolor="white", bbox_inches="tight")
fig.savefig(OUT / "hud_smoking_gun.png", dpi=200, facecolor="white", bbox_inches="tight")
print("wrote", OUT / "hud_smoking_gun.pdf")
