"""Figure 1 for the paper: the Δ-IRIS world model (architecture credited to
Micheli et al. 2024, redrawn) with OUR audit instrumentation overlaid.

Overlay legend:
  blue dots      = representation read-out points (linear probes / CAVs; SAE training data)
  green box      = TopK sparse autoencoder (trained on block-1 output)
  red scissors   = projection-ablation intervention point
  purple dashes  = activation-patching cell grid
  orange         = imagination-loop outcome (detector codes among sampled codes)

Output: figures/audit_overview.pdf (vector) + .png preview.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.lines import Line2D

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

# ----- palette ----------------------------------------------------------
C_FRAME, C_ACTION, C_LATENT = "#f4a261", "#e76f51", "#2a9d8f"
C_ARCH = "#46627a"          # architecture grey-blue (the "their work" colour)
C_READ = "#1f77b4"          # blue  : probe/CAV read taps (ours)
C_SAE  = "#2ca02c"          # green : SAE (ours)
C_ABL  = "#d62728"          # red   : ablation (ours)
C_PATCH= "#9467bd"          # purple: patching (ours)
C_OUT  = "#e07b00"          # orange: imagination outcome (ours)

fig, ax = plt.subplots(figsize=(13.5, 10.5))
ax.set_xlim(0, 13.5); ax.set_ylim(0, 10.5); ax.axis("off")

def box(x, y, w, h, fc, text, sub=None, fs=9.5, tc="white", lw=1.0, ec="black", alpha=0.95):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                                facecolor=fc, edgecolor=ec, lw=lw, alpha=alpha))
    cy = y + h/2 + (0.10 if sub else 0)
    ax.text(x + w/2, cy, text, ha="center", va="center", fontsize=fs, color=tc, weight="bold")
    if sub:
        ax.text(x + w/2, y + h/2 - 0.16, sub, ha="center", va="center", fontsize=7.5, color=tc)

def arrow(p, q, color="black", lw=1.2, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11,
                                 color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))

def read_tap(x, y, label, dx=0.28, dy=0.0, ha="left", fs=8):
    ax.add_patch(Circle((x, y), 0.09, facecolor=C_READ, edgecolor="black", lw=0.8, zorder=12))
    ax.text(x + dx, y + dy, label, fontsize=fs, color=C_READ, ha=ha, va="center", weight="bold")

# ====================== input token row (y ~ 9.0) =======================
pos_x = [1.4, 3.2, 5.0, 6.5, 8.0, 9.5]
labels = [("frame$_t$", "frame_cnn($x_t$)", C_FRAME),
          ("action$_t$", "act_emb($a_t$)", C_ACTION),
          ("lat$_0$", None, C_LATENT), ("lat$_1$", None, C_LATENT),
          ("lat$_2$", None, C_LATENT), ("lat$_3$", None, C_LATENT)]
y_in, bh = 8.95, 0.62
for (x, (name, sub, col)) in zip(pos_x, labels):
    box(x - 0.62, y_in, 1.24, bh, col, name, sub, fs=9)
ax.text(0.30, y_in + bh + 0.28, "one timestep block", fontsize=8.5, color="#555", style="italic")
ax.text(0.30, y_in - 0.30, r"sequence $=$ 21 timestep" "\nblocks (126 tokens)", fontsize=8,
        color="#555", style="italic", va="top")

# tokenizer codes feeding the latent embeddings (and the raw_codes tap)
box(6.0, 10.0, 3.0, 0.44, "white", "", lw=0.9, ec="#888")
ax.text(7.5, 10.22, "tokenizer codes  $c^0_t..c^3_t$  (codebook, 1024)", ha="center",
        va="center", fontsize=8.5, color="#333")
for x in pos_x[2:]:
    arrow((7.5, 10.0), (x, y_in + bh + 0.02), color="#888", lw=0.8, rad=0.0)
read_tap(9.05, 10.22, "raw_codes tap (codebook vectors)", dx=0.22, fs=8)

# frame_emb tap — the HUD-shortcut control
read_tap(pos_x[0] - 0.62, y_in + bh/2, "frame_emb tap\n(input-encoder control)",
         dx=-0.16, ha="right", fs=8)
# wm_latents_emb tap (label below the tap to avoid the loop arrow)
ax.add_patch(Circle((pos_x[5] + 0.62, y_in + bh/2), 0.09, facecolor=C_READ,
                    edgecolor="black", lw=0.8, zorder=12))
ax.text(pos_x[5] + 0.80, y_in - 0.32, "wm_latents_emb tap (mean of 4)",
        fontsize=8, color=C_READ, ha="left", va="center", weight="bold")

# ====================== transformer blocks ==============================
bx, bw = 1.0, 9.4
ys = {0: 7.55, 1: 6.35, 2: 5.15}
for i, yb in ys.items():
    box(bx, yb, bw, 0.72, C_ARCH, f"Transformer block {i}",
        "causal self-attention (8 heads, $d{=}512$) + MLP", fs=10)
# arrows input->b0->b1->b2->LN
for x in pos_x:
    arrow((x, y_in), (x, ys[0] + 0.74), color="#444", lw=0.9)
for i in [0, 1]:
    arrow((bx + bw/2, ys[i]), (bx + bw/2, ys[i+1] + 0.74), color="#444", lw=1.4)
box(bx, 4.42, bw, 0.36, "#6b8299", "final LayerNorm", fs=8.5)
arrow((bx + bw/2, ys[2]), (bx + bw/2, 4.80), color="#444", lw=1.4)

# block-output read taps (right edge)
for i, yb in ys.items():
    read_tap(bx + bw + 0.05, yb + 0.36, f"block-{i} output tap", dx=0.18, fs=8)
read_tap(bx + bw + 0.05, 4.60, "final tap", dx=0.18, fs=8)
ax.text(bx + bw + 0.20, 7.42, "blue taps: linear probes + CAVs\n(47 targets, episode-disjoint AUROC)",
        fontsize=7.5, color=C_READ, ha="left", va="center")

# ====================== SAE (green) =====================================
box(10.85, 5.55, 2.15, 0.80, "white", "", ec=C_SAE, lw=2.0)
ax.text(11.92, 6.13, "TopK SAE", ha="center", fontsize=9.5, color=C_SAE, weight="bold")
ax.text(11.92, 5.85, "2048 feats, $k{=}16$\nFVU 0.043", ha="center", va="center",
        fontsize=7.5, color=C_SAE)
arrow((bx + bw + 0.14, ys[1] + 0.36), (10.85, 5.95), color=C_SAE, lw=1.6, rad=-0.1)

# ====================== ablation (red) ==================================
ax.text(bx - 0.32, ys[1] + 0.36, "✂", fontsize=17, color=C_ABL, ha="center",
        va="center", weight="bold", zorder=13)
ax.text(bx - 0.42, ys[1] + 0.36,
        "projection ablation\n" r"$h \leftarrow h-(h^{\top}\hat d)\,\hat d$"
        "\none-step + sustained\nduring imagination",
        fontsize=8, color=C_ABL, ha="right", va="center")
ax.add_patch(Circle((bx + 0.0, ys[1] + 0.36), 0.10, facecolor=C_ABL,
                    edgecolor="black", lw=0.8, zorder=12))

# ====================== patching grid (purple) ==========================
ax.add_patch(Rectangle((bx - 0.12, 5.02), bw + 0.24, 3.40, fill=False,
                       edgecolor=C_PATCH, lw=1.6, linestyle=(0, (5, 4)), zorder=11))
ax.text(bx + bw + 0.20, 8.28, "activation patching cells\n(block $\\times$ timestep $\\times$ token-type)",
        fontsize=7.5, color=C_PATCH, ha="left", va="top")

# ====================== heads + outputs =================================
hx = {"rew": 1.7, "end": 4.3, "lat": 7.6}
box(hx["rew"] - 0.95, 3.30, 1.9, 0.55, "#8d5a97", "head_rewards", "reads action pos.", fs=8.5)
box(hx["end"] - 0.95, 3.30, 1.9, 0.55, "#8d5a97", "head_ends", "reads action pos.", fs=8.5)
box(hx["lat"] - 1.30, 3.30, 2.6, 0.55, "#5e548e", "head_latents", "reads pos. 1–4, autoregressive", fs=8.5)
for k in hx:
    arrow((hx[k], 4.42), (hx[k], 3.87), color="#444", lw=1.1)
box(0.95, 2.45, 1.5, 0.45, "#cfe3f5", r"$\hat r_t$", fs=9, tc="#222")
box(3.55, 2.45, 1.5, 0.45, "#cfe3f5", r"$\hat d_t$", fs=9, tc="#222")
box(6.30, 2.45, 2.6, 0.45, "#cfe3f5", r"sampled codes  $\hat c^0..\hat c^3$", fs=8.5, tc="#222")
arrow((hx["rew"], 3.30), (hx["rew"], 2.92), color="#444", lw=1.0)
arrow((hx["end"], 3.30), (hx["end"], 2.92), color="#444", lw=1.0)
arrow((hx["lat"], 3.30), (hx["lat"], 2.92), color="#444", lw=1.0)

# outcome highlight on sampled codes
ax.add_patch(Rectangle((6.24, 2.39), 2.72, 0.57, fill=False, edgecolor=C_OUT, lw=2.2, zorder=12))
ax.text(9.1, 2.67, "outcome: detector codes\namong sampled $\\hat c$?", fontsize=8.5,
        color=C_OUT, ha="left", va="center", weight="bold")

# ====================== imagination loop ================================
box(4.55, 1.30, 3.4, 0.55, "white", "tokenizer decoder", r"$\hat x_{t+1}$ from $(x_t, a_t, \hat c)$",
    fs=8.5, tc="#333", ec="#888")
box(8.75, 1.30, 2.7, 0.55, "white", "actor–critic", r"$a_{t+1}$ from $\hat x_{t+1}$", fs=8.5, tc="#333", ec="#888")
arrow((7.6, 2.45), (6.8, 1.87), color="#666", lw=1.2, rad=0.15)
arrow((7.95, 1.57), (8.75, 1.57), color="#666", lw=1.2)
# loop back to inputs (right side)
arrow((11.45, 1.57), (13.22, 1.57), color="#666", lw=1.2)
ax.add_patch(FancyArrowPatch((13.22, 1.57), (13.22, 9.55), arrowstyle="-", color="#666", lw=1.2))
ax.add_patch(FancyArrowPatch((13.22, 9.55), (10.26, 9.50), arrowstyle="-|>",
                             mutation_scale=11, color="#666", lw=1.2,
                             connectionstyle="arc3,rad=0.10"))
ax.text(13.40, 4.2, "imagination loop (12 steps)", fontsize=8.5, color="#555",
        rotation=90, va="center")

# ====================== legend ==========================================
legend_items = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor=C_READ, markeredgecolor="black",
           markersize=8, label="representation read-out (probes / CAVs; SAE training)"),
    Line2D([0], [0], color=C_SAE, lw=2.2, label="TopK sparse autoencoder (dictionary features)"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=C_ABL, markeredgecolor="black",
           markersize=8, label="projection-ablation intervention (block-1 output)"),
    Line2D([0], [0], color=C_PATCH, lw=2.0, linestyle=(0, (5, 4)), label="activation-patching cells"),
    Line2D([0], [0], color=C_OUT, lw=2.2, label="imagination outcome (detector-code occurrence)"),
    Line2D([0], [0], color=C_ARCH, lw=6, label="$\\Delta$-IRIS architecture (Micheli et al., redrawn)"),
]
ax.legend(handles=legend_items, loc="lower left", bbox_to_anchor=(0.005, 0.005),
          fontsize=8.2, frameon=True, framealpha=0.95)

plt.tight_layout()
fig.savefig(OUT / "audit_overview.pdf", bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "audit_overview.png", dpi=130, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote", OUT / "audit_overview.pdf")
