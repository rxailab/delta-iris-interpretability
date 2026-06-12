"""Render an accurate architecture diagram for Δ-IRIS world-model heads.

Each head's reading position is determined by the block_mask the trainer sets:
  head_rewards.block_mask = [0,1,0,0,0,0]      → reads pos 1 (h_action)
  head_ends.block_mask    = [0,1,0,0,0,0]      → reads pos 1 (h_action)
  head_latents.block_mask = [0,1,1,1,1,0]      → reads pos 1..4 (autoregressive)

Output: heads.png and heads.svg under the chosen --out dir.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D


def draw(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_xlim(0, 15); ax.set_ylim(0, 10); ax.axis("off")

    # ---------- colours ----------
    C_FRAME  = "#f4a261"
    C_ACTION = "#e76f51"
    C_LATENT = "#2a9d8f"
    C_HRED   = "#d62828"      # head_rewards
    C_HEND   = "#f77f00"      # head_ends
    C_HLAT   = "#5e548e"      # head_latents
    C_TRANS  = "#264653"
    C_PRED   = "#8ecae6"
    C_BG     = "white"

    # ---------- title ----------
    ax.text(7.5, 9.55, "Δ-IRIS world-model prediction heads (one timestep block)",
            ha="center", fontsize=15, weight="bold")
    ax.text(7.5, 9.20, "block_mask determines which transformer output position each head reads from",
            ha="center", fontsize=10, color="#444")

    # ---------- input token boxes (top row) ----------
    pos_x = [1.7, 3.7, 5.7, 7.7, 9.7, 11.7]
    labels = [
        ("frame_t",      "frame_cnn(x_t)",       C_FRAME),
        ("action_t",     "act_emb(a_t)",         C_ACTION),
        ("lat_0_t",      "latents_emb(c0)",      C_LATENT),
        ("lat_1_t",      "latents_emb(c1)",      C_LATENT),
        ("lat_2_t",      "latents_emb(c2)",      C_LATENT),
        ("lat_3_t",      "latents_emb(c3)",      C_LATENT),
    ]

    y_in = 7.7
    for i, (x, (name, sub, col)) in enumerate(zip(pos_x, labels)):
        ax.add_patch(FancyBboxPatch((x - 0.9, y_in - 0.35), 1.8, 0.7,
                                    boxstyle="round,pad=0.03",
                                    facecolor=col, edgecolor="black", lw=1.0, alpha=0.92))
        ax.text(x, y_in + 0.08, name, ha="center", va="center",
                fontsize=10, weight="bold", color="white")
        ax.text(x, y_in - 0.18, sub, ha="center", va="center",
                fontsize=8, color="white")
        ax.text(x, y_in + 0.62, f"pos {i}", ha="center", fontsize=9, color="#555")

    # label
    ax.text(0.4, y_in, "input\nsequence", ha="center", fontsize=10, weight="bold")

    # ---------- transformer block ----------
    ax.add_patch(FancyBboxPatch((1.0, 5.2), 11.4, 1.5,
                                boxstyle="round,pad=0.05",
                                facecolor=C_TRANS, edgecolor="black", lw=1.5, alpha=0.92))
    ax.text(6.7, 6.20, "Transformer Encoder",
            ha="center", color="white", fontsize=13, weight="bold")
    ax.text(6.7, 5.78, "3 blocks · 8 heads · embed_dim=512 · causal self-attention over all earlier positions",
            ha="center", color="#bbcad0", fontsize=9.5)
    ax.text(6.7, 5.45, "(within this block AND across all preceding timestep blocks in the 21-step window)",
            ha="center", color="#bbcad0", fontsize=8.5, style="italic")

    # arrows from input boxes into transformer
    for x in pos_x:
        ax.add_patch(FancyArrowPatch((x, y_in - 0.4), (x, 6.74),
                                     arrowstyle="-|>", mutation_scale=12,
                                     color="black", lw=1.0))

    # ---------- output (post-transformer) boxes ----------
    y_out = 4.5
    out_labels = ["h_frame", "h_act", "h_lat₀", "h_lat₁", "h_lat₂", "h_lat₃"]
    for x, lab in zip(pos_x, out_labels):
        ax.add_patch(FancyBboxPatch((x - 0.85, y_out - 0.3), 1.7, 0.6,
                                    boxstyle="round,pad=0.03",
                                    facecolor="#f8edeb", edgecolor="black", lw=1.0))
        ax.text(x, y_out, lab, ha="center", va="center", fontsize=9, weight="bold")
        # arrow from transformer to output
        ax.add_patch(FancyArrowPatch((x, 5.2), (x, y_out + 0.32),
                                     arrowstyle="-|>", mutation_scale=12,
                                     color="black", lw=1.0))
    ax.text(0.4, y_out, "hidden\nstates", ha="center", fontsize=10, weight="bold")

    # ---------- heads + reads ----------
    # head_rewards: reads h_act (pos 1) → predicts reward_t
    # head_ends:    reads h_act (pos 1) → predicts end_t
    # head_latents: reads h_act, h_lat₀, h_lat₁, h_lat₂ (pos 1..4) → predicts lat₀..lat₃

    # show block_mask line
    def mask_str(mask):
        return "[ " + "  ".join(str(m) for m in mask) + " ]"

    # head_rewards
    hr_y = 3.4
    ax.add_patch(FancyBboxPatch((1.0, hr_y - 0.45), 4.6, 0.9,
                                boxstyle="round,pad=0.04",
                                facecolor=C_HRED, edgecolor="black", lw=1.2, alpha=0.93))
    ax.text(3.3, hr_y + 0.20, "head_rewards", ha="center", color="white",
            fontsize=11, weight="bold")
    ax.text(3.3, hr_y - 0.13, "MLP(512→512→41 buckets, two-hot)",
            ha="center", color="white", fontsize=8.5)
    ax.text(3.3, hr_y - 0.33, "block_mask = " + mask_str([0,1,0,0,0,0]),
            ha="center", color="#f4d4d2", fontsize=8.5, family="monospace")
    # arrow from h_act → head_rewards (read)
    ax.add_patch(FancyArrowPatch((pos_x[1], y_out - 0.3), (pos_x[1] - 0.1, hr_y + 0.45),
                                 arrowstyle="-|>", mutation_scale=14,
                                 color=C_HRED, lw=2.0,
                                 connectionstyle="arc3,rad=-0.15"))
    # prediction box (right side)
    ax.add_patch(FancyBboxPatch((13.0, hr_y - 0.3), 1.7, 0.6,
                                boxstyle="round,pad=0.03",
                                facecolor=C_PRED, edgecolor="black", lw=1.0))
    ax.text(13.85, hr_y, "reward_t", ha="center", va="center",
            fontsize=10, weight="bold")
    ax.add_patch(FancyArrowPatch((5.6, hr_y), (12.95, hr_y),
                                 arrowstyle="-|>", mutation_scale=14,
                                 color=C_HRED, lw=2.0))

    # head_ends
    he_y = 2.3
    ax.add_patch(FancyBboxPatch((1.0, he_y - 0.45), 4.6, 0.9,
                                boxstyle="round,pad=0.04",
                                facecolor=C_HEND, edgecolor="black", lw=1.2, alpha=0.93))
    ax.text(3.3, he_y + 0.20, "head_ends", ha="center", color="white",
            fontsize=11, weight="bold")
    ax.text(3.3, he_y - 0.13, "MLP(512→512→1, BCE)",
            ha="center", color="white", fontsize=8.5)
    ax.text(3.3, he_y - 0.33, "block_mask = " + mask_str([0,1,0,0,0,0]),
            ha="center", color="#fce0cc", fontsize=8.5, family="monospace")
    ax.add_patch(FancyArrowPatch((pos_x[1], y_out - 0.3), (pos_x[1] - 0.2, he_y + 0.45),
                                 arrowstyle="-|>", mutation_scale=14,
                                 color=C_HEND, lw=2.0,
                                 connectionstyle="arc3,rad=-0.32"))
    ax.add_patch(FancyBboxPatch((13.0, he_y - 0.3), 1.7, 0.6,
                                boxstyle="round,pad=0.03",
                                facecolor=C_PRED, edgecolor="black", lw=1.0))
    ax.text(13.85, he_y, "end_t", ha="center", va="center",
            fontsize=10, weight="bold")
    ax.add_patch(FancyArrowPatch((5.6, he_y), (12.95, he_y),
                                 arrowstyle="-|>", mutation_scale=14,
                                 color=C_HEND, lw=2.0))

    # head_latents — single box, reads from 4 positions, predicts 4 codes
    hl_y = 1.0
    ax.add_patch(FancyBboxPatch((1.0, hl_y - 0.55), 4.6, 1.1,
                                boxstyle="round,pad=0.04",
                                facecolor=C_HLAT, edgecolor="black", lw=1.2, alpha=0.93))
    ax.text(3.3, hl_y + 0.30, "head_latents", ha="center", color="white",
            fontsize=11, weight="bold")
    ax.text(3.3, hl_y + 0.02, "MLP(512→512→1024)  ·  shared across 4 positions",
            ha="center", color="white", fontsize=8.5)
    ax.text(3.3, hl_y - 0.20, "block_mask = " + mask_str([0,1,1,1,1,0]),
            ha="center", color="#d4cee6", fontsize=8.5, family="monospace")
    ax.text(3.3, hl_y - 0.40, "autoregressive within the timestep",
            ha="center", color="#d4cee6", fontsize=8, style="italic")

    # 4 read arrows: pos 1, 2, 3, 4
    rads = [-0.45, -0.32, 0.32, 0.45]
    for i, idx in enumerate([1, 2, 3, 4]):
        ax.add_patch(FancyArrowPatch((pos_x[idx], y_out - 0.3),
                                     (5.6 - 0.05*i, hl_y + 0.55 - 0.06*i),
                                     arrowstyle="-|>", mutation_scale=12,
                                     color=C_HLAT, lw=1.5, alpha=0.85,
                                     connectionstyle=f"arc3,rad={rads[i]}"))

    # 4 prediction boxes (one per slot of the NEXT transition)
    pred_xs = [7.6, 9.4, 11.2, 13.0]
    for idx, (px) in enumerate(pred_xs):
        ax.add_patch(FancyBboxPatch((px, hl_y - 0.3), 1.65, 0.6,
                                    boxstyle="round,pad=0.03",
                                    facecolor=C_PRED, edgecolor="black", lw=1.0))
        ax.text(px + 0.82, hl_y + 0.04, f"lat_{idx}_t", ha="center", va="center",
                fontsize=9.5, weight="bold")
        ax.text(px + 0.82, hl_y - 0.18, f"(P over 1024 codes)", ha="center",
                va="center", fontsize=7.5, color="#333")
    ax.add_patch(FancyArrowPatch((5.6, hl_y), (7.55, hl_y),
                                 arrowstyle="-|>", mutation_scale=14,
                                 color=C_HLAT, lw=2.0))

    # conditioning chain note for head_latents
    ax.text(7.5, 0.30,
            "autoregressive ordering:  lat_0_t ← (frame,action) · lat_1_t ← +lat_0 · "
            "lat_2_t ← +lat_1 · lat_3_t ← +lat_2",
            ha="center", fontsize=8.5, style="italic", color="#555")

    # legend
    legend_lines = [
        Line2D([0], [0], color=C_HRED, lw=2.5, label="head_rewards reads"),
        Line2D([0], [0], color=C_HEND, lw=2.5, label="head_ends reads"),
        Line2D([0], [0], color=C_HLAT, lw=2.5, label="head_latents reads (×4 positions)"),
    ]
    ax.legend(handles=legend_lines, loc="upper right",
              bbox_to_anchor=(0.98, 0.43), fontsize=9, frameon=True)

    fig.savefig(out / "heads.png", dpi=160, bbox_inches="tight", facecolor=C_BG)
    fig.savefig(out / "heads.svg", bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"wrote {out/'heads.png'}  ({(out/'heads.png').stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    draw(a.out)
