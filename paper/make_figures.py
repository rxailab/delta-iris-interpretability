#!/usr/bin/env python
"""Generate publication figures for the ExpWM (Delta-IRIS interpretability) paper.

Run with:
    /mmfs1/storage/users/xiar3/exp/ExpWM/envs/delta-iris-env/bin/python make_figures.py

Outputs PDFs (vector) into paper/figures/, plus copies heads.png.
Style: white background, colorblind-friendly (Okabe-Ito) palette, >= 8pt fonts,
no in-figure titles (captions live in LaTeX), tight_layout.
"""

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------- paths
ANALYSIS = Path("/mmfs1/scratch/hpc/11/xiar3/expwm-runs/analysis-21531393")
RESULTS = Path("/mmfs1/storage/users/xiar3/exp/ExpWM/results")
FIGDIR = Path("/mmfs1/storage/users/xiar3/exp/ExpWM/paper/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- style
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,  # embed TrueType fonts
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Okabe-Ito colorblind-friendly palette
OI = {
    "black": "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#999999",
}


def asym_err(mean, lo, hi):
    """ci_lo/ci_hi are absolute bounds -> asymmetric error bar half-lengths."""
    return np.clip(mean - lo, 0, None), np.clip(hi - mean, 0, None)


# ================================================================ figure 1
# Codebook usage: 4 slots, rank-frequency curves, log y.
def fig_codebook_usage():
    d = np.load(ANALYSIS / "codebook_stats.npz")
    counts = d["counts"]  # (4, 1024)
    colors = [OI["blue"], OI["vermillion"], OI["green"], OI["orange"]]

    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    for slot in range(counts.shape[0]):
        c = np.sort(counts[slot])[::-1].astype(float)
        active = int((counts[slot] > 0).sum())
        c[c <= 0] = np.nan  # log scale: drop unused codes rather than fake them
        ax.plot(
            np.arange(1, len(c) + 1),
            c,
            color=colors[slot],
            lw=1.2,
            label=f"slot {slot} ({active} active)",
        )
    ax.set_yscale("log")
    ax.set_xlabel("code rank (sorted by usage)")
    ax.set_ylabel("usage count")
    ax.set_xlim(0, counts.shape[1] + 5)
    ax.legend(frameon=False, ncol=4, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGDIR / "codebook_usage.pdf")
    plt.close(fig)


# ================================================================ figure 2
# Probe heatmap (centerpiece): reps x labels, AUROC/acc annotated.
REP_ORDER = [
    "raw_codes",
    "frame_emb",
    "wm_latents_emb",
    "wm_input",
    "wm_block_1",
    "wm_block_2",
    "wm_block_3",
]
REP_DISPLAY = {
    "raw_codes": "raw codes",
    "frame_emb": "frame emb",
    "wm_latents_emb": "latents emb",
    "wm_input": "block 0 out",
    "wm_block_1": "block 1 out",
    "wm_block_2": "block 2 out",
    "wm_block_3": "final (post-LN)",
}


def fig_probe_heatmap():
    records = json.loads((ANALYSIS / "probes_hud" / "probe_metrics.json").read_text())
    records = [r for r in records if not r.get("skipped")]
    table = {(r["rep"], r["label"]): r for r in records}
    labels_present = {r["label"] for r in records}

    just = sorted(l for l in labels_present if l.startswith("ach_just["))
    cum = sorted(l for l in labels_present if l.startswith("ach_cum["))
    rows = ["action_taken", "reward_now", "reward_in_next_5"] + just + cum

    def display_row(lab):
        if lab.startswith("ach_just["):
            return "just: " + lab[len("ach_just[") : -1]
        if lab.startswith("ach_cum["):
            return "cum: " + lab[len("ach_cum[") : -1]
        return lab.replace("_", " ")

    M = np.full((len(rows), len(REP_ORDER)), np.nan)
    for i, lab in enumerate(rows):
        for j, rep in enumerate(REP_ORDER):
            r = table.get((rep, lab))
            if r is None:
                continue
            M[i, j] = r["auroc"] if "auroc" in r else r["acc"]

    n_rows = len(rows)
    fig_h = 0.62 + 0.168 * n_rows  # scale height with row count
    fig, ax = plt.subplots(figsize=(7.0, fig_h))
    cmap = plt.get_cmap("viridis")
    vmin, vmax = 0.5, 1.0
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    # cell annotations, text color chosen for contrast against viridis
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                continue
            rgba = cmap((np.clip(v, vmin, vmax) - vmin) / (vmax - vmin))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black" if lum > 0.55 else "white",
            )

    ax.set_xticks(range(len(REP_ORDER)))
    # rotate column headers to avoid adjacent-label collisions
    # (e.g. "block 2 out" running into "final (post-LN)")
    ax.set_xticklabels(
        [REP_DISPLAY[r] for r in REP_ORDER],
        rotation=30,
        ha="left",
        rotation_mode="anchor",
    )
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([display_row(l) for l in rows])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # separators between label groups (specials / just / cum)
    for y in (2.5, 2.5 + len(just)):
        ax.axhline(y, color="white", lw=2.0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cbar.set_label("AUROC / accuracy")
    cbar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "probe_heatmap.pdf")
    plt.close(fig)


# ================================================================ figure 3
# One-step ablation: gap0 vs gap3, 3 bars per concept, shared y scale.
def fig_ablation_onestep():
    def load(path):
        recs = json.loads(path.read_text())
        out = {}
        for r in recs:
            out[(r["concept"], r["moment_type"], r["condition"])] = r
        return out

    g0 = load(RESULTS / "ablation_gap0" / "causal_effects.json")
    g3 = load(RESULTS / "ablation_gap3" / "causal_effects.json")

    concepts = sorted(
        {k[0] for k in g0},
        key=lambda c: g0[(c, "unlock", "feat")]["mean"],
        reverse=True,
    )

    bars = [
        (("unlock", "feat"), "ablate concept feature (unlock)", OI["vermillion"]),
        (("ordinary", "feat"), "ablate concept feature (ordinary)", OI["skyblue"]),
        (("unlock", "rand"), "ablate random direction (unlock)", OI["gray"]),
    ]
    width = 0.26
    x = np.arange(len(concepts))

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for ax, data, tag in zip(axes, (g0, g3), ("gap 0", "gap 3")):
        for b, ((mt, cond), lab, col) in enumerate(bars):
            means = np.array([data[(c, mt, cond)]["mean"] for c in concepts])
            lo = np.array([data[(c, mt, cond)]["ci_lo"] for c in concepts])
            hi = np.array([data[(c, mt, cond)]["ci_hi"] for c in concepts])
            err = asym_err(means, lo, hi)
            ax.bar(
                x + (b - 1) * width,
                means,
                width,
                color=col,
                yerr=err,
                error_kw=dict(lw=0.8, capsize=2, capthick=0.8),
                label=lab,
            )
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", " ") for c in concepts], rotation=40, ha="right")
        ax.text(
            0.97, 0.95, tag, transform=ax.transAxes, ha="right", va="top", fontsize=8
        )
    axes[0].set_ylabel(r"$\Delta$ log P(codes at T)")
    axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.09, 0.99))
    fig.tight_layout()
    fig.savefig(FIGDIR / "ablation_onestep.pdf")
    plt.close(fig)


# ================================================================ figure 4
# Imagination steering: p_first / p_within, 4 conditions per concept.
def fig_imagination():
    recs = json.loads((RESULTS / "imagination" / "imagine_effects.json").read_text())
    table = {(r["concept"], r["condition"]): r for r in recs}
    all_concepts = list(dict.fromkeys(r["concept"] for r in recs))

    kept, dropped = [], []
    for c in all_concepts:
        if table[(c, "baseline")]["p_within"] >= 0.4:
            kept.append(c)
        else:
            dropped.append((c, table[(c, "baseline")]["p_within"]))
    print(f"[imagination] kept {len(kept)} concepts, dropped (baseline p_within<0.4): {dropped}")

    conds = [
        ("baseline", OI["green"]),
        ("sae", OI["vermillion"]),
        ("cav", OI["purple"]),
        ("random", OI["gray"]),
    ]
    width = 0.2
    x = np.arange(len(kept))

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)
    for ax, metric in zip(axes, ("p_first", "p_within")):
        for k, (cond, col) in enumerate(conds):
            means = np.array([table[(c, cond)][metric] for c in kept])
            lo = np.array([table[(c, cond)][metric + "_lo"] for c in kept])
            hi = np.array([table[(c, cond)][metric + "_hi"] for c in kept])
            err = asym_err(means, lo, hi)
            ax.bar(
                x + (k - 1.5) * width,
                means,
                width,
                color=col,
                yerr=err,
                error_kw=dict(lw=0.7, capsize=1.5, capthick=0.7),
                label=cond,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", " ") for c in kept], rotation=60, ha="right")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("P(achievement on first step)")
    axes[1].set_ylabel("P(achievement within horizon)")
    handles, lab = axes[0].get_legend_handles_labels()
    fig.legend(handles, lab, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGDIR / "imagination.pdf")
    plt.close(fig)


# ================================================================ figure 5
# Causal trace: 4 concepts x 3 token groups, restoration heatmaps.
def fig_causal_trace():
    data = json.loads((RESULTS / "causal_trace" / "trace_results.json").read_text())
    concepts = list(data.keys())  # preserve file order
    groups = data[concepts[0]]["groups"]  # ['frame', 'act', 'latents']

    fig, axes = plt.subplots(
        len(concepts), len(groups), figsize=(7.0, 6.0), sharex=True, sharey=True
    )
    im = None
    for i, c in enumerate(concepts):
        grid = np.asarray(data[c]["grid"], dtype=float)  # (n_groups, n_blocks, steps)
        for j, g in enumerate(groups):
            ax = axes[i, j]
            im = ax.imshow(
                np.clip(grid[j], 0, None),
                cmap="viridis",
                vmin=0,
                vmax=1,
                aspect="auto",
                interpolation="nearest",
            )
            n_blocks, n_steps = grid[j].shape
            ax.set_yticks(range(n_blocks))
            ax.set_yticklabels([f"block {b}" for b in range(n_blocks)])
            ax.set_xticks([0, 10, 20])
            ax.tick_params(length=2)
            if i == 0:
                ax.set_title(g, fontsize=8, pad=4)  # column group tag
            if j == 0:
                ax.set_ylabel(c.replace("_", " "), fontsize=8)
            if i == len(concepts) - 1:
                ax.set_xlabel("restored step")

    fig.tight_layout(rect=(0, 0, 0.92, 1))
    cax = fig.add_axes((0.935, 0.12, 0.018, 0.76))
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("restoration")
    cbar.outline.set_visible(False)
    fig.savefig(FIGDIR / "causal_trace.pdf")
    plt.close(fig)


# ================================================================ figure 6
def copy_heads():
    shutil.copyfile(RESULTS / "diagrams" / "heads.png", FIGDIR / "heads.png")


if __name__ == "__main__":
    fig_codebook_usage()
    fig_probe_heatmap()
    fig_ablation_onestep()
    fig_imagination()
    fig_causal_trace()
    copy_heads()
    print("done ->", FIGDIR)
