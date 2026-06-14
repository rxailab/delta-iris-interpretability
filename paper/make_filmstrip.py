#!/usr/bin/env python
"""
Filmstrip publication figures for the SAE-ablation imagination study.

(A) figures/filmstrip_headline.pdf/.png
    collect_wood, chosen moment. Three rows (baseline / SAE-ablated / random-ablated)
    of decoded imagined frames across the horizon, plus a leading "real" start column
    (burnin_last). Frames where the row's detector fired are outlined + tagged.

(B) figures/filmstrip_grid.pdf/.png
    Compact 4-concept summary, baseline vs SAE-ablated, 2 rows x ~6 step columns each,
    annotated with detector-hit counts.

Run with the project python; outputs vector PDF + PNG.
"""
import os
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

# ----------------------------------------------------------------------------
DATA_DIR = "/mmfs1/scratch/hpc/11/xiar3/expwm-runs/analysis-21531393/filmstrips"
OUT_DIR = "/mmfs1/storage/users/xiar3/exp/ExpWM/paper/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Fig.5 palette
COL_BASE = "#0B7D66"   # teal
COL_SAE = "#1B4F8A"    # blue
COL_OFF = "#B23A8E"    # magenta (off-target SAE feature, live control)
COL_RAND = "#7C828B"   # grey
COL_REAL = "#9aa0a6"   # faint box for the real start frame
INK = "#1b1b1b"

# Chosen moment indices (EXACT)
CHOSEN = {
    "collect_wood": 2,
    "collect_coal": 3,
    "collect_iron": 0,
    "place_stone": 1,
}
HEADLINE = "collect_wood"

# Short corner tags per concept
TAG = {
    "collect_wood": "wood",
    "collect_coal": "coal",
    "collect_iron": "iron",
    "place_stone": "stone",
}

UPSCALE = 4  # 64 -> 256 px, nearest neighbour (>= 120 px requirement)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def load(concept):
    return np.load(os.path.join(DATA_DIR, f"filmstrip_{concept}.npz"), allow_pickle=True)


def upscale(frame):
    """64x64x3 uint8 -> nearest-neighbour upscaled uint8."""
    img = Image.fromarray(frame.astype(np.uint8), "RGB")
    img = img.resize((64 * UPSCALE, 64 * UPSCALE), Image.NEAREST)
    return np.asarray(img)


def evenly_spaced(T, n):
    """Indices into [0, T-1], always including endpoints, ~n evenly-spaced, unique."""
    if T <= n:
        return list(range(T))
    idx = np.linspace(0, T - 1, n)
    idx = sorted(set(int(round(x)) for x in idx))
    # ensure endpoints present
    if idx[0] != 0:
        idx[0] = 0
    if idx[-1] != T - 1:
        idx[-1] = T - 1
    return sorted(set(idx))


def draw_frame(ax, img, color=None, lw=0.0, tag=None, tag_color=None):
    """Render an upscaled RGB frame in an axes; optional outline + corner tag."""
    ax.imshow(img, interpolation="nearest", aspect="equal",
              extent=(0, 1, 0, 1), zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        if color is not None and lw > 0:
            s.set_visible(True)
            s.set_edgecolor(color)
            s.set_linewidth(lw)
        else:
            s.set_visible(False)
    if tag is not None:
        # filled chip in upper-left corner
        ax.text(0.06, 0.92, tag, transform=ax.transAxes,
                ha="left", va="top", fontsize=5.4, color="white",
                fontweight="bold", zorder=5,
                bbox=dict(boxstyle="round,pad=0.18", fc=tag_color,
                          ec="white", lw=0.5, alpha=0.95))


# ============================================================================
# FIGURE A — headline filmstrip (collect_wood)
# ============================================================================
def make_headline():
    concept = HEADLINE
    mom = CHOSEN[concept]
    z = load(concept)
    meta = json.loads(str(z["meta"]))
    T = z["frames_baseline"].shape[1]

    steps = evenly_spaced(T, 10)
    ncols = len(steps) + 1  # +1 for the real start column

    off_short = meta.get("offtarget_concept", "other").replace("collect_", "").replace("_", " ")
    rows = [
        ("baseline\n(no intervention)", "frames_baseline", "hit_baseline", COL_BASE),
        ("matched SAE\nfeature ablated", "frames_ablated", "hit_ablated", COL_SAE),
        (f"off-target SAE\nfeature ({off_short})", "frames_offtarget", "hit_offtarget", COL_OFF),
        ("random\ndirection ablated", "frames_random", "hit_random", COL_RAND),
    ]

    # geometry
    cell = 0.62           # in inches per frame cell (square)
    gap = 0.045           # gap between frames
    left_label = 1.30     # inches reserved for row labels
    start_extra = 0.16    # extra gap after the real start column
    top_pad = 0.42
    bot_pad = 0.10

    fig_w = 7.0
    # compute cell size so it fits target width
    avail = fig_w - left_label - 0.10
    cell = (avail - gap * (ncols - 1) - start_extra) / ncols
    fig_h = top_pad + bot_pad + len(rows) * cell + (len(rows) - 1) * gap

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)

    def cell_x(c):
        # column 0 = real start; columns 1.. = imagined steps
        x = left_label
        if c == 0:
            return x
        return x + cell + gap + start_extra + (c - 1) * (cell + gap)

    def cell_y(r):
        return bot_pad + (len(rows) - 1 - r) * (cell + gap)

    fxw = cell / fig_w
    fyh = cell / fig_h

    tag_str = TAG[concept]

    for r, (label, fkey, hkey, color) in enumerate(rows):
        frames = z[fkey][mom]   # (T,64,64,3)
        hits = z[hkey][mom]     # (T,)
        y = cell_y(r) / fig_h

        # row label
        fig.text((left_label - 0.14) / fig_w, (cell_y(r) + cell / 2) / fig_h,
                 label, ha="right", va="center", fontsize=8.0,
                 color=color, fontweight="bold")

        # start (real) column — only on the top row do we render the actual burn-in frame;
        # render on every row for visual continuity (same real frame).
        burn = z["burnin_last"][mom]
        ax = fig.add_axes([cell_x(0) / fig_w, y, fxw, fyh])
        draw_frame(ax, upscale(burn), color=COL_REAL, lw=1.1)

        # imagined step columns
        for ci, t in enumerate(steps):
            c = ci + 1
            ax = fig.add_axes([cell_x(c) / fig_w, y, fxw, fyh])
            fired = bool(hits[t])
            if fired:
                draw_frame(ax, upscale(frames[t]), color=color, lw=2.4,
                           tag=tag_str, tag_color=color)
            else:
                draw_frame(ax, upscale(frames[t]), color=None, lw=0.0)

    # column headers (above top row)
    top_y = (cell_y(0) + cell + 0.06) / fig_h
    # start column header
    fig.text((cell_x(0) + cell / 2) / fig_w, top_y, "real",
             ha="center", va="bottom", fontsize=7.0, color=COL_REAL,
             fontweight="bold", fontstyle="italic")
    for ci, t in enumerate(steps):
        c = ci + 1
        fig.text((cell_x(c) + cell / 2) / fig_w, top_y, f"t={t+1}",
                 ha="center", va="bottom", fontsize=7.0, color=INK)

    pdf = os.path.join(OUT_DIR, "filmstrip_headline.pdf")
    png = os.path.join(OUT_DIR, "filmstrip_headline.png")
    fig.savefig(pdf, format="pdf", facecolor="white", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png, format="png", dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return pdf, png, [t + 1 for t in steps]


# ============================================================================
# FIGURE B — compact 4-concept grid (baseline vs SAE ablated)
# ============================================================================
def make_grid():
    order = ["collect_wood", "collect_coal", "collect_iron", "place_stone"]
    n_steps = 6

    # geometry (inches)
    fig_w = 7.0
    cell = 0.40
    gap = 0.035
    left_label = 0.95           # row-label gutter inside each block
    block_gap_x = 0.34          # gap between left/right block columns
    block_gap_y = 0.34          # gap between top/bottom block rows
    title_h = 0.20              # space above each block for its title
    top_pad = 0.06
    bot_pad = 0.06
    outer_left = 0.04

    rows_in_block = 2           # baseline, ablated
    block_w = left_label + n_steps * cell + (n_steps - 1) * gap
    block_h = title_h + rows_in_block * cell + (rows_in_block - 1) * gap

    fig_h = top_pad + bot_pad + 2 * block_h + block_gap_y

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
    fxw = cell / fig_w
    fyh = cell / fig_h

    sub_rows = [
        ("baseline", "frames_baseline", "hit_baseline", COL_BASE),
        ("SAE ablated", "frames_ablated", "hit_ablated", COL_SAE),
    ]

    for bi, concept in enumerate(order):
        br, bc = divmod(bi, 2)   # block row, block col
        mom = CHOSEN[concept]
        z = load(concept)
        T = z["frames_baseline"].shape[1]
        steps = evenly_spaced(T, n_steps)
        tag_str = TAG[concept]

        block_x0 = outer_left + bc * (block_w + block_gap_x)
        # block_y0 = bottom of block
        block_y0 = bot_pad + (1 - br) * (block_h + block_gap_y)

        # detector hit counts
        n_base = int(z["hit_baseline"][mom].sum())
        n_abl = int(z["hit_ablated"][mom].sum())

        # block title
        fig.text((block_x0 + 0.02) / fig_w,
                 (block_y0 + block_h - 0.02) / fig_h,
                 f"{concept}", ha="left", va="top", fontsize=8.2,
                 color=INK, fontweight="bold")
        fig.text((block_x0 + block_w) / fig_w,
                 (block_y0 + block_h - 0.03) / fig_h,
                 f"hits  base {n_base} / ablated {n_abl}", ha="right", va="top",
                 fontsize=6.6, color="#55585c")

        for r, (label, fkey, hkey, color) in enumerate(sub_rows):
            frames = z[fkey][mom]
            hits = z[hkey][mom]
            # row y (within block; row 0 on top)
            ry = block_y0 + (rows_in_block - 1 - r) * (cell + gap)

            fig.text((block_x0 + left_label - 0.06) / fig_w,
                     (ry + cell / 2) / fig_h,
                     label, ha="right", va="center", fontsize=6.6,
                     color=color, fontweight="bold")

            for ci, t in enumerate(steps):
                x = block_x0 + left_label + ci * (cell + gap)
                ax = fig.add_axes([x / fig_w, ry / fig_h, fxw, fyh])
                fired = bool(hits[t])
                if fired:
                    draw_frame(ax, upscale(frames[t]), color=color, lw=1.8,
                               tag=tag_str, tag_color=color)
                else:
                    draw_frame(ax, upscale(frames[t]), color=None, lw=0.0)

    pdf = os.path.join(OUT_DIR, "filmstrip_grid.pdf")
    png = os.path.join(OUT_DIR, "filmstrip_grid.png")
    fig.savefig(pdf, format="pdf", facecolor="white", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png, format="png", dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return pdf, png


if __name__ == "__main__":
    hpdf, hpng, hsteps = make_headline()
    gpdf, gpng = make_grid()
    print("HEADLINE_STEPS", hsteps)
    for f in [hpdf, hpng, gpdf, gpng]:
        print("OUT", f, os.path.getsize(f))
