"""Render an HTML gallery of top-K transitions for each Δ-IRIS code.

Reads the artefacts produced by codebook_stats.py and the on-disk dataset, pulls
out the (prev_frame, frame) pairs for each kept reference, packs them into PNG
sprites, and emits an HTML page that lets you scroll through codes.

  python render_gallery.py \
    --run /mmfs1/scratch/.../delta-iris-full-21531393/hydra \
    --stats /mmfs1/scratch/.../analysis-21531393 \
    --top-show 8                # transitions per code in the HTML

Output (under --stats/gallery/):
  - sprite_slot{k}_code{c}.png  : 2*top × frame grid for one (slot, code)
  - index.html                  : main page with usage histogram + sortable table
  - codebook.json               : code metadata (count, mean_sim, etc.) for JS
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ACTIONS_CRAFTER = [
    "noop", "move_west", "move_east", "move_north", "move_south",
    "do", "sleep", "place_stone", "place_table", "place_furnace",
    "place_plant", "make_wood_pickaxe", "make_stone_pickaxe",
    "make_iron_pickaxe", "make_wood_sword", "make_stone_sword",
    "make_iron_sword",
]  # 17 in the order the Crafter env returns


def tile_pair(prev: np.ndarray, frame: np.ndarray, action: int) -> Image.Image:
    """Stack prev (left, 64x64) + frame (right, 64x64) into a 128x70 PIL image
    with a 6-pixel header for the action label."""
    H = 6
    canvas = Image.new("RGB", (128, 64 + H), color=(20, 22, 28))
    canvas.paste(Image.fromarray(prev.transpose(1, 2, 0), "RGB"), (0, H))
    canvas.paste(Image.fromarray(frame.transpose(1, 2, 0), "RGB"), (64, H))
    return canvas


def make_sprite(refs, dataset, action_names) -> tuple[Image.Image, list[dict]]:
    """Build one horizontal sprite for a (slot, code)'s top-K refs."""
    tiles = []
    meta = []
    for sim, ep_id, step in refs:
        if ep_id < 0:
            continue
        try:
            ep = dataset.load_episode(int(ep_id))
        except FileNotFoundError:
            continue
        obs = ep.observations.numpy()                       # (T, 3, 64, 64) uint8
        act = int(ep.actions[step].item())
        rew = float(ep.rewards[step + 1].item()) if step + 1 < ep.rewards.shape[0] else 0.0
        tiles.append(tile_pair(obs[step], obs[step + 1], act))
        meta.append(dict(sim=float(sim), ep=int(ep_id), step=int(step),
                         action=act, action_name=action_names[act] if act < len(action_names) else f"act{act}",
                         reward=rew))
    if not tiles:
        return None, []
    W, H = tiles[0].size
    sprite = Image.new("RGB", (W * len(tiles), H), color=(20, 22, 28))
    for i, t in enumerate(tiles):
        sprite.paste(t, (i * W, 0))
    return sprite, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--stats", required=True, type=Path)
    ap.add_argument("--top-show", type=int, default=8,
                    help="how many top-K samples to show per code (≤ stored top-K)")
    ap.add_argument("--min-count", type=int, default=10,
                    help="skip codes used fewer than this many times in total")
    args = ap.parse_args()

    dataset_dir = args.run / "checkpoints" / "dataset" / "train"
    sys.path.insert(0, str(args.run / "src"))
    from data.dataset import EpisodeDataset
    dataset = EpisodeDataset(dataset_dir, name="train")
    stats = np.load(args.stats / "codebook_stats.npz")
    top = np.load(args.stats / "top_samples.npz")
    meta_info = json.loads((args.stats / "meta.json").read_text())

    counts = stats["counts"]                   # (K_slots, C)
    K_slots, C = counts.shape
    out_dir = args.stats / "gallery"
    out_dir.mkdir(parents=True, exist_ok=True)
    sprites_dir = out_dir / "sprites"; sprites_dir.mkdir(exist_ok=True)

    code_records = []
    print(f"rendering up to {K_slots}×{C} = {K_slots*C} codes "
          f"(min-count={args.min_count}, top-show={args.top_show})")

    n_rendered = 0
    for k_slot in range(K_slots):
        for c in range(C):
            n = int(counts[k_slot, c])
            if n < args.min_count:
                continue
            refs = list(zip(
                top["sims"][k_slot, c, :args.top_show],
                top["eps"][k_slot, c, :args.top_show],
                top["steps"][k_slot, c, :args.top_show],
            ))
            sprite, samples = make_sprite(refs, dataset, ACTIONS_CRAFTER)
            if sprite is None:
                continue
            sprite_path = sprites_dir / f"s{k_slot}_c{c}.png"
            sprite.save(sprite_path, optimize=True)
            code_records.append(dict(
                slot=k_slot, code=c, count=n,
                mean_sim=float(np.nanmean(top["sims"][k_slot, c])),
                sprite=str(sprite_path.relative_to(out_dir)),
                samples=samples,
            ))
            n_rendered += 1
            if n_rendered % 100 == 0:
                print(f"  {n_rendered} codes rendered…", flush=True)

    print(f"rendered {n_rendered} codes")

    # Slot-wise usage histogram, embedded as PNG.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, K_slots, figsize=(4*K_slots, 3), sharey=True)
        if K_slots == 1: axes = [axes]
        for k_slot in range(K_slots):
            sorted_counts = np.sort(counts[k_slot])[::-1]
            axes[k_slot].fill_between(np.arange(C), sorted_counts, color="#58a6ff")
            axes[k_slot].set_yscale("log")
            axes[k_slot].set_title(f"slot {k_slot}  ({(counts[k_slot]>0).sum()}/{C} active)")
            axes[k_slot].set_xlabel("code rank")
        axes[0].set_ylabel("count")
        plt.tight_layout()
        buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=110); plt.close(fig)
        hist_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as e:
        hist_b64 = ""
        print(f"  matplotlib unavailable, skipping histogram: {e}")

    (out_dir / "codebook.json").write_text(json.dumps(code_records))

    html = HTML_TEMPLATE.replace("__META__", json.dumps(meta_info)) \
                       .replace("__HIST__", hist_b64) \
                       .replace("__SLOTS__", str(K_slots)) \
                       .replace("__C__", str(C))
    (out_dir / "index.html").write_text(html)
    print(f"\nopen: {out_dir / 'index.html'}")


HTML_TEMPLATE = """<!doctype html>
<html lang=en>
<head>
<meta charset=utf-8>
<title>Δ-IRIS codebook</title>
<style>
:root { --bg:#0d1117; --panel:#161b22; --border:#30363d; --text:#c9d1d9; --muted:#8b949e; --slot0:#d29922; --slot1:#3fb950; --slot2:#58a6ff; --slot3:#d2a8ff; }
body{margin:0;background:var(--bg);color:var(--text);font:14px ui-monospace,Menlo,Consolas,monospace;padding:18px;}
h1{margin:0;font-size:18px;color:#79c0ff;}
.meta{color:var(--muted);margin:4px 0 14px;font-size:12px;}
.controls{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px;}
.controls label{background:var(--panel);border:1px solid var(--border);padding:6px 10px;border-radius:6px;}
.controls input,.controls select{background:transparent;color:var(--text);border:none;outline:none;font:inherit;width:6em;}
.hist{background:var(--panel);border:1px solid var(--border);border-radius:6px;padding:10px;margin-bottom:14px;}
.hist img{max-width:100%;display:block;}
.code{display:flex;gap:14px;align-items:center;background:var(--panel);border:1px solid var(--border);border-radius:6px;padding:8px 10px;margin-bottom:8px;}
.code .lbl{flex:0 0 auto;min-width:140px;font-size:12px;}
.code .lbl b{font-size:15px;}
.code img{max-height:78px;image-rendering:pixelated;border-radius:4px;}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;vertical-align:middle;}
.s0{background:var(--slot0);} .s1{background:var(--slot1);} .s2{background:var(--slot2);} .s3{background:var(--slot3);}
.actions{color:var(--muted);font-size:11px;margin-top:4px;line-height:1.4;}
#empty{color:var(--muted);padding:12px;}
</style>
</head>
<body>
<h1>Δ-IRIS codebook explorer</h1>
<div class=meta id=meta>—</div>
<div class=hist><img alt="usage histogram" src="data:image/png;base64,__HIST__"></div>

<div class=controls>
  <label>slot <select id=slot><option value=-1>any</option><option>0</option><option>1</option><option>2</option><option>3</option></select></label>
  <label>min count <input id=mincount type=number value=200></label>
  <label>sort <select id=sort><option value=count>count (high→low)</option><option value=sim>mean cosine sim</option><option value=code>code id</option></select></label>
  <label>search code <input id=q placeholder="e.g. 472"></label>
  <label>limit <input id=limit type=number value=200></label>
  <button id=apply style="background:#1f6feb;color:white;border:0;border-radius:4px;padding:6px 12px;cursor:pointer">apply</button>
</div>

<div id=list></div>
<div id=empty></div>

<script>
const META = __META__;
const SLOTS = __SLOTS__;
const C = __C__;
document.getElementById("meta").textContent =
  `${META.transitions_scanned.toLocaleString()} transitions scanned from ${META.episodes_scanned.toLocaleString()} episodes · `
  + `${META.n_active_codes_total}/${SLOTS*C} (slot,code) pairs active · `
  + `top-K stored = ${META.top_k}`;

let CODES = [];
fetch("codebook.json").then(r=>r.json()).then(d=>{CODES=d; apply();});

function apply() {
  const slot = parseInt(document.getElementById("slot").value);
  const minc = parseInt(document.getElementById("mincount").value)||0;
  const sort = document.getElementById("sort").value;
  const q = document.getElementById("q").value.trim();
  const lim = parseInt(document.getElementById("limit").value)||200;
  let rows = CODES.filter(r => (slot<0||r.slot===slot) && r.count>=minc);
  if (q) rows = rows.filter(r => String(r.code).includes(q) || String(r.slot)+":"+String(r.code)===q);
  rows.sort((a,b)=>{
    if (sort==="count") return b.count - a.count;
    if (sort==="sim") return b.mean_sim - a.mean_sim;
    return a.code - b.code;
  });
  rows = rows.slice(0, lim);
  const list = document.getElementById("list"); list.innerHTML = "";
  for (const r of rows) {
    const el = document.createElement("div"); el.className="code";
    const lbl = document.createElement("div"); lbl.className="lbl";
    lbl.innerHTML = `<span class="dot s${r.slot}"></span><b>code ${r.code}</b><br><span style="color:var(--muted)">slot ${r.slot} · n=${r.count.toLocaleString()} · sim̄=${r.mean_sim.toFixed(3)}</span>`;
    const img = document.createElement("img"); img.src = r.sprite;
    const acts = document.createElement("div"); acts.className="actions";
    acts.textContent = "actions: " + r.samples.slice(0,8).map(s=>s.action_name).join(" · ");
    const right = document.createElement("div"); right.style.flex="1";
    right.appendChild(img); right.appendChild(acts);
    el.appendChild(lbl); el.appendChild(right);
    list.appendChild(el);
  }
  document.getElementById("empty").textContent = rows.length ? "" : "no codes match your filters.";
}
document.getElementById("apply").addEventListener("click", apply);
["slot","mincount","sort","limit","q"].forEach(id=>{
  const e=document.getElementById(id);
  e.addEventListener("change", apply);
  e.addEventListener("keypress", ev=>{ if (ev.key==="Enter") apply(); });
});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
