"""For each (slot, code, achievement) compute lift, support, and a binomial p-value
of the null hypothesis that the code is independent of the achievement unlock.

Inputs:
  - rollouts.npz produced by rollout_with_info.py
  - (optional) gallery sprites from render_gallery.py so the HTML can show
    visual examples next to each association

Output (under --out):
  - mi_table.json       : ranked list of (slot, code, achievement, lift, p)
  - achievements.html   : one section per achievement with top-K associated codes
"""
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollouts", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--gallery", type=Path, default=None,
                    help="path to a gallery dir from render_gallery.py; if given,"
                         " HTML will inline sprite thumbnails")
    ap.add_argument("--top-k", type=int, default=12,
                    help="show top-K codes per achievement")
    ap.add_argument("--min-support", type=int, default=20,
                    help="ignore (slot, code, achievement) cells with fewer co-occurrences")
    args = ap.parse_args()

    d = np.load(args.rollouts, allow_pickle=True)
    tokens = d["tokens"]                 # (N, 4) int
    aj = d["ach_just_unlocked"]          # (N, 22) bool
    ach_names = [str(x) for x in d["achievement_names"]]
    N, K = tokens.shape
    A = aj.shape[1]
    C = int(tokens.max()) + 1
    print(f"loaded {N} steps · K_slots={K} · codes seen ≤ {C} · achievements={A}")

    # Per-(slot, code) baseline frequency P(c | slot=k) = count / N
    counts = np.zeros((K, C), dtype=np.int64)
    for k in range(K):
        np.add.at(counts[k], tokens[:, k], 1)
    p_code = counts / max(N, 1)

    # Per-achievement count of "just-unlocked" events.
    ach_count = aj.sum(axis=0)   # (A,)
    print("unlock counts per achievement:",
          ", ".join(f"{n}={c}" for n, c in zip(ach_names, ach_count) if c > 0))

    # For each achievement, for each (slot, code), count co-occurrences.
    # co[k, c, a] = #steps where token at slot k = c AND achievement a just unlocked at that step
    records: list[dict] = []
    for a in range(A):
        mask = aj[:, a]                           # (N,)
        n_a = int(mask.sum())
        if n_a == 0: continue
        for k in range(K):
            tok_k = tokens[:, k]
            # for codes appearing in unlock steps, count co-occurrences
            unlock_codes, unlock_counts = np.unique(tok_k[mask], return_counts=True)
            for c, n_kc_a in zip(unlock_codes, unlock_counts):
                if n_kc_a < args.min_support: continue
                # expected under independence:
                expected = n_a * p_code[k, int(c)]
                if expected <= 0: continue
                lift = float(n_kc_a / expected)
                p_a_given_c = n_kc_a / max(counts[k, int(c)], 1)
                # binomial test: under H0 the n_kc_a successes are drawn from
                # Bernoulli(p=n_a/N) over counts[k, c] trials
                p_baseline = n_a / N
                try:
                    pv = float(binomtest(int(n_kc_a), int(counts[k, int(c)]),
                                         p_baseline, alternative="greater").pvalue)
                except Exception:
                    pv = float("nan")
                records.append(dict(
                    achievement=ach_names[a],
                    slot=int(k), code=int(c),
                    n_unlock=int(n_a),
                    n_co=int(n_kc_a),
                    n_code=int(counts[k, int(c)]),
                    p_a_given_c=p_a_given_c,
                    lift=lift,
                    pvalue=pv,
                ))

    records.sort(key=lambda r: (-r["lift"], r["pvalue"]))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "mi_table.json").write_text(json.dumps(records, indent=2))
    print(f"wrote {len(records)} significant (slot, code, achievement) cells → {args.out/'mi_table.json'}")

    # ---- HTML report ----
    sprite_rel = None
    if args.gallery is not None and args.gallery.exists():
        sprite_rel = lambda k, c: f"sprites/s{k}_c{c}.png"
    parts = ['<!doctype html><meta charset=utf-8><title>code↔achievement associations</title>',
             '<style>',
             'body{font:14px ui-monospace,Menlo,Consolas,monospace;background:#0d1117;color:#c9d1d9;padding:18px;}',
             'h1{color:#79c0ff;font-size:18px;margin:0 0 12px;}',
             'h2{color:#d2a8ff;font-size:16px;margin:18px 0 6px;border-bottom:1px solid #30363d;padding-bottom:4px;}',
             'table{width:100%;border-collapse:collapse;font-size:13px;margin-bottom:12px;}',
             'th,td{padding:5px 10px;text-align:left;border-bottom:1px solid #21262d;}',
             'th{color:#8b949e;text-transform:uppercase;letter-spacing:.08em;font-size:11px;font-weight:600;}',
             'img{max-height:64px;image-rendering:pixelated;border-radius:3px;}',
             '.lift{color:#3fb950;font-weight:600;}',
             '.dim{color:#8b949e;}',
             '.s0{color:#d29922;} .s1{color:#3fb950;} .s2{color:#58a6ff;} .s3{color:#d2a8ff;}',
             '</style>',
             '<h1>Δ-IRIS code → achievement associations</h1>',
             f'<div class=dim>{N:,} rollout steps · top-{args.top_k} codes per achievement, sorted by lift; '
             f'min co-occurrence support = {args.min_support}</div>']

    for a, name in enumerate(ach_names):
        rs = [r for r in records if r["achievement"] == name][:args.top_k]
        if not rs: continue
        n_a = rs[0]["n_unlock"]
        parts.append(f'<h2>{html.escape(name)} <span class=dim>· {n_a} unlocks</span></h2>')
        parts.append('<table><tr><th>slot</th><th>code</th><th>n_unlock∩code</th>'
                     '<th>P(ach | code)</th><th>lift</th><th>p-value</th>'
                     + ('<th>sprite</th>' if sprite_rel else '') + '</tr>')
        for r in rs:
            sprite_html = ''
            if sprite_rel is not None:
                p = args.gallery / sprite_rel(r["slot"], r["code"])
                if p.exists():
                    sprite_html = f'<td><img src="../gallery/{sprite_rel(r["slot"],r["code"])}"></td>'
                else:
                    sprite_html = '<td class=dim>—</td>'
            parts.append(
                f'<tr><td class="s{r["slot"]}">{r["slot"]}</td>'
                f'<td>{r["code"]}</td>'
                f'<td>{r["n_co"]:,} / {r["n_code"]:,}</td>'
                f'<td>{r["p_a_given_c"]*100:.1f}%</td>'
                f'<td class=lift>{r["lift"]:.1f}×</td>'
                f'<td class=dim>{r["pvalue"]:.1e}</td>'
                f'{sprite_html}</tr>')
        parts.append('</table>')

    (args.out / "achievements.html").write_text("\n".join(parts))
    print(f"wrote → {args.out / 'achievements.html'}")


if __name__ == "__main__":
    main()
