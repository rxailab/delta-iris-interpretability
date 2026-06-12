"""Render the probe sweep as an HTML heatmap + per-concept breakdown."""
from __future__ import annotations
import argparse, base64, io, json
from pathlib import Path
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True, type=Path,
                    help="dir containing probe_metrics.json + probe_meta.json")
    args = ap.parse_args()

    results = json.loads((args.probes / "probe_metrics.json").read_text())
    meta    = json.loads((args.probes / "probe_meta.json").read_text())
    reps    = meta["representations"]

    # build {label -> {rep -> auroc or acc}}
    by_label: dict[str, dict[str, dict]] = {}
    for r in results:
        by_label.setdefault(r["label"], {})[r["rep"]] = r

    # collect ordered label list — primary metric per family
    def primary(rec):
        if rec is None: return None
        if "auroc" in rec and rec.get("auroc") is not None: return rec["auroc"]
        if "acc" in rec and rec["acc"] is not None: return rec["acc"]
        return None

    labels = sorted(by_label.keys(), key=lambda L: (
        0 if L == "action_taken" else
        1 if L.startswith("reward") else
        2 if L.startswith("ach_just") else 3,
        L,
    ))

    # build heatmap matrix
    mat = np.full((len(labels), len(reps)), np.nan, dtype=np.float64)
    for i, L in enumerate(labels):
        for j, rep in enumerate(reps):
            mat[i, j] = (primary(by_label[L].get(rep)) or np.nan)

    # ----- render image with matplotlib -----------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_h = max(6, 0.20 * len(labels) + 1.0)
    fig, ax = plt.subplots(figsize=(1.2 * len(reps) + 2, fig_h))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(reps))); ax.set_xticklabels(reps, rotation=30, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(reps)):
            v = mat[i, j]
            if np.isnan(v): continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < 0.78 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, label="AUROC (binary) / accuracy (multiclass)")
    ax.set_title("Δ-IRIS layer-wise probes — primary metric per (rep, label)")
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=110); plt.close(fig)
    img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # also dump a CSV for spreadsheets
    csv_lines = ["label," + ",".join(reps)]
    for i, L in enumerate(labels):
        row = ",".join("" if np.isnan(mat[i,j]) else f"{mat[i,j]:.4f}" for j in range(len(reps)))
        csv_lines.append(f"{L},{row}")
    (args.probes / "probe_metrics.csv").write_text("\n".join(csv_lines))

    # ----- HTML --------------------------------------------------------------
    parts = [
        "<!doctype html><meta charset=utf-8><title>Δ-IRIS layer probes</title>",
        "<style>"
        "body{font:14px ui-monospace,Menlo,Consolas,monospace;background:#0d1117;color:#c9d1d9;padding:18px;}"
        "h1{color:#79c0ff;font-size:18px;margin:0 0 12px;}"
        "h2{color:#d2a8ff;font-size:16px;margin:18px 0 6px;border-bottom:1px solid #30363d;padding-bottom:4px;}"
        "table{border-collapse:collapse;font-size:13px;margin-bottom:14px;}"
        "th,td{padding:5px 8px;text-align:right;border-bottom:1px solid #21262d;}"
        "th{color:#8b949e;text-transform:uppercase;font-size:11px;}"
        "td.label{text-align:left;color:#c9d1d9;}"
        ".dim{color:#8b949e;}"
        "img{max-width:100%;}"
        "</style>",
        "<h1>Δ-IRIS layer-wise linear probes</h1>",
        f"<div class=dim>{meta['n_samples']:,} samples · {meta['n_episodes']} episodes · "
        f"reps: {', '.join(reps)} · primary metric = AUROC for binary, accuracy for action</div>",
        f"<img src='data:image/png;base64,{img_b64}'>",
        "<h2>Full table</h2>",
        "<table><tr><th>label</th>" + "".join(f"<th>{r}</th>" for r in reps) + "<th>n_pos_test</th></tr>",
    ]
    for L in labels:
        row = [f"<tr><td class=label>{L}</td>"]
        npos = "—"
        for r in reps:
            rec = by_label[L].get(r)
            if not rec:
                row.append("<td>—</td>"); continue
            if rec.get("skipped"):
                row.append(f"<td class=dim>skip</td>"); continue
            v = primary(rec)
            row.append(f"<td>{v:.3f}</td>" if v is not None else "<td>—</td>")
            if "n_pos_test" in rec:
                npos = str(rec["n_pos_test"])
        row.append(f"<td class=dim>{npos}</td></tr>")
        parts.append("".join(row))
    parts.append("</table>")
    (args.probes / "probes.html").write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {args.probes / 'probes.html'} + probe_metrics.csv")


if __name__ == "__main__":
    main()
