"""Build Figure 5 (imagination ablation) from the Claude Design source HTML,
with the REAL data substituted for the design's digitized placeholders.

The design HTML hardcodes approximate, eye-digitized probabilities (and, e.g.,
shows no effect for collect_wood, which is in fact the headline 1.00->0.00 drop).
We keep its exact visual design — the default "effect" view: per-concept
lollipops showing the change in detector-code probability relative to the
no-intervention baseline (the 0 line), with a baseline 95% CI band — but
replace CONCEPTS / PF / PW with values loaded from the actual result file:

  results/imagination/imagine_effects.json

Series: baseline, sae, cav, random. We keep the 15 concepts with baseline
within-horizon probability >= 0.4 (the design's own inclusion rule), and let
the design JS sort rows by effect size.

Because the real SAE effect for collect_wood reaches -1.0 (vs the design's
+/-0.45 axis), we widen the effect-view x-axis to fit the data.

Pipeline mirrors fig4_from_design.py. The design sets both #fig and #cap, so
the DOM shim captures innerHTML per element id and we keep only #fig.

Run:  python fig5_from_design.py
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "design-src" / "Fig5-Imagination-Ablation.html"
OUT = HERE / "figures"
RESULTS = HERE.parent / "results"

TWEAKS = ('{ showTitle: false, view: "effect", showCI: true, '
          'zebra: true, sort: "effect", palette: "color" }')

# widened effect-view axis (design used -0.45..0.12; real data reaches -1.0)
NEW_AXIS = ('      xmin = -1.05; xmax = 0.12; ticks = [-1, -0.75, -0.5, -0.25, 0]; '
            'fmt = fmtD;')
OLD_AXIS_RE = re.compile(
    r'      xmin = -0\.45; xmax = 0\.12; ticks = \[-0\.4, -0\.3, -0\.2, -0\.1, 0, 0\.1\]; fmt = fmtD;')


def js_triplet(v):
    return "[" + ", ".join(f"{x:.4f}" for x in v) + "]"


def build_data_block() -> str:
    recs = {}
    for r in json.loads((RESULTS / "imagination" / "imagine_effects.json").read_text()):
        recs[(r["concept"], r["condition"])] = r
    concepts = sorted({k[0] for k in recs})
    kept = [c for c in concepts if recs[(c, "baseline")]["p_within"] >= 0.4]
    conds = ["baseline", "sae", "cav", "random"]

    def block(name, lo_key, mid_key, hi_key):
        lines = [f"  const {name} = {{"]
        for cond in conds:
            row = ", ".join(
                js_triplet([recs[(c, cond)][mid_key], recs[(c, cond)][lo_key],
                            recs[(c, cond)][hi_key]]) for c in kept)
            lines.append(f"    {cond}: [{row}],")
        lines.append("  };")
        return "\n".join(lines)

    concepts_js = "  const CONCEPTS = [\n    " + ", ".join(
        f"'{c.replace('_', ' ')}'" for c in kept) + "\n  ];"
    pf = block("PF", "p_first_lo", "p_first", "p_first_hi")
    pw = block("PW", "p_within_lo", "p_within", "p_within_hi")
    return concepts_js + "\n\n" + pf + "\n\n" + pw + "\n"


def extract_js(html: str) -> str:
    m = re.search(r'^<script>\n(.*?)^</script>$', html, re.S | re.M)
    if not m:
        raise SystemExit("could not find the inline figure script in the design HTML")
    return m.group(1)


def substitute(core: str, data_block: str) -> str:
    pat = re.compile(r'  const CONCEPTS = \[.*?(?=  const SERIES = \[)', re.S)
    core, n = pat.subn(data_block + "\n", core)
    if n != 1:
        raise SystemExit(f"data-substitution matched {n} times (expected 1)")
    core, n = OLD_AXIS_RE.subn(NEW_AXIS, core)
    if n != 1:
        raise SystemExit(f"axis-substitution matched {n} times (expected 1)")
    return core


def run_node(core_js: str) -> str:
    with tempfile.TemporaryDirectory() as td:
        core = Path(td) / "core.js"
        core.write_text(core_js, encoding="utf-8")
        runner = Path(td) / "run.js"
        # id-aware shim: the design writes both #fig and #cap; keep only #fig.
        runner.write_text(
            "const store = {};\n"
            "globalThis.window = globalThis;\n"
            "globalThis.document = {\n"
            "  getElementById: (id) => ({ set innerHTML(v) { store[id] = v; } }),\n"
            "  addEventListener: () => {},\n"
            "};\n"
            f"require({str(core)!r});\n"
            f"window.renderFigure({TWEAKS});\n"
            "process.stdout.write(store['fig']);\n",
            encoding="utf-8")
        return subprocess.run(["node", str(runner)], check=True,
                              capture_output=True, encoding="utf-8").stdout


def main() -> None:
    html = SRC.read_text(encoding="utf-8")
    svg = run_node(substitute(extract_js(html), build_data_block()))

    # tighten top margin (title off; topmost mark is the legend ~y=96)
    m = re.search(r'viewBox="0 0 (\d+) (\d+)"', svg)
    W, H = int(m.group(1)), int(m.group(2))
    crop = 96
    svg = svg.replace(f'viewBox="0 0 {W} {H}"', f'viewBox="0 {crop} {W} {H - crop}"', 1)
    svg = svg.replace(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"></rect>',
                      f'<rect x="0" y="{crop}" width="{W}" height="{H - crop}" fill="#ffffff"></rect>', 1)

    OUT.mkdir(exist_ok=True)
    (OUT / "imagination.svg").write_text(svg, encoding="utf-8")

    import cairosvg
    cairosvg.svg2pdf(bytestring=svg.encode("utf-8"), write_to=str(OUT / "imagination.pdf"))
    cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=str(OUT / "imagination.png"),
                     output_width=1480)
    print("wrote", OUT / "imagination.pdf")


if __name__ == "__main__":
    main()
