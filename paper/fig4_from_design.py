"""Build Figure 4 (single-step ablation) from the Claude Design source HTML,
but with the REAL data substituted for the design's digitized placeholders.

The design HTML hardcodes approximate, eye-digitized bar values (and synthetic
gap-3 jitter). We keep its exact visual design — two-panel deep-blue / light-blue
/ grey bars, zoomed gap-3 inset, zebra banding, CI whiskers, value annotations —
but replace CONCEPTS / G0 / G3 with values loaded from the actual result files:

  results/ablation_gap0/causal_effects.json   (ablate at target step T)
  results/ablation_gap3/causal_effects.json   (ablate three steps before T)

Series mapping:
  feat_unlock   = (moment_type=unlock,   condition=feat)
  feat_ordinary = (moment_type=ordinary, condition=feat)
  rand_unlock   = (moment_type=unlock,   condition=rand)

Row order: concepts sorted by gap-0 feat_unlock mean, descending (as in the design).

Pipeline mirrors fig1_from_design.py: substitute data into the design JS, run it
in Node with a DOM shim, patch the SVG for cairosvg, convert to PDF/PNG/SVG.

Run:  python fig4_from_design.py
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "design-src" / "Fig4-Single-Step-Ablation.html"
OUT = HERE / "figures"
RESULTS = HERE.parent / "results"

TWEAKS = ('{ showTitle: false, showCI: true, zebra: true, '
          'annotate: true, palette: "color", nullPanel: "zoom" }')


def load(path: Path) -> dict:
    m = {}
    for r in json.loads(path.read_text()):
        m[(r["concept"], r["moment_type"], r["condition"])] = (r["mean"], r["ci_lo"], r["ci_hi"])
    return m


def js_triplet(v):
    return "[" + ", ".join(f"{x:.4f}" for x in v) + "]"


def build_data_block() -> str:
    g0 = load(RESULTS / "ablation_gap0" / "causal_effects.json")
    g3 = load(RESULTS / "ablation_gap3" / "causal_effects.json")
    concepts = sorted({k[0] for k in g0}, key=lambda c: -g0[(c, "unlock", "feat")][0])
    series = [("feat_unlock", "unlock", "feat"),
              ("feat_ordinary", "ordinary", "feat"),
              ("rand_unlock", "unlock", "rand")]

    def block(name, src):
        lines = [f"  const {name} = {{"]
        for key, mt, cond in series:
            row = ", ".join(js_triplet(src[(c, mt, cond)]) for c in concepts)
            lines.append(f"    {key}: [{row}],")
        lines.append("  };")
        return "\n".join(lines)

    concepts_js = "  const CONCEPTS = [\n    " + ", ".join(
        f"'{c.replace('_', ' ')}'" for c in concepts) + "\n  ];"
    return concepts_js + "\n\n" + block("G0", g0) + "\n\n" + block("G3", g3) + "\n"


def extract_js(html: str) -> str:
    m = re.search(r'^<script>\n(.*?)^</script>$', html, re.S | re.M)
    if not m:
        raise SystemExit("could not find the inline figure script in the design HTML")
    return m.group(1)


def substitute_data(core: str, data_block: str) -> str:
    # Replace everything from 'const CONCEPTS = [' up to (but not including)
    # 'const SERIES = [' with our real-data block.
    pat = re.compile(r'  const CONCEPTS = \[.*?(?=  const SERIES = \[)', re.S)
    new, n = pat.subn(data_block + "\n", core)
    if n != 1:
        raise SystemExit(f"data-substitution matched {n} times (expected 1)")
    return new


def run_node(core_js: str) -> str:
    with tempfile.TemporaryDirectory() as td:
        core = Path(td) / "core.js"
        core.write_text(core_js, encoding="utf-8")
        runner = Path(td) / "run.js"
        runner.write_text(
            "let captured = null;\n"
            "globalThis.window = globalThis;\n"
            "globalThis.document = {\n"
            "  getElementById: () => ({ set innerHTML(v) { captured = v; } }),\n"
            "  addEventListener: () => {},\n"
            "};\n"
            f"require({str(core)!r});\n"
            f"window.renderFigure({TWEAKS});\n"
            "process.stdout.write(captured);\n",
            encoding="utf-8")
        return subprocess.run(["node", str(runner)], check=True,
                              capture_output=True, encoding="utf-8").stdout


def main() -> None:
    html = SRC.read_text(encoding="utf-8")
    core = substitute_data(extract_js(html), build_data_block())
    svg = run_node(core)

    # Tighten the top margin (title is off): the topmost mark is the legend at y~95.
    m = re.search(r'viewBox="0 0 (\d+) (\d+)"', svg)
    W, H = int(m.group(1)), int(m.group(2))
    crop = 86
    svg = svg.replace(f'viewBox="0 0 {W} {H}"', f'viewBox="0 {crop} {W} {H - crop}"', 1)
    svg = svg.replace(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"></rect>',
                      f'<rect x="0" y="{crop}" width="{W}" height="{H - crop}" fill="#ffffff"></rect>', 1)

    OUT.mkdir(exist_ok=True)
    (OUT / "ablation_onestep.svg").write_text(svg, encoding="utf-8")

    import cairosvg
    cairosvg.svg2pdf(bytestring=svg.encode("utf-8"), write_to=str(OUT / "ablation_onestep.pdf"))
    cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=str(OUT / "ablation_onestep.png"),
                     output_width=1480)
    print("wrote", OUT / "ablation_onestep.pdf")


if __name__ == "__main__":
    main()
