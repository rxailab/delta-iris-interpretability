"""Build Figure 1 (audit overview) from the Claude Design source HTML.

Pipeline:
  1. Extract the inline SVG-generator <script> from design-src/Fig1-Audit-Overview.html
     and execute it in Node with a tiny DOM shim (tweaks: title off, legend on).
  2. Patch the emitted SVG: crop the empty title band, normalise marker orientation
     for cairosvg, and flatten <tspan> sub/superscripts to plain text
     (cairosvg mis-positions anchored multi-tspan runs; IBM Plex lacks
     U+209C/U+208A/U+1D40/U+22A4, so non-digit subscripts become code-style "_t"
     and the transpose formula becomes the dot-product form).
  3. Convert to figures/audit_overview.pdf + .png with cairosvg
     (requires IBM Plex Sans/Mono installed, e.g. ~/.local/share/fonts + fc-cache).

Run:  python fig1_from_design.py
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "design-src" / "Fig1-Audit-Overview.html"
OUT = HERE / "figures"

TWEAKS = '{ showTitle: false, showLegend: true, dimArch: false }'

SUB_DIGIT = {'0': '₀', '1': '₁', '2': '₂', '3': '₃'}
SUP_DIGIT = {'0': '⁰', '1': '¹', '2': '²', '3': '³'}


def extract_js(html: str) -> str:
    m = re.search(r'^<script>\n(.*?)^</script>$', html, re.S | re.M)
    if not m:
        raise SystemExit("could not find the inline figure script in the design HTML")
    return m.group(1)


def run_node(core_js: str) -> str:
    with tempfile.TemporaryDirectory() as td:
        core = Path(td) / "core.js"
        core.write_text(core_js, encoding="utf-8")
        runner = Path(td) / "run.js"
        runner.write_text(
            "const fs = require('fs');\n"
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


def flatten_tspans(svg: str) -> str:
    def flatten(m):
        attrs, body = m.group(1), m.group(2)
        shift = 0.0
        out = []
        for tm in re.finditer(r'<tspan dy="(-?[\d.]+)"[^>]*>([^<]*)</tspan>', body):
            dy, content = float(tm.group(1)), tm.group(2)
            shift += dy
            if shift > 0.5:        # subscript run
                if all(ch in SUB_DIGIT for ch in content):
                    content = ''.join(SUB_DIGIT[ch] for ch in content)
                else:               # e.g. "t", "t+1" -> code-style suffix
                    content = '_' + content
            elif shift < -0.5:     # superscript run
                if all(ch in SUP_DIGIT for ch in content):
                    content = ''.join(SUP_DIGIT[ch] for ch in content)
                elif content == '⊤':   # transpose -> dot-product form
                    content = '·'
            out.append(content)
        return f'<text {attrs}>{"".join(out)}</text>'

    return re.sub(r'<text ([^>]*)>((?:<tspan[^>]*>[^<]*</tspan>)+)</text>', flatten, svg)


def main() -> None:
    html = SRC.read_text(encoding="utf-8")
    svg = run_node(extract_js(html))

    # crop the empty title band (title suppressed; content starts at y=100)
    svg = svg.replace('viewBox="0 0 1480 960"', 'viewBox="0 84 1480 876"', 1)
    svg = svg.replace('<rect x="0" y="0" width="1480" height="960" fill="#ffffff"></rect>',
                      '<rect x="0" y="84" width="1480" height="876" fill="#ffffff"></rect>', 1)
    # cairosvg marker compatibility (only marker-end is used, so plain auto is equivalent)
    svg = svg.replace('orient="auto-start-reverse"', 'orient="auto"')
    svg = flatten_tspans(svg)

    OUT.mkdir(exist_ok=True)
    (OUT / "audit_overview.svg").write_text(svg, encoding="utf-8")

    import cairosvg
    cairosvg.svg2pdf(bytestring=svg.encode("utf-8"), write_to=str(OUT / "audit_overview.pdf"))
    cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=str(OUT / "audit_overview.png"),
                     output_width=1480)
    print("wrote", OUT / "audit_overview.pdf")


if __name__ == "__main__":
    sys.exit(main())
