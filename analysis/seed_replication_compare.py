"""Compare the headline dissociation across the original agent and the two seeds.
Prints a replication table: codebook use, event-vocabulary lift, decode AUROC,
HUD shortcut (pre-transformer rep vs blocks on cumulative state), the SAE single-
step causal effect vs random, and the behavioural cascade (write-does-not-drive).
"""
import json, numpy as np
from pathlib import Path

BASE = Path("/mmfs1/scratch/hpc/11/xiar3/expwm-runs")
AGENTS = [("original", "analysis-21531393"), ("seed1", "analysis-seed1"), ("seed2", "analysis-seed2")]
WOOD_TOOLS = ["make_wood_pickaxe", "make_wood_sword", "place_table"]  # depth-1 downstream tier

def jload(p): return json.loads(Path(p).read_text())

def codebook(d):
    z = np.load(d / "codebook_stats.npz")
    c = z["counts"]                       # (4 slots, 1024)
    active = int((c.sum(0) > 0).sum())
    ent = []
    for s in range(c.shape[0]):
        p = c[s] / max(c[s].sum(), 1); p = p[p > 0]
        ent.append(float(-(p * np.log2(p)).sum()))
    return active, np.mean(ent)

def mi_stats(d):
    rows = jload(d / "mi" / "mi_table.json")
    per = {}
    for r in rows:
        if r.get("lift") is None:
            continue
        a = r["achievement"]
        if a not in per or r["lift"] > per[a]["lift"]:
            per[a] = r
    lifts = [v["lift"] for v in per.values()]
    pbest = [v.get("p_a_given_c") for v in per.values() if v.get("p_a_given_c") is not None]
    return max(lifts), float(np.mean(pbest)), sum(p >= 0.5 for p in pbest)

def probes(d):
    pm = jload(d / "probes" / "probe_metrics.json")
    by = {}
    for r in pm:
        by.setdefault(r["rep"], {})[r["label"]] = r
    blocks = [k for k in by if k.startswith("wm_block")]
    ctrl = "wm_input"
    # decode: just-unlocked from raw codes
    just = [v["auroc"] for k, v in by.get("raw_codes", {}).items()
            if k.startswith("ach_just[") and v.get("auroc") is not None]
    dec_mean = float(np.mean(just)); dec_ge85 = sum(a >= 0.85 for a in just); dec_n = len(just)
    # HUD shortcut: pre-transformer ctrl vs best block on cumulative concepts
    wins = tot = 0; ctrl_a = []; blk_a = []
    for lab in by.get(ctrl, {}):
        if not lab.startswith("ach_cum["): continue
        cv = by[ctrl][lab].get("auroc")
        bv = [by[b][lab]["auroc"] for b in blocks
              if lab in by[b] and by[b][lab].get("auroc") is not None]
        if cv is None or not bv: continue
        tot += 1; ctrl_a.append(cv); blk_a.append(max(bv))
        if cv >= max(bv) - 0.01: wins += 1
    return dec_mean, dec_ge85, dec_n, wins, tot, float(np.mean(ctrl_a)), float(np.mean(blk_a))

def ablation(d):
    ce = jload(d / "ablation_gap0" / "causal_effects.json")
    feat = [r["mean"] for r in ce if r["condition"] == "feat" and r["moment_type"] == "unlock"]
    rand = [r["mean"] for r in ce if r["condition"] == "rand" and r["moment_type"] == "unlock"]
    return float(np.mean(np.abs(feat))), float(np.mean(np.abs(rand)))

def cascade(d):
    ce = jload(d / "cascade_collect_wood" / "cascade_effects.json")
    pa = {r["achievement"]: r for r in ce["per_ach"]}
    b = float(np.mean([pa[a]["baseline"]["p"] for a in WOOD_TOOLS if a in pa]))
    s = float(np.mean([pa[a]["sae"]["p"] for a in WOOD_TOOLS if a in pa]))
    wood_b = pa["collect_wood"]["baseline"]["p"] if "collect_wood" in pa else float("nan")
    wood_s = pa["collect_wood"]["sae"]["p"] if "collect_wood" in pa else float("nan")
    return ce["feat"], b, s, ce["return"]["baseline"]["mean"], ce["return"]["sae"]["mean"], wood_b, wood_s

rows = {}
for name, sub in AGENTS:
    d = BASE / sub
    try:
        rows[name] = dict(cb=codebook(d), mi=mi_stats(d), pr=probes(d), ab=ablation(d), ca=cascade(d))
    except Exception as e:
        import traceback; traceback.print_exc()
        rows[name] = {"error": repr(e)}

def col(name): return rows[name]
print(f"\n{'METRIC':<46}{'original':>16}{'seed1':>16}{'seed2':>16}")
print("-" * 94)
def line(lbl, fn):
    print(f"{lbl:<46}" + "".join(f"{fn(col(n)):>16}" if 'error' not in col(n) else f"{'ERR':>16}" for n, _ in AGENTS))

line("Codebook: active codes /1024",         lambda r: f"{r['cb'][0]}")
line("Codebook: mean per-slot entropy (bits)",lambda r: f"{r['cb'][1]:.2f}")
line("Event vocab: max lift (x chance)",      lambda r: f"{r['mi'][0]:.0f}")
line("Event vocab: mean best P(ach|code)",    lambda r: f"{r['mi'][1]:.2f}")
line("Decode just-unlock: mean AUROC (raw codes)", lambda r: f"{r['pr'][0]:.3f}")
line("Decode: # achievements AUROC>=0.85",    lambda r: f"{r['pr'][1]}/{r['pr'][2]}")
line("HUD shortcut: ctrl>=blocks (cumul.)",   lambda r: f"{r['pr'][3]}/{r['pr'][4]}")
line("HUD: mean ctrl AUROC (cumul.)",         lambda r: f"{r['pr'][5]:.3f}")
line("HUD: mean best-block AUROC (cumul.)",   lambda r: f"{r['pr'][6]:.3f}")
line("Read-write: |SAE| dlogP @unlock",       lambda r: f"{r['ab'][0]:.3f}")
line("Read-write: |random| dlogP @unlock",    lambda r: f"{r['ab'][1]:.3f}")
line("Cascade: wood SAE feature id",          lambda r: f"{r['ca'][0]}")
line("Cascade: wood code base->SAE",          lambda r: f"{r['ca'][5]:.2f}->{r['ca'][6]:.2f}")
line("Cascade: downstream tools base->SAE",   lambda r: f"{r['ca'][1]:.2f}->{r['ca'][2]:.2f}")
line("Cascade: imagined return base->SAE",    lambda r: f"{r['ca'][3]:.2f}->{r['ca'][4]:.2f}")
for n, _ in AGENTS:
    if 'error' in rows[n]: print(f"\n{n} ERROR: {rows[n]['error']}")
