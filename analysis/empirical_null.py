"""Empirical random-direction null for the SAE-feature ablation effect.

The single-step causal ablation experiment (analysis/ablate_features.py) compares
the matched SAE-feature direction against ONE random direction as a specificity
control. A single random draw is a weak null: it could be lucky or unlucky.

This script strengthens the control. For each dominant concept and each unlock
moment, we measure the single-step Delta-logP caused by ablating the matched SAE
direction, and we ALSO ablate R=50 independent random unit directions to build an
EMPIRICAL NULL DISTRIBUTION of Delta-logP at the same moments. The matched effect
is then reported as a z-score and an empirical percentile / p-value against that
null (e.g. "matched effect exceeds the 99th percentile of 50 random directions").

Design (matches ablate_features.py EXACTLY):
  * window  = 21-step WM context ending at the unlock step T.
  * gap     = 3: ablate at step T-gap, positions 2..5 (the 4 latent slots) of that
              step, AFTER WM transformer block 1 (target_block_idx=1).
  * effect  = Delta-logP = logp_base - logp_ablated  (>0 means ablation hurts the
              model's prediction of the real codes at step T).
  * matched effect per moment = delta from the SAE decoder column for the concept.
  * null    = R=50 deltas, each from a fresh random direction (same construction as
              the single random control in ablate_features.py: torch.randn_like,
              normalised inside the hook).

We aggregate per concept across the n<=40 moments:
  - matched_mean : mean matched Delta-logP over moments.
  - null_mean/sd : mean and sd of the per-direction mean Delta-logP over the R
                   random directions (the null distribution of the *statistic*
                   "mean Delta-logP over moments").
  - z            : (matched_mean - null_mean) / null_sd.
  - emp_p        : fraction of random directions whose mean Delta-logP >=
                   matched_mean (empirical one-sided p-value).
  - null p95/p99 : 95th / 99th percentile of the null distribution.

Outputs under --out:
  null_effects.json   : per-concept summary (matched mean, null mean/sd/p95/p99,
                        z, empirical p) + the raw R per-direction means.
  empirical_null.html : bar chart (matched vs null p99) + table.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import OmegaConf


# ----------------------------- helpers (from ablate_features.py) -------------
def load_agent(run: Path, device):
    sys.path.insert(0, str(run / "src"))
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    cfg = OmegaConf.load(run / ".hydra" / "config.yaml")
    if cfg.params.tokenizer.num_actions is None: cfg.params.tokenizer.num_actions = 17
    if cfg.params.world_model.num_actions is None: cfg.params.world_model.num_actions = 17
    if cfg.params.actor_critic.model.num_actions is None: cfg.params.actor_critic.model.num_actions = 17
    OmegaConf.resolve(cfg)
    from models.tokenizer import Tokenizer
    from models.world_model import WorldModel
    from models.actor_critic import ActorCritic
    from agent import Agent
    agent = Agent(
        Tokenizer(instantiate(cfg.params.tokenizer)),
        WorldModel(instantiate(cfg.params.world_model)),
        ActorCritic(instantiate(cfg.params.actor_critic)),
    ).to(device).eval()
    agent.load(run / "checkpoints" / "last.pt", device=device,
               load_tokenizer=True, load_world_model=True, load_actor_critic=True, strict=False)
    return agent


def load_sae(sae_path: Path, device):
    """Return (decoder_weight (D, n_features), b_dec (D,), feature metadata)."""
    blob = torch.load(sae_path, map_location=device, weights_only=False)
    sd = blob["state_dict"]
    cfg = blob["config"]
    decoder = sd["decoder.weight"].to(device).float()        # (D, n_features)
    b_dec = sd["b_dec"].to(device).float()                    # (D,)
    return decoder, b_dec, cfg


def pick_targets(features_json: Path, top_n_per_concept: int = 1, min_lift: float = 50.0):
    """Pick the strongest SAE feature per achievement concept.

    Returns list of dicts: {concept, sae_feat_id, ach_lift, ach_p, density}
    """
    feats = json.loads(features_json.read_text())["features"]
    per_concept: dict[str, list] = defaultdict(list)
    for f in feats:
        a = f.get("best_ach")
        if not a or a["lift"] < min_lift: continue
        if f["density"] < 0.001: continue
        per_concept[a["name"]].append((a["lift"], a["p"], f))
    targets = []
    for name, items in per_concept.items():
        items.sort(reverse=True, key=lambda x: x[0])
        for lift, p, f in items[:top_n_per_concept]:
            targets.append(dict(concept=name, sae_feat_id=f["feature"],
                                ach_lift=lift, ach_p=p, density=f["density"],
                                best_code=f.get("best_code"),
                                best_cav=f.get("best_cav")))
    return targets


@torch.no_grad()
def forward_with_optional_ablation(
    agent, obs_t, act_t, lat_t,
    target_block_idx: int = 1,  # WM block to hook (0..num_layers-1)
    direction: torch.Tensor | None = None,
    ablation_step: int = -1,
    strength: float = 1.0,
):
    """Run the WM forward over one chunk; optionally ablate `direction` at the 4
    latent positions of `ablation_step` after block `target_block_idx`.
    Returns the WorldModelOutput.
    """
    wm = agent.world_model
    tpb = wm.config.transformer_config.tokens_per_block            # 6
    L = obs_t.shape[1]                                              # chunk length

    frames_emb = wm.frame_cnn(obs_t)                               # (1, L, 1, D)
    a_emb = wm.act_emb(act_t).unsqueeze(2)                         # (1, L, 1, D)
    l_emb = wm.latents_emb(lat_t)                                  # (1, L, K, D)
    seq = torch.cat([frames_emb, a_emb, l_emb], dim=2)            # (1, L, tpb, D)
    seq_flat = rearrange(seq, "b t p d -> b (t p) d")            # (1, L*tpb, D)

    handle = None
    if direction is not None:
        d = F.normalize(direction.float(), dim=-1).to(seq_flat.device)
        pos_start = ablation_step * tpb + 2     # first latent position of that step
        pos_end = ablation_step * tpb + 6       # 4 latent positions

        def hook(_m, _inp, output):
            out = output
            x = out[:, pos_start:pos_end]                  # (1, 4, D)
            scores = (x @ d).unsqueeze(-1)                  # (1, 4, 1)
            proj = scores * d                                # (1, 4, D)
            new_out = out.clone()
            new_out[:, pos_start:pos_end] = x - strength * proj
            return new_out

        handle = wm.transformer.blocks[target_block_idx].register_forward_hook(hook)

    try:
        wm_out = wm(seq_flat)
    finally:
        if handle is not None: handle.remove()

    return wm_out


def log_prob_codes_at_step(wm_out, real_lat: torch.Tensor, target_step: int) -> float:
    """log P(lat0..lat3 at target_step | context)."""
    logits = wm_out.logits_latents                # may be (B, T, K, V) or similar
    if logits.dim() == 4:
        per_slot_logits = logits[0, target_step]              # (K, V)
    elif logits.dim() == 3:
        T = logits.size(1) // real_lat.size(1)                # K = real_lat shape[1]
        logits_resh = logits.view(logits.size(0), T, -1, logits.size(-1))
        per_slot_logits = logits_resh[0, target_step]
    else:
        raise RuntimeError(f"unexpected logits shape {logits.shape}")
    log_probs = F.log_softmax(per_slot_logits.float(), dim=-1)
    target = real_lat[target_step].long()                     # (K,)
    return float(log_probs.gather(-1, target.unsqueeze(-1)).sum().item())


# ----------------------------- main -----------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--rollouts", required=True, type=Path)
    ap.add_argument("--obs", required=True, type=Path)
    ap.add_argument("--sae", required=True, type=Path,
                    help="path to sae.pt produced by train_sae.py")
    ap.add_argument("--features", required=True, type=Path,
                    help="path to features.json produced by train_sae.py")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-moments", type=int, default=40,
                    help="max unlock moments per concept (caps runtime)")
    ap.add_argument("--n-random", type=int, default=50,
                    help="R: number of random directions in the empirical null")
    ap.add_argument("--max-concepts", type=int, default=8,
                    help="cap to the N highest-lift dominant concepts")
    ap.add_argument("--window", type=int, default=21,
                    help="WM context window length (= max_blocks)")
    ap.add_argument("--gap", type=int, default=3,
                    help="distance between ablation step and target step (target = end-of-window)")
    ap.add_argument("--target-block", type=int, default=1,
                    help="WM transformer block to hook (0-indexed); SAE was trained at block 1")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--min-lift", type=float, default=80.0,
                    help="lift threshold for candidate concepts (matches ablate_features default)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    run = args.run.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("loading agent...", flush=True)
    agent = load_agent(run, device)
    decoder, b_dec, sae_cfg = load_sae(args.sae, device)
    D = decoder.shape[0]
    print(f"SAE: D={D}  n_features={decoder.shape[1]}  layer={sae_cfg['layer']}", flush=True)

    # ----- rollouts (identical loading block to ablate_features.py) ----------
    r = np.load(args.rollouts, allow_pickle=True)
    obs_npz = np.load(args.obs)
    all_obs = obs_npz["obs"]                       # (sum(T+1), 3, 64, 64) uint8
    obs_starts = obs_npz["episode_starts"]
    tokens = r["tokens"]
    actions = r["actions"].astype(np.int64)
    aj = r["ach_just_unlocked"]
    ep_ids = r["episode_ids"].astype(np.int64)
    ep_lens = r["episode_lengths"].astype(np.int64)
    ach_names = [str(x) for x in r["achievement_names"]]
    print(f"rollouts: {tokens.shape[0]} steps  {ep_lens.shape[0]} episodes", flush=True)

    # ----- pick & cap targets to the ~8 dominant concepts -------------------
    targets = pick_targets(args.features, top_n_per_concept=1, min_lift=args.min_lift)
    targets.sort(key=lambda t: t["ach_lift"], reverse=True)      # dominant first
    targets = targets[:args.max_concepts]
    assert len(targets) > 0, "no target concepts after filtering"
    print(f"selected {len(targets)} dominant concepts (R={args.n_random} random dirs each):", flush=True)
    for t in targets:
        print(f"  {t['concept']:25s}  feat #{t['sae_feat_id']:>4d}  lift={t['ach_lift']:7.1f}  P={t['ach_p']*100:5.1f}%", flush=True)

    rng = np.random.default_rng(args.seed)
    # per-direction torch RNG so random directions are reproducible & independent
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    ep_start_row = np.concatenate(([0], np.cumsum(ep_lens)))     # (n_ep+1,)

    target_step = args.window - 1                                # last step in window
    ablation_step = target_step - args.gap                       # gap steps before target
    assert 0 <= ablation_step < args.window, \
        f"ablation_step {ablation_step} out of [0,{args.window})"

    K = int(tokens.shape[1])                                     # latent slots (4)
    trials = []                                                  # raw per-(concept,moment) records
    summary = []
    t0 = time.time()

    for ti, target in enumerate(targets):
        concept = target["concept"]
        concept_idx = ach_names.index(concept)
        direction = decoder[:, target["sae_feat_id"]].clone()   # (D,)
        assert direction.shape == (D,), f"bad SAE direction shape {direction.shape}"

        # unlock moments: rows where this achievement just unlocked, with history
        unlock_global = np.where(aj[:, concept_idx])[0]
        valid_unlock = []
        for g in unlock_global:
            ep = ep_ids[g]
            ep_step = g - ep_start_row[ep]
            if ep_step >= args.window - 1:
                valid_unlock.append(int(g))
        if len(valid_unlock) < 5:
            print(f"  {concept}: not enough unlock moments with history, skipping", flush=True)
            continue
        unlock_sel = rng.choice(valid_unlock,
                                size=min(args.n_moments, len(valid_unlock)),
                                replace=False)
        n_mom = len(unlock_sel)

        # Pre-draw R random directions ONCE per concept and reuse across moments,
        # so the null per-direction "mean over moments" is well defined.
        rand_dirs = torch.randn(args.n_random, D, generator=gen, device=device)  # (R, D)
        assert rand_dirs.shape == (args.n_random, D)

        matched_deltas = np.zeros(n_mom, dtype=np.float64)         # (n_mom,)
        rand_deltas = np.zeros((args.n_random, n_mom), dtype=np.float64)  # (R, n_mom)

        for mi, g in enumerate(unlock_sel):
            g = int(g)
            ep = int(ep_ids[g])
            g_local = int(g - ep_start_row[ep])

            lo = g_local - args.window + 1
            hi = g_local + 1                              # exclusive
            ep_lat = tokens[ep_start_row[ep]:ep_start_row[ep+1]]   # (T, K)
            ep_act = actions[ep_start_row[ep]:ep_start_row[ep+1]]  # (T,)
            ep_obs = all_obs[obs_starts[ep]:obs_starts[ep+1]]      # (T+1, 3, 64, 64)

            lat_chunk = torch.from_numpy(ep_lat[lo:hi].astype(np.int64)).to(device)   # (window, K)
            act_chunk = torch.from_numpy(ep_act[lo:hi]).to(device)                   # (window,)
            obs_chunk = torch.from_numpy(ep_obs[lo:hi+1]).to(device).float().div(255)
            assert lat_chunk.shape == (args.window, K), f"lat_chunk {lat_chunk.shape}"
            assert obs_chunk.shape[0] >= args.window, f"obs_chunk too short {obs_chunk.shape}"

            obs_in = obs_chunk[:args.window].unsqueeze(0)     # (1, window, 3, 64, 64)
            act_in = act_chunk.unsqueeze(0)                   # (1, window)
            lat_in = lat_chunk.unsqueeze(0)                   # (1, window, K)

            # baseline (no intervention)
            wm_out = forward_with_optional_ablation(
                agent, obs_in, act_in, lat_in,
                target_block_idx=args.target_block, direction=None,
                ablation_step=-1, strength=0.0)
            logp_base = log_prob_codes_at_step(wm_out, lat_chunk, target_step)

            # matched SAE-feature ablation
            wm_out = forward_with_optional_ablation(
                agent, obs_in, act_in, lat_in,
                target_block_idx=args.target_block, direction=direction,
                ablation_step=ablation_step, strength=args.strength)
            logp_feat = log_prob_codes_at_step(wm_out, lat_chunk, target_step)
            matched_deltas[mi] = logp_base - logp_feat        # >0 = ablation hurts

            # R random directions (empirical null)
            per_dir = np.zeros(args.n_random, dtype=np.float64)
            for ri in range(args.n_random):
                wm_out = forward_with_optional_ablation(
                    agent, obs_in, act_in, lat_in,
                    target_block_idx=args.target_block, direction=rand_dirs[ri],
                    ablation_step=ablation_step, strength=args.strength)
                logp_r = log_prob_codes_at_step(wm_out, lat_chunk, target_step)
                per_dir[ri] = logp_base - logp_r
            rand_deltas[:, mi] = per_dir

            trials.append(dict(
                concept=concept,
                sae_feat_id=int(target["sae_feat_id"]),
                episode=ep,
                global_step=g,
                ep_step=g_local,
                logp_base=logp_base,
                delta_feat=float(matched_deltas[mi]),
                delta_rand_mean=float(per_dir.mean()),
                delta_rand_std=float(per_dir.std()),
            ))

        # ----- per-concept null statistics ----------------------------------
        matched_mean = float(matched_deltas.mean())              # the statistic
        # null distribution of the SAME statistic ("mean over moments"):
        null_means = rand_deltas.mean(axis=1)                    # (R,)
        null_mean = float(null_means.mean())
        null_sd = float(null_means.std(ddof=1)) if null_means.size > 1 else 0.0
        null_p95 = float(np.percentile(null_means, 95))
        null_p99 = float(np.percentile(null_means, 99))
        z = float((matched_mean - null_mean) / null_sd) if null_sd > 0 else float("inf")
        # one-sided empirical p-value with the standard +1 / R+1 correction
        n_ge = int(np.sum(null_means >= matched_mean))
        emp_p = float((n_ge + 1) / (args.n_random + 1))
        # pooled-trial null for reference (all R x n_mom random deltas flattened)
        pooled = rand_deltas.reshape(-1)
        pooled_mean = float(pooled.mean())
        pooled_sd = float(pooled.std(ddof=1)) if pooled.size > 1 else 0.0

        rec = dict(
            concept=concept,
            sae_feat_id=int(target["sae_feat_id"]),
            ach_lift=float(target["ach_lift"]),
            n_moments=int(n_mom),
            n_random=int(args.n_random),
            matched_mean=matched_mean,
            matched_std=float(matched_deltas.std()),
            null_mean=null_mean,
            null_sd=null_sd,
            null_p95=null_p95,
            null_p99=null_p99,
            z=z,
            emp_p=emp_p,
            exceeds_p99=bool(matched_mean > null_p99),
            exceeds_p95=bool(matched_mean > null_p95),
            pooled_null_mean=pooled_mean,
            pooled_null_sd=pooled_sd,
            null_means=[float(v) for v in null_means],   # raw R per-direction means
        )
        summary.append(rec)

        elapsed = time.time() - t0
        done = len(summary)
        rate = done / max(elapsed, 1e-3)
        eta_min = (len(targets) - ti - 1) / max(rate, 1e-6) / 60
        print(f"  [{ti+1}/{len(targets)}] {concept:22s}  n={n_mom:>2d}  "
              f"matched={matched_mean:+.3f}  null={null_mean:+.3f} (sd {null_sd:.3f})  "
              f"p99={null_p99:+.3f}  z={z:+.2f}  emp_p={emp_p:.3f}  "
              f"{'OK>p99' if rec['exceeds_p99'] else ('>p95' if rec['exceeds_p95'] else '')}  "
              f"eta {eta_min:.1f} min", flush=True)

    print(f"\n{len(summary)} concepts in {(time.time()-t0)/60:.1f} min "
          f"({len(trials)} matched moments)", flush=True)

    (args.out / "null_effects.json").write_text(json.dumps(summary, indent=2))
    (args.out / "trials.json").write_text(json.dumps(trials, indent=2))
    print(f"saved {args.out/'null_effects.json'} + trials.json", flush=True)

    # ----- render -----------------------------------------------------------
    if summary:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        concepts = [s["concept"] for s in summary]
        n_c = len(concepts)
        x = np.arange(n_c)
        width = 0.38

        fig, ax = plt.subplots(figsize=(1.5 + 0.95 * n_c, 5.5))
        matched = [s["matched_mean"] for s in summary]
        nmean = [s["null_mean"] for s in summary]
        nsd = [s["null_sd"] for s in summary]
        np99 = [s["null_p99"] for s in summary]
        ax.bar(x - width / 2, matched, width, label="matched SAE feature",
               color="#d62728", edgecolor="black", linewidth=0.6)
        ax.bar(x + width / 2, nmean, width, yerr=nsd, capsize=3,
               label=f"random null mean +/- sd (R={summary[0]['n_random']})",
               color="#999999", edgecolor="black", linewidth=0.6)
        # mark the 99th percentile of the null
        for xi, p99 in zip(x, np99):
            ax.plot([xi + width / 2 - width / 2, xi + width / 2 + width / 2],
                    [p99, p99], color="#222222", lw=1.4, ls="--")
        ax.plot([], [], color="#222222", lw=1.4, ls="--", label="null p99")
        ax.set_xticks(x)
        ax.set_xticklabels(concepts, rotation=25, ha="right")
        ax.set_ylabel("mean Delta logP(real codes at T)  =  baseline - ablated")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(f"Matched SAE-feature effect vs empirical random-direction null\n"
                     f"ablation at T-{args.gap}, target T (window={args.window}, "
                     f"block {args.target_block}, R={summary[0]['n_random']})")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=140); plt.close(fig)
        img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        parts = [
            "<!doctype html><meta charset=utf-8><title>empirical random-direction null</title>",
            "<style>"
            "body{font:14px ui-monospace,Menlo,Consolas,monospace;background:#0d1117;color:#c9d1d9;padding:18px;}"
            "h1{color:#79c0ff;font-size:18px;margin:0 0 8px;}"
            "h2{color:#d2a8ff;font-size:15px;margin:14px 0 6px;}"
            ".dim{color:#8b949e;}"
            "img{max-width:100%;background:white;border:1px solid #21262d;border-radius:6px;}"
            "table{width:100%;border-collapse:collapse;font-size:12.5px;margin-top:8px;}"
            "th,td{padding:5px 8px;border-bottom:1px solid #21262d;text-align:right;}"
            "td.label{text-align:left;}th{color:#8b949e;text-transform:uppercase;font-size:11px;}"
            ".hi{color:#3fb950;font-weight:600;} .lo{color:#888;}"
            "</style>",
            "<h1>Empirical random-direction null for SAE-feature ablation</h1>",
            f"<div class=dim>{len(summary)} dominant concepts; per concept "
            f"n&le;{args.n_moments} unlock moments, R={summary[0]['n_random']} random unit directions; "
            f"ablation at step T&minus;{args.gap}, target at step T (end of {args.window}-step window); "
            f"hook at WM block {args.target_block}; strength={args.strength}. "
            f"Null statistic = mean Delta logP over moments per random direction.</div>",
            f"<img src='data:image/png;base64,{img_b64}'>",
            "<h2>Summary table</h2>",
            "<table><tr><th class=label>concept</th><th>n</th><th>matched mean</th>"
            "<th>null mean</th><th>null sd</th><th>null p95</th><th>null p99</th>"
            "<th>z</th><th>emp p</th><th>verdict</th></tr>",
        ]
        for s in summary:
            verdict = (">p99" if s["exceeds_p99"] else (">p95" if s["exceeds_p95"] else "ns"))
            cls = "hi" if s["exceeds_p99"] else ("" if s["exceeds_p95"] else "lo")
            zstr = f"{s['z']:+.2f}" if np.isfinite(s["z"]) else "inf"
            parts.append(
                f"<tr><td class=label>{s['concept']}</td>"
                f"<td>{s['n_moments']}</td>"
                f"<td>{s['matched_mean']:+.3f}</td>"
                f"<td>{s['null_mean']:+.3f}</td>"
                f"<td>{s['null_sd']:.3f}</td>"
                f"<td>{s['null_p95']:+.3f}</td>"
                f"<td>{s['null_p99']:+.3f}</td>"
                f"<td class='{cls}'>{zstr}</td>"
                f"<td>{s['emp_p']:.3f}</td>"
                f"<td class='{cls}'>{verdict}</td></tr>")
        parts.append("</table>")
        (args.out / "empirical_null.html").write_text("\n".join(parts), encoding="utf-8")
        print(f"wrote {args.out / 'empirical_null.html'}", flush=True)


if __name__ == "__main__":
    main()
