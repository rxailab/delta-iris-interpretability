"""Causal feature ablation: does removing an SAE-feature direction from the WM's
residual stream actually break the model's predictions about the corresponding
concept?

Setup (per target concept):
  1. Pick the strongest SAE feature for the concept (from sae/features.json).
  2. From the rollouts buffer:
       - "unlock moments" = steps where the concept's achievement just unlocked
       - "ordinary moments" = random other steps (matched count, same episodes)
  3. For each moment T, build a 21-step context window ending at T.
  4. Run the world model forward 3 times for each trial:
       (a) baseline    : no intervention
       (b) feature abl : subtract the SAE decoder direction at step T-3,
                         positions 2..5 (the 4 latent slots) of that step
       (c) random  abl : same intervention with a random unit direction
                         (specificity control)
  5. Measure log P(real lat₀..lat₃ at step T  |  context) under each condition.
  6. Aggregate Δ_logP per (concept, moment-type, condition) with bootstrap CIs.

Output:
  causal_effects.json   : per-(concept,moment_type,condition) summary stats
  trials.json           : every trial's raw numbers (for re-analysis)
  intervention.html     : bar chart + per-concept breakdown
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


# ----------------------------- helpers --------------------------------------
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
    # Reconstruct enough of the SAE: we only need decoder.weight and b_dec.
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


# ----------------------------- core experiment ------------------------------
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

    # build input sequence
    frames_emb = wm.frame_cnn(obs_t)                                # (1, L, 1, D)
    a_emb = wm.act_emb(act_t).unsqueeze(2)                          # (1, L, 1, D)
    l_emb = wm.latents_emb(lat_t)                                   # (1, L, K, D)
    seq = torch.cat([frames_emb, a_emb, l_emb], dim=2)              # (1, L, tpb, D)
    seq_flat = rearrange(seq, "b t p d -> b (t p) d")               # (1, L*tpb, D)

    handle = None
    if direction is not None:
        d = F.normalize(direction.float(), dim=-1).to(seq_flat.device)
        pos_start = ablation_step * tpb + 2     # first latent position of that step
        pos_end = ablation_step * tpb + 6       # 4 latent positions

        def hook(_m, _inp, output):
            out = output
            # Project out direction at the 4 latent positions of the ablation step.
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
    """log P(lat₀..lat₃ at target_step | context).

    wm_out.logits_latents shape (after Head): (B, T_pred, K, V) where T_pred is
    the number of timesteps for which the head outputs predictions.
    """
    logits = wm_out.logits_latents                # may be (B, T, K, V) or similar
    if logits.dim() == 4:
        per_slot_logits = logits[0, target_step]              # (K, V)
    elif logits.dim() == 3:
        # head emitted (B, T*K, V) — reshape
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
    ap.add_argument("--n-trials-per-set", type=int, default=40)
    ap.add_argument("--window", type=int, default=21,
                    help="WM context window length (= max_blocks)")
    ap.add_argument("--gap", type=int, default=3,
                    help="distance between ablation step and target step (target = end-of-window)")
    ap.add_argument("--target-block", type=int, default=1,
                    help="WM transformer block to hook (0-indexed); SAE was trained at block 1")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    run = args.run.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("loading agent…", flush=True)
    agent = load_agent(run, device)
    decoder, b_dec, sae_cfg = load_sae(args.sae, device)
    print(f"SAE: D={decoder.shape[0]}  n_features={decoder.shape[1]}  layer={sae_cfg['layer']}", flush=True)

    # rollouts
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
    print(f"rollouts: {tokens.shape[0]} steps · {ep_lens.shape[0]} episodes", flush=True)

    # pick targets
    targets = pick_targets(args.features, top_n_per_concept=1, min_lift=80.0)
    print(f"selected {len(targets)} target concepts:", flush=True)
    for t in targets:
        print(f"  {t['concept']:25s}  feat #{t['sae_feat_id']:>4d}  lift={t['ach_lift']:6.1f}  P={t['ach_p']*100:5.1f}%", flush=True)

    rng = np.random.default_rng(args.seed)

    # episode_starts gives obs offset per episode; build per-episode ranges of token rows
    ep_start_row = np.concatenate(([0], np.cumsum(ep_lens)))     # (n_ep+1,)

    trials = []
    t0 = time.time()
    for ti, target in enumerate(targets):
        concept_idx = ach_names.index(target["concept"])
        direction = decoder[:, target["sae_feat_id"]].clone()    # (D,)

        # unlock moments: step indices (global row in `tokens`) where aj fired
        unlock_global = np.where(aj[:, concept_idx])[0]
        # filter: ensure step has enough history (>= args.window-1 steps prior in episode)
        valid_unlock = []
        for g in unlock_global:
            ep = ep_ids[g]
            ep_step = g - ep_start_row[ep]
            if ep_step >= args.window - 1:
                valid_unlock.append(g)
        if len(valid_unlock) < 5:
            print(f"  {target['concept']}: not enough unlock moments with history, skipping", flush=True)
            continue
        # sample N
        unlock_sel = rng.choice(valid_unlock, size=min(args.n_trials_per_set, len(valid_unlock)), replace=False)

        # ordinary moments: random rows from same episodes (with enough history), no unlock at that step
        candidates = []
        for ep in np.unique(ep_ids[unlock_sel]):
            ep_start = int(ep_start_row[ep])
            ep_end = int(ep_start_row[ep+1])
            for g in range(ep_start + args.window - 1, ep_end):
                if aj[g, concept_idx]:
                    continue
                candidates.append(g)
        ordinary_sel = rng.choice(candidates, size=min(args.n_trials_per_set, len(candidates)), replace=False)

        for moment_type, sel in [("unlock", unlock_sel), ("ordinary", ordinary_sel)]:
            for g in sel:
                ep = int(ep_ids[g])
                g_local = int(g - ep_start_row[ep])

                # window: steps [g_local - window + 1 .. g_local] in episode
                lo = g_local - args.window + 1
                hi = g_local + 1                              # exclusive
                # ep token rows
                ep_lat = tokens[ep_start_row[ep]:ep_start_row[ep+1]]   # (T, K) int16
                ep_act = actions[ep_start_row[ep]:ep_start_row[ep+1]]  # (T,) int64
                # episode's obs: T+1 entries
                ep_obs = all_obs[obs_starts[ep]:obs_starts[ep+1]]      # (T+1, 3, 64, 64)

                lat_chunk = torch.from_numpy(ep_lat[lo:hi].astype(np.int64)).to(device)   # (window, K)
                act_chunk = torch.from_numpy(ep_act[lo:hi]).to(device)                   # (window,)
                # obs chunk: window+1 frames so frame_cnn can run for all window steps
                obs_chunk = torch.from_numpy(ep_obs[lo:hi+1]).to(device).float().div(255)
                # We feed `obs[lo:hi]` (window frames) — frame_cnn only sees current frame per step
                obs_in = obs_chunk[:args.window].unsqueeze(0)     # (1, window, 3, 64, 64)
                act_in = act_chunk.unsqueeze(0)                   # (1, window)
                lat_in = lat_chunk.unsqueeze(0)                   # (1, window, K)

                target_step = args.window - 1                     # last step in window
                ablation_step = target_step - args.gap            # gap steps before target

                # (a) baseline
                wm_out = forward_with_optional_ablation(
                    agent, obs_in, act_in, lat_in,
                    target_block_idx=args.target_block, direction=None,
                    ablation_step=-1, strength=0.0)
                logp_base = log_prob_codes_at_step(wm_out, lat_chunk, target_step)

                # (b) feature ablation
                wm_out = forward_with_optional_ablation(
                    agent, obs_in, act_in, lat_in,
                    target_block_idx=args.target_block, direction=direction,
                    ablation_step=ablation_step, strength=args.strength)
                logp_feat = log_prob_codes_at_step(wm_out, lat_chunk, target_step)

                # (c) random direction (with same norm distribution as feature direction)
                rand_dir = torch.randn_like(direction)
                wm_out = forward_with_optional_ablation(
                    agent, obs_in, act_in, lat_in,
                    target_block_idx=args.target_block, direction=rand_dir,
                    ablation_step=ablation_step, strength=args.strength)
                logp_rand = log_prob_codes_at_step(wm_out, lat_chunk, target_step)

                trials.append(dict(
                    concept=target["concept"],
                    sae_feat_id=target["sae_feat_id"],
                    moment_type=moment_type,
                    episode=ep,
                    global_step=int(g),
                    ep_step=g_local,
                    logp_base=logp_base,
                    logp_feat=logp_feat,
                    logp_rand=logp_rand,
                    delta_feat=logp_base - logp_feat,    # >0 if ablation hurts prediction
                    delta_rand=logp_base - logp_rand,
                ))

        elapsed = time.time() - t0
        rate = (ti + 1) / max(elapsed, 1e-3)
        eta_min = (len(targets) - ti - 1) / max(rate, 1e-6) / 60
        print(f"  done concept {ti+1}/{len(targets)}: {target['concept']}  "
              f"({len([x for x in trials if x['concept']==target['concept']])} trials)  "
              f"eta {eta_min:.1f} min", flush=True)

    # ----- aggregate ---------------------------------------------------
    print(f"\n{len(trials)} trials in {(time.time()-t0)/60:.1f} min", flush=True)
    by = defaultdict(list)
    for t in trials:
        by[(t["concept"], t["moment_type"], "feat")].append(t["delta_feat"])
        by[(t["concept"], t["moment_type"], "rand")].append(t["delta_rand"])

    def bootstrap_ci(arr, B=500):
        rng2 = np.random.default_rng(0)
        means = [rng2.choice(arr, size=len(arr), replace=True).mean() for _ in range(B)]
        return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    summary = []
    for (concept, mtype, cond), vals in by.items():
        arr = np.asarray(vals, dtype=float)
        lo, hi = bootstrap_ci(arr) if arr.size > 1 else (float(arr.mean()), float(arr.mean()))
        summary.append(dict(concept=concept, moment_type=mtype, condition=cond,
                            n=int(arr.size), mean=float(arr.mean()),
                            std=float(arr.std()), ci_lo=lo, ci_hi=hi))

    (args.out / "trials.json").write_text(json.dumps(trials, indent=2))
    (args.out / "causal_effects.json").write_text(json.dumps(summary, indent=2))
    print(f"saved {args.out/'trials.json'} + causal_effects.json", flush=True)

    # ----- render ------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    concepts = sorted({s["concept"] for s in summary})
    n_c = len(concepts)
    width = 0.18

    fig, ax = plt.subplots(figsize=(1.0 + 0.95 * n_c, 5.5))
    x = np.arange(n_c)
    for j, (mtype, cond, label, color) in enumerate([
        ("unlock",  "feat", "unlock  · SAE-feat ablated", "#d62728"),
        ("ordinary","feat", "ordinary · SAE-feat ablated", "#9ecae1"),
        ("unlock",  "rand", "unlock  · random direction",  "#999999"),
        ("ordinary","rand", "ordinary · random direction", "#dddddd"),
    ]):
        ys, ylo, yhi = [], [], []
        for c in concepts:
            rec = next((s for s in summary if s["concept"]==c and s["moment_type"]==mtype and s["condition"]==cond), None)
            ys.append(rec["mean"] if rec else 0)
            ylo.append((rec["mean"] - rec["ci_lo"]) if rec else 0)
            yhi.append((rec["ci_hi"] - rec["mean"]) if rec else 0)
        ax.bar(x + (j - 1.5) * width, ys, width, label=label, color=color,
               yerr=[ylo, yhi], capsize=2, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(concepts, rotation=25, ha="right")
    ax.set_ylabel("Δ log P(real codes at target step)  =  baseline − ablated")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title(f"Causal effect of SAE-feature ablation at step T−{args.gap}\n"
                 f"on prediction of real codes at step T  (window={args.window}, block {args.target_block})")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=140); plt.close(fig)
    img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    parts = [
        "<!doctype html><meta charset=utf-8><title>causal feature ablation</title>",
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
        "<h1>Δ-IRIS causal feature-ablation experiment</h1>",
        f"<div class=dim>{len(trials)} trials · ablation at step T−{args.gap}, "
        f"target at step T (end of {args.window}-step window) · "
        f"hook at WM block {args.target_block} (after SAE-training layer) · "
        f"strength={args.strength}</div>",
        f"<img src='data:image/png;base64,{img_b64}'>",
        "<h2>Summary table</h2>",
        "<table><tr><th class=label>concept</th><th>moment</th><th>condition</th>"
        "<th>n</th><th>mean Δ logP</th><th>95% CI</th></tr>"
    ]
    for s in sorted(summary, key=lambda x: (x["concept"], x["moment_type"], x["condition"])):
        hi_class = "hi" if (s["moment_type"]=="unlock" and s["condition"]=="feat" and s["mean"] > 0.5) else ""
        parts.append(f"<tr><td class=label>{s['concept']}</td>"
                     f"<td>{s['moment_type']}</td><td>{s['condition']}</td>"
                     f"<td>{s['n']}</td>"
                     f"<td class='{hi_class}'>{s['mean']:+.3f}</td>"
                     f"<td>[{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}]</td></tr>")
    parts.append("</table>")
    (args.out / "intervention.html").write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {args.out / 'intervention.html'}", flush=True)


if __name__ == "__main__":
    main()
