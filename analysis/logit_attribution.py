"""Direct-effect / logit attribution of SAE features and CAVs onto head_latents.

Question
--------
The ablation experiments (ablate_features.py, imagine_ablation.py) show that
*removing* a concept's SAE-feature / CAV direction from the WM residual stream
hurts the model's prediction of that concept's codes. But that aggregate causal
effect mixes a *direct* path (the direction feeds the head_latents logit of the
detector code through the remaining frozen layers) with an *indirect* path (the
direction changes attention/MLP processing of later positions). This script
isolates the DIRECT logit-attribution path.

Setup mirrors ablate_features.py exactly:
  - WM transformer block hooked = block 1 (0-indexed), the SAE/CAV training space.
  - The "block-1 latent-position activations" of a step are positions
        pos = step*tpb + 2 .. step*tpb + 6      (tpb = tokens_per_block = 6)
    i.e. the 4 latent slots of that step, AFTER block `target_block_idx`.
  - head_latents reads positions [1,2,3,4] of a block (block_mask =
    act_and_latents_but_last) and emits K=4 logit vectors per step:
        head-input position 1 (action slot)   -> predicts latent slot 0
        head-input position 2 (latent-0 slot)  -> predicts latent slot 1
        head-input position 3 (latent-1 slot)  -> predicts latent slot 2
        head-input position 4 (latent-2 slot)  -> predicts latent slot 3
    so detector slot s (s in 1..3) is read from head-input position s+1, which
    is exactly one of the 4 ablation positions (pos = step*tpb + 2 + (s-1)).
    Slot 0 is read from the action slot (position step*tpb+1) which is NOT one of
    the 4 ablation latent positions; for slot-0 detectors we therefore attribute
    over the same 4 latent positions but the head-input position used to read the
    logit is the action slot.

Method (stated approximation): GRADIENT x ACTIVATION direct logit attribution.
  For a unit direction u (SAE decoder column / CAV / random), let h_p be the
  block-1 output activation at latent position p of the attribution step, and let
  s_p = <h_p, u> be the scalar projection of h_p onto u. The model's detector-code
  logit  z = head_latents(...)[detector_slot, detector_code]  is a differentiable
  function of all the h_p (push the direction through the remaining FROZEN layers:
  block 2, final LayerNorm, head MLP). The first-order direct attribution of the
  u-component to the logit is

        attrib(u) = sum_p (dz/ds_p) * s_p
                  = sum_p ( <dz/dh_p, u> ) * ( <h_p, u> )

  i.e. gradient x activation of the detector-code logit w.r.t. the projection of
  h onto u, summed over the 4 latent positions of the attribution step. We report
  attrib on the DETECTOR code and on a CONTROL code (a random other code in the
  same slot), for matched-SAE vs matched-CAV vs random directions.

Outputs under --out:
  logit_attrib.json   per-(concept, direction, code-kind) summary + per-trial raw
  logit_attribution.html  bar chart + table
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


# ----------------------------- loaders (reused from ablate_features.py) -------
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


def detector_codes_from_mi(mi_table: Path, top_n: int = 2) -> dict[str, list[tuple[int, int]]]:
    """achievement -> top-N (slot, code) detectors by lift (reused from imagine_ablation.py)."""
    records = json.loads(mi_table.read_text())
    per: dict[str, list] = defaultdict(list)
    for r in records:
        per[r["achievement"]].append((r["lift"], r["slot"], r["code"]))
    out = {}
    for ach, rows in per.items():
        rows.sort(reverse=True)
        out[ach] = [(int(s), int(c)) for _, s, c in rows[:top_n]]
    return out


def pick_best_feat(features_json: Path, min_lift: float = 80.0) -> dict[str, dict]:
    """concept -> {feat, lift, p}  (strongest SAE feature per achievement)."""
    feats = json.loads(features_json.read_text())["features"]
    best: dict[str, dict] = {}
    for f in feats:
        a = f.get("best_ach")
        if not a or a["lift"] < min_lift: continue
        if f["density"] < 0.001: continue
        cur = best.get(a["name"])
        if cur is None or a["lift"] > cur["lift"]:
            best[a["name"]] = dict(feat=int(f["feature"]), lift=float(a["lift"]), p=float(a["p"]))
    return best


# ----------------------------- core: gradient x activation attribution -------
def attribute_logit(
    agent, obs_in, act_in, lat_in,
    target_block_idx: int,
    attrib_step: int,
    head_pos_in_block: int,        # which head-input position to read (1..4)
    code_id: int,                  # which code's logit to attribute
    directions: dict[str, torch.Tensor],
):
    """Gradient x activation direct logit attribution.

    Captures the block-`target_block_idx` OUTPUT activation h at the 4 latent
    positions of `attrib_step`, requires grad on those, runs the rest of the WM
    forward (remaining frozen blocks + final LN + head_latents), reads the logit
    of `code_id` at head-input position `head_pos_in_block` of `attrib_step`, and
    backprops to get dz/dh at the 4 latent positions. Then for each named unit
    direction u returns  sum_p (<dz/dh_p, u>) * (<h_p, u>).

    Returns: (attrib_dict {name: float}, logit_value float, proj_dict {name: (4,) np}).
    """
    wm = agent.world_model
    tpb = wm.config.transformer_config.tokens_per_block           # 6
    L = obs_in.shape[1]                                           # window length

    # ---- build the WM input sequence (same as ablate_features.forward_*) -----
    frames_emb = wm.frame_cnn(obs_in)                            # (1, L, 1, D)
    a_emb = wm.act_emb(act_in).unsqueeze(2)                      # (1, L, 1, D)
    l_emb = wm.latents_emb(lat_in)                               # (1, L, K, D)
    seq = torch.cat([frames_emb, a_emb, l_emb], dim=2)          # (1, L, tpb, D)
    seq_flat = rearrange(seq, "b t p d -> b (t p) d")          # (1, L*tpb, D)

    pos_start = attrib_step * tpb + 2     # first of the 4 latent positions
    pos_end = attrib_step * tpb + 6       # exclusive -> 4 positions [2,3,4,5]
    assert pos_end <= seq_flat.shape[1], \
        f"attrib positions {pos_start}:{pos_end} exceed seq len {seq_flat.shape[1]}"

    captured = {}

    def hook(_m, _inp, output):
        # `output` is the block-`target_block_idx` output (1, L*tpb, D).
        # Re-inject the 4 latent positions as a fresh leaf so we can grad w.r.t. them.
        h_leaf = output[:, pos_start:pos_end].detach().clone().requires_grad_(True)  # (1,4,D)
        captured["h_leaf"] = h_leaf
        new_out = output.clone()
        new_out[:, pos_start:pos_end] = h_leaf
        return new_out

    handle = wm.transformer.blocks[target_block_idx].register_forward_hook(hook)
    try:
        # Use the full WM forward so head_latents slicing/positions are identical
        # to training/eval. Need grad, so do NOT wrap in no_grad.
        wm_out = wm(seq_flat)
    finally:
        handle.remove()

    h_leaf = captured["h_leaf"]                                  # (1, 4, D), requires_grad

    # ---- read the detector-code logit at (attrib_step, head_pos_in_block) ----
    logits = wm_out.logits_latents                              # see log_prob_codes_at_step
    K = lat_in.shape[2]
    if logits.dim() == 4:
        # (B, T_pred, K, V): head emits K logit vectors per step
        per_slot_logits = logits[0, attrib_step]                # (K, V)
        # head-input positions map to slots 0..K-1 in order; head_pos_in_block
        # 1..K maps to slot index head_pos_in_block-1
        slot_logits = per_slot_logits[head_pos_in_block - 1]    # (V,)
    elif logits.dim() == 3:
        T = logits.size(1) // K
        logits_resh = logits.view(logits.size(0), T, -1, logits.size(-1))
        per_slot_logits = logits_resh[0, attrib_step]           # (K, V)
        slot_logits = per_slot_logits[head_pos_in_block - 1]
    else:
        raise RuntimeError(f"unexpected logits shape {logits.shape}")

    z = slot_logits[code_id]                                    # scalar logit
    logit_value = float(z.item())

    grad_h, = torch.autograd.grad(z, h_leaf, retain_graph=False)  # (1, 4, D)
    grad_h = grad_h[0].detach()                                 # (4, D)
    h_det = h_leaf[0].detach()                                  # (4, D)

    attrib, proj = {}, {}
    for name, raw in directions.items():
        u = F.normalize(raw.float(), dim=-1).to(h_det.device)   # (D,)
        s_p = (h_det @ u)                                       # (4,) projections
        g_p = (grad_h @ u)                                      # (4,) dz/ds_p
        contrib = (g_p * s_p)                                   # (4,) per-position attrib
        attrib[name] = float(contrib.sum().item())
        proj[name] = s_p.detach().cpu().numpy().tolist()
    return attrib, logit_value, proj


# ----------------------------- main -----------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--rollouts", required=True, type=Path)
    ap.add_argument("--obs", required=True, type=Path)
    ap.add_argument("--sae", required=True, type=Path)
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--cavs", required=True, type=Path)
    ap.add_argument("--mi-table", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-trials", type=int, default=40,
                    help="max unlock moments per concept")
    ap.add_argument("--window", type=int, default=21,
                    help="WM context window length (= max_blocks)")
    ap.add_argument("--gap", type=int, default=3,
                    help="distance between attribution step and end-of-window (matches ablate_features)")
    ap.add_argument("--target-block", type=int, default=1,
                    help="WM transformer block to hook (0-indexed); SAE/CAV space is block 1")
    ap.add_argument("--n-rand", type=int, default=5,
                    help="number of random control directions to average over")
    ap.add_argument("--max-concepts", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    run = args.run.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    print("loading agent...", flush=True)
    agent = load_agent(run, device)
    wm = agent.world_model
    tpb = wm.config.transformer_config.tokens_per_block
    num_layers = wm.config.transformer_config.num_layers
    embed_dim = wm.config.transformer_config.embed_dim
    V = wm.config.latent_vocab_size
    assert tpb == 6, f"expected tokens_per_block=6, got {tpb}"
    assert 0 <= args.target_block < num_layers, \
        f"target-block {args.target_block} out of range for {num_layers} layers"
    print(f"WM: tpb={tpb} num_layers={num_layers} embed_dim={embed_dim} vocab={V}", flush=True)

    decoder, b_dec, sae_cfg = load_sae(args.sae, device)
    assert decoder.shape[0] == embed_dim, \
        f"SAE decoder dim {decoder.shape[0]} != embed_dim {embed_dim}"
    print(f"SAE: D={decoder.shape[0]} n_features={decoder.shape[1]} layer={sae_cfg['layer']}", flush=True)

    # CAVs (same loading convention as imagine_ablation.cav_direction)
    cav_npz = np.load(args.cavs, allow_pickle=True)
    cav_names = [str(x) for x in cav_npz["concept_names"]]
    layer_key = f"wm_block_{args.target_block}"
    assert f"w_{layer_key}" in cav_npz.files, f"missing w_{layer_key} in cavs.npz"
    W_cav = cav_npz[f"w_{layer_key}"]
    sd_cav = cav_npz[f"std_{layer_key}"]
    assert W_cav.shape[1] == embed_dim and sd_cav.shape[0] == embed_dim

    def cav_direction(ach: str):
        name = f"just[{ach}]"
        if name not in cav_names: return None
        w = W_cav[cav_names.index(name)]
        if not np.any(w): return None
        d = w / sd_cav          # undo standardisation -> raw block-1 activation space
        return torch.from_numpy(d).to(device).float()

    best_feat = pick_best_feat(args.features, min_lift=80.0)
    detectors = detector_codes_from_mi(args.mi_table, top_n=2)

    # rollouts (identical block to ablate_features.py)
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
    ep_start_row = np.concatenate(([0], np.cumsum(ep_lens)))
    K = tokens.shape[1]
    print(f"rollouts: {tokens.shape[0]} steps · {ep_lens.shape[0]} episodes · K={K}", flush=True)

    # concepts: strongest SAE feature AND detector codes available
    targets = [c for c in best_feat if c in detectors]
    targets.sort(key=lambda c: -best_feat[c]["lift"])
    targets = targets[:args.max_concepts]
    print(f"target concepts ({len(targets)}):", flush=True)
    for c in targets:
        has_cav = cav_direction(c) is not None
        print(f"  {c:25s} feat#{best_feat[c]['feat']:>4d} lift={best_feat[c]['lift']:7.1f} "
              f"det={detectors[c]} cav={'Y' if has_cav else '-'}", flush=True)

    trials = []
    t0 = time.time()
    for ci, concept in enumerate(targets):
        a_idx = ach_names.index(concept)
        d_sae = decoder[:, best_feat[concept]["feat"]].clone()   # (D,)
        d_cav = cav_direction(concept)

        # use the TOP detector (slot, code) for this concept (highest MI lift)
        det_slot, det_code = detectors[concept][0]
        assert 0 <= det_slot < K, f"detector slot {det_slot} out of range K={K}"
        # head-input position that reads slot `det_slot`:
        #   slot s is predicted from head-input position s+1 (action slot=1 -> slot0,
        #   latent0 slot=2 -> slot1, ...). head_pos_in_block is 1-indexed within block.
        head_pos = det_slot + 1
        assert 1 <= head_pos <= 4, f"head_pos {head_pos} outside head block_mask [1..4]"

        # control code: a random OTHER code in the SAME slot (not the detector code)
        ctrl_code = int(rng.integers(0, V))
        while ctrl_code == det_code:
            ctrl_code = int(rng.integers(0, V))

        # unlock moments with enough history for a full window (same filter as ablate_features)
        unlock_global = np.where(aj[:, a_idx])[0]
        valid = []
        for g in unlock_global:
            ep = ep_ids[g]
            ep_step = g - ep_start_row[ep]
            if ep_step >= args.window - 1:
                valid.append(int(g))
        if len(valid) < 3:
            print(f"  {concept}: too few unlock moments with history, skipping", flush=True)
            continue
        sel = rng.choice(valid, size=min(args.n_trials, len(valid)), replace=False)

        for g in sel:
            ep = int(ep_ids[g])
            g_local = int(g - ep_start_row[ep])
            lo = g_local - args.window + 1
            hi = g_local + 1                                   # exclusive

            ep_lat = tokens[ep_start_row[ep]:ep_start_row[ep + 1]]   # (T, K) int16
            ep_act = actions[ep_start_row[ep]:ep_start_row[ep + 1]]  # (T,)
            ep_obs = all_obs[obs_starts[ep]:obs_starts[ep + 1]]      # (T+1, 3, 64, 64)

            lat_chunk = torch.from_numpy(ep_lat[lo:hi].astype(np.int64)).to(device)  # (window, K)
            act_chunk = torch.from_numpy(ep_act[lo:hi]).to(device)                  # (window,)
            obs_chunk = torch.from_numpy(ep_obs[lo:hi + 1]).to(device).float().div(255)
            obs_in = obs_chunk[:args.window].unsqueeze(0)     # (1, window, 3, 64, 64)
            act_in = act_chunk.unsqueeze(0)                   # (1, window)
            lat_in = lat_chunk.unsqueeze(0)                   # (1, window, K)
            assert obs_in.shape[1] == args.window and lat_in.shape[1] == args.window

            attrib_step = (args.window - 1) - args.gap        # same step the ablation hook hits

            # fresh random directions per trial (averaged over n_rand)
            rand_attrib_det = []
            rand_attrib_ctrl = []
            directions = {"sae": d_sae}
            if d_cav is not None:
                directions["cav"] = d_cav
            for ri in range(args.n_rand):
                directions[f"rand{ri}"] = torch.randn(embed_dim, device=device)

            # attribute on the DETECTOR code
            attrib_det, logit_det, _ = attribute_logit(
                agent, obs_in, act_in, lat_in,
                target_block_idx=args.target_block, attrib_step=attrib_step,
                head_pos_in_block=head_pos, code_id=det_code, directions=directions)
            # attribute on the CONTROL code (same positions / step / head pos)
            attrib_ctrl, logit_ctrl, _ = attribute_logit(
                agent, obs_in, act_in, lat_in,
                target_block_idx=args.target_block, attrib_step=attrib_step,
                head_pos_in_block=head_pos, code_id=ctrl_code, directions=directions)

            rand_det = float(np.mean([attrib_det[f"rand{ri}"] for ri in range(args.n_rand)]))
            rand_ctrl = float(np.mean([attrib_ctrl[f"rand{ri}"] for ri in range(args.n_rand)]))

            trials.append(dict(
                concept=concept, sae_feat_id=best_feat[concept]["feat"],
                det_slot=det_slot, det_code=det_code, ctrl_code=ctrl_code,
                episode=ep, global_step=int(g), ep_step=g_local,
                logit_det=logit_det, logit_ctrl=logit_ctrl,
                # direct logit attribution onto the DETECTOR code
                sae_det=attrib_det["sae"],
                cav_det=(attrib_det["cav"] if "cav" in attrib_det else None),
                rand_det=rand_det,
                # ... onto the CONTROL code
                sae_ctrl=attrib_ctrl["sae"],
                cav_ctrl=(attrib_ctrl["cav"] if "cav" in attrib_ctrl else None),
                rand_ctrl=rand_ctrl,
            ))

        n_done = len([t for t in trials if t["concept"] == concept])
        elapsed = time.time() - t0
        rate = (ci + 1) / max(elapsed, 1e-3)
        eta_min = (len(targets) - ci - 1) / max(rate, 1e-6) / 60
        print(f"  [{ci+1}/{len(targets)}] {concept}: {n_done} trials  "
              f"({elapsed/60:.1f} min, eta {eta_min:.1f} min)", flush=True)

    print(f"\n{len(trials)} trials in {(time.time()-t0)/60:.1f} min", flush=True)

    # ----- aggregate -------------------------------------------------------
    def boot_ci(arr, B=1000):
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0: return 0.0, 0.0
        if arr.size == 1: return float(arr[0]), float(arr[0])
        rng2 = np.random.default_rng(0)
        means = [rng2.choice(arr, size=arr.size, replace=True).mean() for _ in range(B)]
        return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    # one summary row per (concept, direction, code-kind)
    summary = []
    by_concept = defaultdict(list)
    for t in trials:
        by_concept[t["concept"]].append(t)
    for concept, sub in by_concept.items():
        for direction in ["sae", "cav", "rand"]:
            for kind in ["det", "ctrl"]:
                key = f"{direction}_{kind}"
                vals = [t[key] for t in sub if t.get(key) is not None]
                if not vals: continue
                arr = np.asarray(vals, dtype=float)
                lo, hi = boot_ci(arr)
                summary.append(dict(
                    concept=concept, direction=direction, code_kind=kind,
                    n=int(arr.size), mean=float(arr.mean()), std=float(arr.std()),
                    ci_lo=lo, ci_hi=hi))

    out_blob = dict(
        method="gradient_x_activation_direct_logit_attribution",
        description=("attrib(u) = sum over 4 block-%d latent positions of "
                     "(d logit / d <h,u>) * <h,u>; logit = head_latents logit of the "
                     "top-MI detector code for the concept, read at the head-input "
                     "position predicting that slot; control = random other code, same "
                     "slot/step/positions." % args.target_block),
        target_block=args.target_block, window=args.window, gap=args.gap,
        n_trials=len(trials), n_rand=args.n_rand,
        summary=summary, trials=trials,
    )
    (args.out / "logit_attrib.json").write_text(json.dumps(out_blob, indent=2))
    print(f"saved {args.out/'logit_attrib.json'}", flush=True)

    # ----- render ----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    concepts = sorted({s["concept"] for s in summary})
    n_c = len(concepts)
    if n_c == 0:
        print("no concepts to plot; done.", flush=True)
        return

    # plot: detector-code attribution per concept, sae vs cav vs rand
    fig, ax = plt.subplots(figsize=(1.5 + 0.95 * n_c, 5.5))
    x = np.arange(n_c)
    width = 0.25
    series = [
        ("sae", "#d62728", "SAE feature -> detector"),
        ("cav", "#9467bd", "CAV -> detector"),
        ("rand", "#999999", "random dir -> detector"),
    ]
    for j, (direction, color, label) in enumerate(series):
        ys, ylo, yhi = [], [], []
        for c in concepts:
            rec = next((s for s in summary if s["concept"] == c
                        and s["direction"] == direction and s["code_kind"] == "det"), None)
            if rec:
                ys.append(rec["mean"])
                ylo.append(max(0.0, rec["mean"] - rec["ci_lo"]))
                yhi.append(max(0.0, rec["ci_hi"] - rec["mean"]))
            else:
                ys.append(0); ylo.append(0); yhi.append(0)
        ax.bar(x + (j - 1) * width, ys, width, label=label, color=color,
               yerr=[ylo, yhi], capsize=2, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(concepts, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("direct logit attribution onto detector code")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title(f"Direct (grad x activation) logit attribution of concept directions\n"
                 f"onto head_latents detector-code logit (block {args.target_block}, "
                 f"step T-{args.gap}, window {args.window})", fontsize=10)
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=140); plt.close(fig)
    img = base64.b64encode(buf.getvalue()).decode("ascii")

    rows = []
    for s in sorted(summary, key=lambda x: (x["concept"], x["direction"], x["code_kind"])):
        hi_class = "hi" if (s["direction"] == "sae" and s["code_kind"] == "det"
                            and s["mean"] > 0) else ""
        rows.append(f"<tr><td class=label>{s['concept']}</td>"
                    f"<td>{s['direction']}</td><td>{s['code_kind']}</td>"
                    f"<td>{s['n']}</td>"
                    f"<td class='{hi_class}'>{s['mean']:+.4f}</td>"
                    f"<td>[{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}]</td></tr>")
    html = (
        "<!doctype html><meta charset=utf-8><title>logit attribution</title>"
        "<style>body{font:14px ui-monospace,Menlo,Consolas,monospace;background:#0d1117;"
        "color:#c9d1d9;padding:18px;}h1{color:#79c0ff;font-size:18px;margin:0 0 8px;}"
        "h2{color:#d2a8ff;font-size:15px;margin:14px 0 6px;}.dim{color:#8b949e;}"
        "img{max-width:100%;background:white;border:1px solid #21262d;border-radius:6px;}"
        "table{width:100%;border-collapse:collapse;font-size:12.5px;margin-top:8px;}"
        "th,td{padding:5px 8px;border-bottom:1px solid #21262d;text-align:right;}"
        "td.label{text-align:left;}th{color:#8b949e;text-transform:uppercase;font-size:11px;}"
        ".hi{color:#3fb950;font-weight:600;}</style>"
        "<h1>Direct logit attribution of SAE features & CAVs onto head_latents</h1>"
        f"<div class=dim>{len(trials)} trials at unlock moments · method = gradient x activation "
        f"of the detector-code logit w.r.t. the projection of block-{args.target_block} latent-"
        f"position activations onto each direction (sum over the 4 latent positions of step "
        f"T-{args.gap}). Detector = top-MI (slot,code) per concept; control = random other code, "
        f"same slot/positions. Positive = direction directly pushes UP the detector logit.</div>"
        f"<img src='data:image/png;base64,{img}'>"
        "<h2>Summary table</h2>"
        "<table><tr><th class=label>concept</th><th>direction</th><th>code</th>"
        "<th>n</th><th>mean attrib</th><th>95% CI</th></tr>"
        + "".join(rows) + "</table>")
    (args.out / "logit_attribution.html").write_text(html, encoding="utf-8")
    print(f"wrote {args.out/'logit_attribution.html'}", flush=True)


if __name__ == "__main__":
    main()
