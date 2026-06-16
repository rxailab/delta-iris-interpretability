"""Bootstrap / cross-validated confidence intervals on every probe + CAV AUROC.

This re-runs the layer-wise probing of `probe_layers.py` (representation
extraction and logistic-regression probe fitting are reused VERBATIM) but adds
uncertainty quantification on the held-out AUROC of each (representation, target):

  - bootstrap CI: resample the held-out TEST set B times (default 1000),
    recompute AUROC on each resample, report mean +/- [2.5%, 97.5%] percentiles.
  - 5-fold episode-disjoint CV: refit the probe on 5 episode-disjoint train/test
    splits, report mean +/- (1.96 * SE) across folds.

CRITICAL deliverable (the "HUD-shortcut" claim):
  For the 18 cumulative-state concepts (ach_cum[*] with enough test positives),
  compare frame_emb AUROC against each transformer block (wm_block_1..3) with a
  PAIRED bootstrap CI on the difference frame_emb - block. The paired bootstrap
  reuses the SAME resampled test indices for both representations so the CI is on
  the difference (often < 1e-3), letting "frame_emb matches-or-exceeds every block
  on 16/18" be stated with uncertainty rather than fragile point estimates.

Outputs under --out:
  probe_ci.json    : per (rep, target) {mean_auroc, boot_lo, boot_hi,
                     cv_mean, cv_lo, cv_hi, n_pos_test, ...}
  hud_margin.json  : per cumulative concept, frame_emb vs each block with paired
                     bootstrap CI on the difference, plus a summary count of how
                     many concepts frame_emb matches-or-exceeds (CI-aware).

Reuses (copied to keep this script self-contained, matching repo style):
  - probe_layers.py representation extraction (frame_emb / wm_latents_emb /
    wm_input / wm_block_{i}, summary = mean over the 4 latent positions 2..2+K)
  - probe_layers.py probe fitting (standardise on train stats, balanced LogReg,
    decision_function -> roc_auc_score)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# --------------------------------------------------------------------------- #
#  bootstrap helpers                                                          #
# --------------------------------------------------------------------------- #
def auroc_safe(y, score):
    """roc_auc_score that returns None when undefined (single-class subset)."""
    try:
        if y.min() == y.max():
            return None
        return float(roc_auc_score(y, score))
    except Exception:
        return None


def bootstrap_auroc_ci(y, score, B=1000, seed=0):
    """Bootstrap CI on AUROC by resampling the test set with replacement.

    Returns (mean, lo, hi, point) where point is the AUROC on the full test set.
    """
    point = auroc_safe(y, score)
    n = y.shape[0]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        au = auroc_safe(y[idx], score[idx])
        if au is not None:
            vals.append(au)
    if not vals:
        return (point, point, point, point)
    vals = np.asarray(vals, dtype=float)
    return (float(vals.mean()), float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)), point)


def paired_bootstrap_diff_ci(y, score_a, score_b, B=1000, seed=0):
    """Paired bootstrap CI on AUROC(a) - AUROC(b) over the SAME test resamples.

    `score_a`, `score_b` are decision scores for the two representations on the
    identical held-out rows `y`. Resampling the same indices for both makes the
    CI tight on the (often tiny) difference.
    Returns (mean_diff, lo, hi, point_diff).
    """
    n = y.shape[0]
    point_a = auroc_safe(y, score_a)
    point_b = auroc_safe(y, score_b)
    point_diff = (point_a - point_b) if (point_a is not None and point_b is not None) else None
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        ya = y[idx]
        au_a = auroc_safe(ya, score_a[idx])
        au_b = auroc_safe(ya, score_b[idx])
        if au_a is not None and au_b is not None:
            diffs.append(au_a - au_b)
    if not diffs:
        return (point_diff, point_diff, point_diff, point_diff)
    diffs = np.asarray(diffs, dtype=float)
    return (float(diffs.mean()), float(np.percentile(diffs, 2.5)),
            float(np.percentile(diffs, 97.5)), point_diff)


# --------------------------------------------------------------------------- #
#  probe fitting (reused verbatim from probe_layers.py)                       #
# --------------------------------------------------------------------------- #
def fit_probe_scores(Xtr, ytr, Xte):
    """Standardise on train stats, fit balanced LogReg, return test decision scores.

    Mirrors probe_layers.py probe_binary EXACTLY (same standardisation, same clf).
    """
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    clf = LogisticRegression(max_iter=200, class_weight="balanced", C=1.0, solver="lbfgs")
    clf.fit(Xtr, ytr)
    return clf.decision_function(Xte)


# --------------------------------------------------------------------------- #
#  main                                                                       #
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--rollouts", required=True, type=Path)
    ap.add_argument("--obs", required=True, type=Path,
                    help="path to obs.npz from rollout_with_info.py --save-obs")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="if >0, subsample to this many steps (speed)")
    ap.add_argument("--reward-soon-k", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=1000,
                    help="bootstrap resamples for AUROC CIs")
    ap.add_argument("--n-folds", type=int, default=5,
                    help="episode-disjoint CV folds")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    run = args.run.resolve()
    sys.path.insert(0, str(run / "src"))
    args.out.mkdir(parents=True, exist_ok=True)

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    cfg = OmegaConf.load(run / ".hydra" / "config.yaml")
    if cfg.params.tokenizer.num_actions is None: cfg.params.tokenizer.num_actions = 17
    if cfg.params.world_model.num_actions is None: cfg.params.world_model.num_actions = 17
    if cfg.params.actor_critic.model.num_actions is None: cfg.params.actor_critic.model.num_actions = 17
    OmegaConf.resolve(cfg)

    device = torch.device(args.device)
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
               load_tokenizer=True, load_world_model=True, load_actor_critic=True,
               strict=False)
    wm = agent.world_model
    tk = agent.tokenizer
    num_layers = wm.config.transformer_config.num_layers
    embed_dim = wm.config.transformer_config.embed_dim
    tokens_per_block = wm.config.transformer_config.tokens_per_block      # 6
    max_blocks = wm.config.transformer_config.max_blocks                  # 21
    print(f"WM: {num_layers} transformer blocks x embed_dim={embed_dim}, "
          f"tokens_per_block={tokens_per_block}, max_blocks={max_blocks}", flush=True)

    # ----- load rollouts + obs (verbatim from probe_layers.py) ------------
    r = np.load(args.rollouts, allow_pickle=True)
    obs_npz = np.load(args.obs)
    obs = obs_npz["obs"]                            # (sum(T+1), 3, 64, 64) uint8
    obs_starts = obs_npz["episode_starts"]          # (n_episodes+1,) int64

    tokens   = r["tokens"]                           # (N, K)
    actions  = r["actions"].astype(np.int64)         # (N,)
    rewards  = r["rewards"]                          # (N,)
    aj       = r["ach_just_unlocked"]                # (N, 22)
    acum     = r["ach_cumulative"]                   # (N, 22)
    ep_ids   = r["episode_ids"].astype(np.int64)     # (N,)
    ep_lens  = r["episode_lengths"].astype(np.int64) # (N_ep,)
    ach_names = [str(x) for x in r["achievement_names"]]
    N, K = tokens.shape
    n_ep = ep_lens.shape[0]
    print(f"loaded {N} steps from {n_ep} episodes, {len(ach_names)} achievements", flush=True)
    assert obs_starts.shape[0] == n_ep + 1, \
        f"obs_starts {obs_starts.shape} should be n_ep+1={n_ep+1}"
    assert aj.shape[1] == len(ach_names) and acum.shape[1] == len(ach_names)
    # The summary-position slice below is hs[:, 2:2+K]; for it to cover exactly the
    # 4 latent positions of a [frame, action, K latents] block we need K == tpb-2.
    assert K == tokens_per_block - 2, \
        f"K={K} but tokens_per_block={tokens_per_block}; latent slice 2:2+K would be wrong"
    assert int(ep_lens.sum()) == N, f"sum(ep_lens)={int(ep_lens.sum())} != N={N}"

    # ----- precompute per-step labels (verbatim) --------------------------
    reward_now = (rewards > 0).astype(np.int8)
    reward_soon = np.zeros(N, dtype=np.int8)
    cur = 0
    for ep_len in ep_lens:
        rew = rewards[cur:cur+ep_len]
        for t in range(ep_len):
            stop = min(t + 1 + args.reward_soon_k, ep_len)
            reward_soon[cur + t] = int((rew[t+1:stop] > 0).any())
        cur += ep_len

    # ----- extract representations via forward hooks (verbatim) -----------
    capture: dict[int, torch.Tensor] = {}
    hooks = []
    for i, block in enumerate(wm.transformer.blocks):
        def make_hook(i=i):
            def h(_module, _inp, output): capture[i] = output.detach()
            return h
        hooks.append(block.register_forward_hook(make_hook()))

    reps_to_collect = ["raw_codes", "frame_emb", "wm_latents_emb", "wm_input"] \
                      + [f"wm_block_{i+1}" for i in range(num_layers)]
    rep_arrays = {name: [] for name in reps_to_collect}
    label_index = []

    t0 = time.time()
    steps_done = 0
    for ep in range(n_ep):
        T = int(ep_lens[ep])
        ep_obs = obs[obs_starts[ep]:obs_starts[ep+1]]    # (T+1, 3, 64, 64) uint8
        ep_act = actions[ep_ids == ep]                   # (T,)
        ep_tok = tokens[ep_ids == ep]                    # (T, K)
        ep_start_global = int(np.flatnonzero(ep_ids == ep)[0])

        for chunk_start in range(0, T, max_blocks):
            chunk_end = min(chunk_start + max_blocks, T)
            chunk_len = chunk_end - chunk_start
            obs_t = torch.from_numpy(ep_obs[chunk_start:chunk_end+1]).to(device).float().div(255).unsqueeze(0)
            act_t = torch.from_numpy(ep_act[chunk_start:chunk_end]).to(device).unsqueeze(0)
            lat_t = torch.from_numpy(ep_tok[chunk_start:chunk_end].astype(np.int64)).to(device).unsqueeze(0)

            with torch.no_grad():
                frames_emb = wm.frame_cnn(obs_t[:, :chunk_len])                       # (1, chunk_len, 1, embed)
                act_emb = wm.act_emb(act_t).unsqueeze(2)                                 # (1, chunk_len, 1, embed)
                lat_emb = wm.latents_emb(lat_t)                                          # (1, chunk_len, K, embed)
                sequence = torch.cat((frames_emb, act_emb, lat_emb), dim=2)              # (1, chunk_len, tpb, embed)
                seq_flat = rearrange(sequence, 'b t p e -> b (t p) e')

                cb_emb = tk.quantizer.codebook[lat_t.squeeze(0).reshape(-1)].reshape(chunk_len, K, -1)
                raw_codes_feat = cb_emb.mean(dim=1).cpu().numpy()                       # (chunk_len, codebook_dim)
                frame_emb_feat = frames_emb.squeeze(0).squeeze(1).cpu().numpy()         # (chunk_len, embed)
                wm_lat_emb_feat = lat_emb.squeeze(0).mean(dim=1).cpu().numpy()          # (chunk_len, embed)

                capture.clear()
                wm_out = wm.transformer(seq_flat, use_kv_cache=False)                    # (1, chunk_len*tpb, embed)
                capture[num_layers] = wm_out.detach()

            for layer_idx in range(num_layers + 1):
                hs = capture[layer_idx][0].reshape(chunk_len, tokens_per_block, embed_dim)
                summary = hs[:, 2:2+K].mean(dim=1).cpu().numpy()                         # (T, E)
                if layer_idx == 0:
                    rep_arrays["wm_input"].append(summary)
                else:
                    rep_arrays[f"wm_block_{layer_idx}"].append(summary)

            rep_arrays["raw_codes"].append(raw_codes_feat)
            rep_arrays["frame_emb"].append(frame_emb_feat)
            rep_arrays["wm_latents_emb"].append(wm_lat_emb_feat)
            label_index.extend(range(ep_start_global + chunk_start, ep_start_global + chunk_end))
            steps_done += chunk_len

        if (ep + 1) % max(1, n_ep // 20) == 0 or ep == n_ep - 1:
            elapsed = time.time() - t0
            print(f"  ep {ep+1}/{n_ep}  steps {steps_done}  {steps_done/max(elapsed,1):.0f} st/s  "
                  f"eta {(n_ep - ep - 1) * elapsed/(ep+1)/60:.1f} min", flush=True)

    for h in hooks: h.remove()
    label_index = np.array(label_index, dtype=np.int64)
    reps = {name: np.concatenate(arrs, axis=0).astype(np.float32) for name, arrs in rep_arrays.items()}

    # Optional subsample for speed (verbatim semantics)
    if args.max_steps > 0 and label_index.shape[0] > args.max_steps:
        sel = np.random.default_rng(0).choice(label_index.shape[0], args.max_steps, replace=False)
        sel.sort()
        label_index = label_index[sel]
        reps = {k: v[sel] for k, v in reps.items()}
    n_samples = label_index.shape[0]
    for k, v in reps.items():
        assert v.shape[0] == n_samples, f"rep {k} has {v.shape[0]} rows, expected {n_samples}"
    print(f"\ngathered {n_samples} samples x {len(reps)} representations", flush=True)

    # ----- assemble labels aligned with label_index (verbatim) ------------
    sel_ep = ep_ids[label_index]
    sel_action = actions[label_index]
    sel_reward_now = reward_now[label_index]
    sel_reward_soon = reward_soon[label_index]
    sel_aj = aj[label_index]
    sel_acum = acum[label_index]

    # ----- the canonical episode-disjoint 80/20 split (verbatim, seed 0) --
    rng = np.random.default_rng(0)
    unique_eps = np.unique(sel_ep)
    rng.shuffle(unique_eps)
    n_train_ep = int(len(unique_eps) * 0.8)
    train_eps = set(unique_eps[:n_train_ep].tolist())
    is_train = np.isin(sel_ep, list(train_eps))
    is_test = ~is_train
    print(f"primary split: {is_train.sum()} train ({n_train_ep} eps) / "
          f"{is_test.sum()} test ({len(unique_eps)-n_train_ep} eps)", flush=True)

    # ----- 5-fold episode-disjoint CV assignment --------------------------
    # Assign whole episodes to folds so test sets stay episode-disjoint.
    rng_cv = np.random.default_rng(1)
    shuffled_eps = unique_eps.copy()
    rng_cv.shuffle(shuffled_eps)
    fold_of_ep = {int(e): (i % args.n_folds) for i, e in enumerate(shuffled_eps)}
    sel_fold = np.array([fold_of_ep[int(e)] for e in sel_ep], dtype=np.int64)

    # ----- binary targets to evaluate (binary => AUROC defined) -----------
    # mirror probe_layers.py label set, minus the multiclass action probe.
    binary_targets: list[tuple[str, np.ndarray]] = []
    binary_targets.append(("reward_now", sel_reward_now.astype(np.int64)))
    binary_targets.append((f"reward_in_next_{args.reward_soon_k}", sel_reward_soon.astype(np.int64)))
    for i, an in enumerate(ach_names):
        binary_targets.append((f"ach_just[{an}]", sel_aj[:, i].astype(np.int64)))
    for i, an in enumerate(ach_names):
        binary_targets.append((f"ach_cum[{an}]", sel_acum[:, i].astype(np.int64)))

    rep_names = list(reps.keys())

    # =================================================================== #
    #  PASS 1: per (rep, target) bootstrap + CV AUROC CIs                  #
    #  Cache the held-out decision scores for paired analysis in pass 2.  #
    # =================================================================== #
    # scores_cache[(rep, target)] = (y_test, decision_scores) on PRIMARY split
    scores_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    results: list[dict] = []

    for tname, y in binary_targets:
        yte_primary = y[is_test]
        n_pos = int(yte_primary.sum())
        n_neg = int((yte_primary == 0).sum())
        defined = (n_pos >= 5) and (n_neg >= 5)

        for rep in rep_names:
            feats = reps[rep]
            rec = dict(rep=rep, label=tname,
                       n_pos_test=n_pos, n_test=int(is_test.sum()),
                       n_train=int(is_train.sum()))
            if not defined:
                rec.update(point_auroc=None, boot_mean=None, boot_lo=None, boot_hi=None,
                           cv_mean=None, cv_lo=None, cv_hi=None, cv_folds=None,
                           skipped="too few positives/negatives in test")
                results.append(rec)
                continue

            # --- primary-split probe + bootstrap CI on test AUROC ---
            score = fit_probe_scores(feats[is_train], y[is_train], feats[is_test])
            scores_cache[(rep, tname)] = (yte_primary.astype(np.int64), score.astype(np.float64))
            bmean, blo, bhi, bpoint = bootstrap_auroc_ci(
                yte_primary.astype(np.int64), score.astype(np.float64),
                B=args.n_boot, seed=0)

            # --- 5-fold episode-disjoint CV AUROC ---
            fold_aurocs = []
            for f in range(args.n_folds):
                te = (sel_fold == f)
                tr = ~te
                yf_te = y[te]
                if yf_te.sum() < 5 or (yf_te == 0).sum() < 5:
                    continue
                sf = fit_probe_scores(feats[tr], y[tr], feats[te])
                au = auroc_safe(yf_te.astype(np.int64), sf.astype(np.float64))
                if au is not None:
                    fold_aurocs.append(au)
            if fold_aurocs:
                fa = np.asarray(fold_aurocs, dtype=float)
                cv_mean = float(fa.mean())
                if fa.size > 1:
                    se = float(fa.std(ddof=1) / np.sqrt(fa.size))
                    cv_lo, cv_hi = cv_mean - 1.96 * se, cv_mean + 1.96 * se
                else:
                    cv_lo = cv_hi = cv_mean
            else:
                cv_mean = cv_lo = cv_hi = None

            rec.update(point_auroc=bpoint, boot_mean=bmean, boot_lo=blo, boot_hi=bhi,
                       cv_mean=cv_mean, cv_lo=cv_lo, cv_hi=cv_hi,
                       cv_folds=len(fold_aurocs))
            results.append(rec)

        print(f"  [{tname}] done (defined={defined}, n_pos_test={n_pos})", flush=True)

    (args.out / "probe_ci.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {len(results)} probe-CI records -> {args.out / 'probe_ci.json'}", flush=True)

    # =================================================================== #
    #  PASS 2: HUD-shortcut margins.                                       #
    #  For each cumulative concept with a defined AUROC, paired bootstrap  #
    #  CI on frame_emb - block for each transformer block.                #
    # =================================================================== #
    block_reps = [f"wm_block_{i+1}" for i in range(num_layers)]
    hud_records = []
    cum_labels = [f"ach_cum[{an}]" for an in ach_names]

    n_concepts_defined = 0
    n_frame_ge_all_point = 0          # point-estimate: frame_emb >= every block
    n_frame_ge_all_ci = 0            # CI-aware: frame_emb not significantly below ANY block
                                      # (i.e. for every block, diff CI upper bound >= 0)
    for lab in cum_labels:
        if ("frame_emb", lab) not in scores_cache:
            continue  # undefined target (too few test positives) -> not one of the 18
        y_fe, s_fe = scores_cache[("frame_emb", lab)]
        per_block = []
        all_point_ge = True
        all_ci_not_below = True
        for brep in block_reps:
            if (brep, lab) not in scores_cache:
                continue
            y_bl, s_bl = scores_cache[(brep, lab)]
            # paired -> same test rows / same y
            assert np.array_equal(y_fe, y_bl), \
                f"primary-split y mismatch for {lab}: {brep} vs frame_emb"
            mdiff, dlo, dhi, dpoint = paired_bootstrap_diff_ci(
                y_fe, s_fe, s_bl, B=args.n_boot, seed=0)
            fe_point = auroc_safe(y_fe, s_fe)
            bl_point = auroc_safe(y_bl, s_bl)
            per_block.append(dict(
                block=brep,
                frame_emb_auroc=fe_point,
                block_auroc=bl_point,
                diff_point=dpoint,                 # frame_emb - block
                diff_boot_mean=mdiff,
                diff_ci_lo=dlo,
                diff_ci_hi=dhi,
                frame_emb_wins_point=bool(dpoint is not None and dpoint >= 0),
                frame_emb_not_below_ci=bool(dhi is not None and dhi >= 0),
            ))
            if not (dpoint is not None and dpoint >= 0):
                all_point_ge = False
            if not (dhi is not None and dhi >= 0):
                all_ci_not_below = False
        if not per_block:
            continue
        n_concepts_defined += 1
        if all_point_ge:
            n_frame_ge_all_point += 1
        if all_ci_not_below:
            n_frame_ge_all_ci += 1
        hud_records.append(dict(
            concept=lab,
            n_pos_test=int(y_fe.sum()),
            n_test=int(y_fe.shape[0]),
            frame_emb_matches_or_exceeds_all_point=all_point_ge,
            frame_emb_not_significantly_below_all_ci=all_ci_not_below,
            per_block=per_block,
        ))

    hud = dict(
        n_cumulative_concepts=n_concepts_defined,
        n_blocks=len(block_reps),
        n_boot=args.n_boot,
        claim_point=f"frame_emb matches-or-exceeds every block on "
                    f"{n_frame_ge_all_point}/{n_concepts_defined} cumulative concepts "
                    f"(point estimate)",
        claim_ci=f"frame_emb is not significantly below ANY block (paired-bootstrap "
                 f"diff CI upper bound >= 0) on {n_frame_ge_all_ci}/{n_concepts_defined} "
                 f"cumulative concepts",
        concepts=hud_records,
    )
    (args.out / "hud_margin.json").write_text(json.dumps(hud, indent=2))
    print(f"\nHUD-shortcut summary:", flush=True)
    print(f"  cumulative concepts evaluated: {n_concepts_defined}", flush=True)
    print(f"  frame_emb >= every block (point):     {n_frame_ge_all_point}/{n_concepts_defined}", flush=True)
    print(f"  frame_emb not below any block (CI):   {n_frame_ge_all_ci}/{n_concepts_defined}", flush=True)
    print(f"wrote {args.out / 'hud_margin.json'}", flush=True)

    # ----- meta -----------------------------------------------------------
    meta = dict(run=str(run), rollouts=str(args.rollouts), obs=str(args.obs),
                n_samples=int(n_samples), n_episodes=int(n_ep),
                n_boot=int(args.n_boot), n_folds=int(args.n_folds),
                achievements=ach_names, representations=rep_names,
                primary_test_episodes=int(len(unique_eps) - n_train_ep))
    (args.out / "probe_ci_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {args.out / 'probe_ci_meta.json'}", flush=True)


if __name__ == "__main__":
    main()
