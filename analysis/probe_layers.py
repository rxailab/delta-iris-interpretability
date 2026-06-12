"""Layer-wise linear probes for Δ-IRIS's tokenizer + world-model representations.

For each "representation" R(t) and each target label y(t), train a tiny linear
probe and report performance. The representations are:

  0  raw-codes      4 × 64-dim codebook embeddings   (tokenizer output, no context)
  1  WM-pre         frame_emb + act_emb + latents (the input to the transformer)
  2  WM-block-1     after transformer block 1
  3  WM-block-2     after transformer block 2
  4  WM-block-3     after transformer block 3 (= final hidden state)

For each representation we use the activation at the *summary position* of
timestep t — the last latent token (index 5 within the block) — averaged over
the 4 latent positions to reduce variance.

Labels probed:
  - action_taken   (17-way softmax)
  - reward_now     (binary, reward>0 at step t)
  - reward_soon    (binary, any reward in next K=5 steps)
  - ach_just[i]    (binary, achievement i flipped 0→1 at step t)
  - ach_cum[i]     (binary, achievement i currently held)

Outputs:
  - probe_metrics.json   : per (rep, label) {acc, auroc, n_pos, n_train, n_test}
  - probe_report.html    : heatmap + per-concept breakdown
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler


@dataclass
class Rep:
    name: str
    feats: np.ndarray              # (N, D) float32


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--rollouts", required=True, type=Path)
    ap.add_argument("--obs", required=True, type=Path,
                    help="path to obs.npz from rollout_with_info.py --save-obs")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="if >0, subsample to this many steps")
    ap.add_argument("--reward-soon-k", type=int, default=5)
    ap.add_argument("--batch-windows", type=int, default=8,
                    help="WM context windows processed in parallel")
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
    print(f"WM: {num_layers} transformer blocks × embed_dim={embed_dim}, "
          f"tokens_per_block={tokens_per_block}, max_blocks={max_blocks}")

    # ----- load rollouts + obs --------------------------------------------
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
    print(f"loaded {N} steps from {n_ep} episodes, {len(ach_names)} achievements")

    # ----- precompute per-step labels -------------------------------------
    reward_now = (rewards > 0).astype(np.int8)
    reward_soon = np.zeros(N, dtype=np.int8)
    # within each episode, set reward_soon[t]=1 if any reward in (t, t+K] from same episode
    cur = 0
    for ep_len in ep_lens:
        rew = rewards[cur:cur+ep_len]
        for t in range(ep_len):
            stop = min(t + 1 + args.reward_soon_k, ep_len)
            reward_soon[cur + t] = int((rew[t+1:stop] > 0).any())
        cur += ep_len

    # ----- extract representations via forward hooks ----------------------
    # Captures will be lists, one tensor per layer, shape (B, T*tokens_per_block, embed_dim).
    capture: dict[int, torch.Tensor] = {}
    hooks = []
    for i, block in enumerate(wm.transformer.blocks):
        def make_hook(i=i):
            def h(_module, _inp, output): capture[i] = output.detach()
            return h
        hooks.append(block.register_forward_hook(make_hook()))

    # Naming note: "wm_input" is actually the output of WM transformer block 0 (the hook
    # captures block outputs, not inputs). The two new pre-transformer reps below
    # disambiguate that and test the HUD-inventory hypothesis for ach_cum[*]:
    #   - frame_emb       : frame_cnn(current frame), 512-d, NO transformer
    #   - wm_latents_emb  : wm.latents_emb mean over the 4 codes, 512-d, NO attention/frame
    reps_to_collect = ["raw_codes", "frame_emb", "wm_latents_emb", "wm_input"] \
                      + [f"wm_block_{i+1}" for i in range(num_layers)]
    rep_arrays = {name: [] for name in reps_to_collect}
    label_index = []                  # (global_step_idx,) corresponding to each row in rep_arrays

    # ----- iterate episodes, chunk each into max_blocks-step windows ------
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
            # build WM input for one window: obs[start..end+1], act[start..end], latents[start..end]
            obs_t = torch.from_numpy(ep_obs[chunk_start:chunk_end+1]).to(device).float().div(255).unsqueeze(0)  # (1, chunk_len+1, 3,64,64)
            act_t = torch.from_numpy(ep_act[chunk_start:chunk_end]).to(device).unsqueeze(0)                     # (1, chunk_len)
            lat_t = torch.from_numpy(ep_tok[chunk_start:chunk_end].astype(np.int64)).to(device).unsqueeze(0)    # (1, chunk_len, K)

            with torch.no_grad():
                # build the sequence the same way WorldModel.compute_loss does, but for inference
                # frames: full sequence including post-step obs of last timestep so embedding matches
                frames_emb = wm.frame_cnn(obs_t[:, :chunk_len])                       # (1, chunk_len, 1, embed)
                act_emb = wm.act_emb(act_t).unsqueeze(2)                                 # (1, chunk_len, 1, embed)
                lat_emb = wm.latents_emb(lat_t)                                          # (1, chunk_len, K, embed)
                sequence = torch.cat((frames_emb, act_emb, lat_emb), dim=2)              # (1, chunk_len, tokens_per_block, embed)
                seq_flat = rearrange(sequence, 'b t p e -> b (t p) e')

                # ---- raw_codes baseline: just the 4 codebook embeddings -----
                cb_emb = tk.quantizer.codebook[lat_t.squeeze(0).reshape(-1)].reshape(chunk_len, K, -1)  # (chunk_len, K, codebook_dim)
                raw_codes_feat = cb_emb.mean(dim=1).cpu().numpy()                       # (chunk_len, codebook_dim)
                # ---- pre-transformer reps (HUD test) ------------------------
                # frame_emb: CNN over the current frame only (no action, no latents, no attention)
                frame_emb_feat = frames_emb.squeeze(0).squeeze(1).cpu().numpy()         # (chunk_len, embed)
                # wm_latents_emb: WM's own lookup over the 4 latent codes, mean-pooled
                wm_lat_emb_feat = lat_emb.squeeze(0).mean(dim=1).cpu().numpy()          # (chunk_len, embed)

                # ---- run WM forward, capturing hidden states ------------------
                capture.clear()
                wm_out = wm.transformer(seq_flat, use_kv_cache=False)                    # (1, chunk_len*tpb, embed)
                # Note: hook captures each block's pre-LN output. wm_out is the FINAL post-LN output.
                # We treat "wm_block_{num_layers}" as the post-final-LN representation.
                capture[num_layers] = wm_out.detach()

            # For each timestep t in the chunk: summary position = average over latent positions (2..5)
            for layer_idx in range(num_layers + 1):
                hs = capture[layer_idx][0].reshape(chunk_len, tokens_per_block, embed_dim)  # (T, 6, E)
                # use average of the 4 latent positions
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

    # Optional subsample for speed
    if args.max_steps > 0 and label_index.shape[0] > args.max_steps:
        sel = np.random.default_rng(0).choice(label_index.shape[0], args.max_steps, replace=False)
        sel.sort()
        label_index = label_index[sel]
        reps = {k: v[sel] for k, v in reps.items()}
    n_samples = label_index.shape[0]
    print(f"\ngathered {n_samples} samples × {len(reps)} representations")

    # ----- assemble labels aligned with label_index -----------------------
    sel_ep = ep_ids[label_index]
    sel_action = actions[label_index]
    sel_reward_now = reward_now[label_index]
    sel_reward_soon = reward_soon[label_index]
    sel_aj = aj[label_index]
    sel_acum = acum[label_index]

    # ----- episode-disjoint 80/20 split -----------------------------------
    rng = np.random.default_rng(0)
    unique_eps = np.unique(sel_ep)
    rng.shuffle(unique_eps)
    n_train_ep = int(len(unique_eps) * 0.8)
    train_eps = set(unique_eps[:n_train_ep].tolist())
    is_train = np.isin(sel_ep, list(train_eps))
    print(f"split: {is_train.sum()} train ({n_train_ep} eps) / {(~is_train).sum()} test ({len(unique_eps)-n_train_ep} eps)")

    # ----- run probes -----------------------------------------------------
    results: list[dict] = []

    def probe_binary(name, y, rep_name, feats):
        ytr, yte = y[is_train], y[~is_train]
        n_pos = int(yte.sum())
        if n_pos < 5 or n_pos == yte.shape[0]:
            return dict(rep=rep_name, label=name, acc=None, auroc=None, f1=None,
                        n_pos_test=n_pos, n_train=int(is_train.sum()), n_test=int((~is_train).sum()),
                        skipped="too few positives in test")
        Xtr, Xte = feats[is_train], feats[~is_train]
        # standardise per-feature using train stats
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
        clf = LogisticRegression(max_iter=200, class_weight="balanced", C=1.0, solver="lbfgs")
        clf.fit(Xtr, ytr)
        score = clf.decision_function(Xte)
        try:
            auroc = float(roc_auc_score(yte, score))
        except Exception:
            auroc = None
        pred = clf.predict(Xte)
        return dict(rep=rep_name, label=name,
                    acc=float(accuracy_score(yte, pred)),
                    f1=float(f1_score(yte, pred, zero_division=0)),
                    auroc=auroc,
                    n_pos_test=n_pos, n_train=int(is_train.sum()), n_test=int((~is_train).sum()))

    def probe_multiclass(name, y, rep_name, feats, n_classes):
        Xtr, Xte = feats[is_train], feats[~is_train]
        ytr, yte = y[is_train], y[~is_train]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
        clf = LogisticRegression(max_iter=200, C=1.0, solver="lbfgs", multi_class="multinomial")
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        chance = max(np.bincount(yte) / yte.shape[0]) if yte.size else 0
        return dict(rep=rep_name, label=name,
                    acc=float(accuracy_score(yte, pred)),
                    macro_f1=float(f1_score(yte, pred, average="macro", zero_division=0)),
                    chance=float(chance),
                    n_train=int(is_train.sum()), n_test=int((~is_train).sum()))

    for rep_name, feats in reps.items():
        print(f"\n--- probing rep: {rep_name}  (D={feats.shape[1]}) ---", flush=True)
        # multi-class: action
        results.append(probe_multiclass("action_taken", sel_action.astype(np.int64), rep_name, feats, 17))
        # binary: reward
        results.append(probe_binary("reward_now", sel_reward_now.astype(np.int64), rep_name, feats))
        results.append(probe_binary(f"reward_in_next_{args.reward_soon_k}", sel_reward_soon.astype(np.int64), rep_name, feats))
        # achievement_just
        for i, an in enumerate(ach_names):
            results.append(probe_binary(f"ach_just[{an}]", sel_aj[:, i].astype(np.int64), rep_name, feats))
        # achievement_cumulative
        for i, an in enumerate(ach_names):
            results.append(probe_binary(f"ach_cum[{an}]", sel_acum[:, i].astype(np.int64), rep_name, feats))

    (args.out / "probe_metrics.json").write_text(json.dumps(results, indent=2))
    meta = dict(run=str(run), rollouts=str(args.rollouts), obs=str(args.obs),
                n_samples=int(n_samples), n_episodes=int(n_ep),
                achievements=ach_names, representations=list(reps.keys()))
    (args.out / "probe_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {len(results)} probe results -> {args.out / 'probe_metrics.json'}")


if __name__ == "__main__":
    main()
