"""Multi-direction / INLP subspace ablation: does a rank-r CAV subspace EVER
become causally load-bearing in the Delta-IRIS world model, or does the
single-direction asymmetry survive multi-direction removal?

THE HEADLINE MISSING EXPERIMENT.  We generalise the single-direction projection
ablation of analysis/ablate_features.py to remove an ORTHONORMAL BASIS Q (D x r)
from the residual stream:

    h  <-  h - Q (Q^T h)        at the 4 latent positions of the ablation step,
                                AFTER world-model transformer block 1.

For each of the ~8 dominant concepts we build THREE rank-r bases for
r in {1,2,4,8,12,16}:

  (i)  MATCHED-CAV subspace : Gram-Schmidt over the concept's own CAV plus its
       tech-tree-neighbour CAVs (the other concepts whose block-1 CAV directions
       are most cosine-similar to the target's, i.e. nearest in representation
       space — a data-driven proxy for Crafter's tech tree).  rank up to r.
  (ii) SAE-feature stack    : top-r SAE decoder columns for the concept, ranked
       by that concept's achievement lift in each feature's best_ach.  Orthonormalised.
  (iii)RANDOM subspace      : a random rank-r orthonormal subspace (control).

CAUSAL METRIC (per concept, basis, rank):
  single-step Delta-logP  =  logP_baseline - logP_ablated  of the realised
  latent codes at the unlock step T, with ablation applied at step T-gap,
  exactly as ablate_features.py does (window=21, gap default 3, block 1).
  Measured for both "unlock" moments and matched "ordinary" moments.
  Bootstrap 95% CIs.

PROBE-AUROC-RECOVERY CURVE (per concept, rank):
  Refit a logistic-regression probe on the block-1 summary activations AFTER
  projecting out the rank-r MATCHED-CAV subspace; report how the held-out AUROC
  for just[concept] decays as a function of r.  Shows how many directions must be
  removed before the concept becomes (linearly) undecodable.

Outputs under --out:
  subspace_effects.json   per (concept, basis, rank, moment_type): mean Delta-logP, CI, n
  auroc_recovery.json     per (concept, rank): held-out AUROC after projecting out
                          the rank-r CAV subspace (+ baseline AUROC at rank 0)
  subspace_meta.json      run config / concept list / ranks
  subspace.html           plots + tables
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


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
    """Return (decoder_weight (D, n_features), b_dec (D,), config)."""
    blob = torch.load(sae_path, map_location=device, weights_only=False)
    sd = blob["state_dict"]
    cfg = blob["config"]
    decoder = sd["decoder.weight"].to(device).float()        # (D, n_features)
    b_dec = sd["b_dec"].to(device).float()                    # (D,)
    return decoder, b_dec, cfg


# ----------------------------- basis construction ----------------------------
def gram_schmidt(cols: list[torch.Tensor], rank: int, eps: float = 1e-8) -> torch.Tensor | None:
    """Orthonormalise a list of (D,) vectors into a (D, r) basis (r <= rank).

    Skips near-zero residual vectors. Returns None if no usable column.
    """
    basis: list[torch.Tensor] = []
    for v in cols:
        if len(basis) >= rank:
            break
        w = v.float().clone()
        for b in basis:
            w = w - (b @ w) * b
        n = torch.linalg.norm(w)
        if n < eps:
            continue
        basis.append(w / n)
    if not basis:
        return None
    return torch.stack(basis, dim=1)                          # (D, r)


def random_orthonormal(D: int, rank: int, device, generator: torch.Generator) -> torch.Tensor:
    """A random (D, r) column-orthonormal matrix via QR of a Gaussian."""
    g = torch.randn(D, rank, generator=generator, device=device, dtype=torch.float32)
    q, _ = torch.linalg.qr(g)                                  # q: (D, r), orthonormal columns
    return q[:, :rank].contiguous()


# ----------------------------- core forward (generalised from ablate_features.py)
@torch.no_grad()
def forward_with_subspace_ablation(
    agent, obs_t, act_t, lat_t,
    target_block_idx: int = 1,
    Q: torch.Tensor | None = None,         # (D, r) orthonormal basis, or None
    ablation_step: int = -1,
    strength: float = 1.0,
):
    """Run the WM forward over one chunk; optionally remove the orthonormal
    subspace Q (h <- h - strength * Q (Q^T h)) at the 4 latent positions of
    `ablation_step` after block `target_block_idx`. Returns the WorldModelOutput.

    For r==1 this is exactly the single-direction projection of ablate_features.py.
    """
    wm = agent.world_model
    tpb = wm.config.transformer_config.tokens_per_block            # 6
    L = obs_t.shape[1]

    frames_emb = wm.frame_cnn(obs_t)                               # (1, L, 1, D)
    a_emb = wm.act_emb(act_t).unsqueeze(2)                         # (1, L, 1, D)
    l_emb = wm.latents_emb(lat_t)                                  # (1, L, K, D)
    seq = torch.cat([frames_emb, a_emb, l_emb], dim=2)            # (1, L, tpb, D)
    seq_flat = rearrange(seq, "b t p d -> b (t p) d")             # (1, L*tpb, D)

    handle = None
    if Q is not None:
        Qd = Q.float().to(seq_flat.device)                        # (D, r)
        pos_start = ablation_step * tpb + 2                       # first latent pos of that step
        pos_end = ablation_step * tpb + 6                         # 4 latent positions

        def hook(_m, _inp, output):
            out = output
            x = out[:, pos_start:pos_end]                          # (1, 4, D)
            coeffs = x @ Qd                                        # (1, 4, r)
            proj = coeffs @ Qd.t()                                 # (1, 4, D)
            new_out = out.clone()
            new_out[:, pos_start:pos_end] = x - strength * proj
            return new_out

        handle = wm.transformer.blocks[target_block_idx].register_forward_hook(hook)

    try:
        wm_out = wm(seq_flat)
    finally:
        if handle is not None:
            handle.remove()
    return wm_out


def log_prob_codes_at_step(wm_out, real_lat: torch.Tensor, target_step: int) -> float:
    """log P(lat_0..lat_3 at target_step | context).  Copied from ablate_features.py."""
    logits = wm_out.logits_latents
    if logits.dim() == 4:
        per_slot_logits = logits[0, target_step]                  # (K, V)
    elif logits.dim() == 3:
        T = logits.size(1) // real_lat.size(1)
        logits_resh = logits.view(logits.size(0), T, -1, logits.size(-1))
        per_slot_logits = logits_resh[0, target_step]
    else:
        raise RuntimeError(f"unexpected logits shape {logits.shape}")
    log_probs = F.log_softmax(per_slot_logits.float(), dim=-1)
    target = real_lat[target_step].long()                         # (K,)
    return float(log_probs.gather(-1, target.unsqueeze(-1)).sum().item())


def bootstrap_ci(arr, B=500):
    """Reused from ablate_features.py."""
    arr = np.asarray(arr, dtype=float)
    if arr.size <= 1:
        m = float(arr.mean()) if arr.size else 0.0
        return m, m
    rng2 = np.random.default_rng(0)
    means = [rng2.choice(arr, size=arr.size, replace=True).mean() for _ in range(B)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ----------------------------- block-1 activation extraction (from probe_layers.py)
@torch.no_grad()
def extract_block1_summary(agent, obs, obs_starts, tokens, actions, ep_ids, ep_lens, device):
    """Block-1 summary activations (mean over the 4 latent positions) per step.

    Returns (feats (N, D) float32, label_index (N,) global step rows). Mirrors the
    extraction in probe_layers.py / cav_traces.py EXACTLY (hook on block 1, reshape
    to (L, tokens_per_block, embed_dim), take positions 2:2+K and mean-pool).
    """
    wm, tk = agent.world_model, agent.tokenizer
    embed_dim = wm.config.transformer_config.embed_dim
    tokens_per_block = wm.config.transformer_config.tokens_per_block       # 6
    max_blocks = wm.config.transformer_config.max_blocks                   # 21
    K = tokens.shape[1]
    n_ep = ep_lens.shape[0]

    capture: dict[int, torch.Tensor] = {}
    block_idx = 1
    handle = wm.transformer.blocks[block_idx].register_forward_hook(
        lambda _m, _i, o: capture.__setitem__(block_idx, o.detach()))

    feats: list[np.ndarray] = []
    label_idx: list[int] = []
    t0 = time.time()
    steps_done = 0
    for ep in range(n_ep):
        T = int(ep_lens[ep])
        ep_obs = obs[obs_starts[ep]:obs_starts[ep + 1]]
        ep_act = actions[ep_ids == ep]
        ep_tok = tokens[ep_ids == ep]
        global_start = int(np.flatnonzero(ep_ids == ep)[0])
        for s in range(0, T, max_blocks):
            e = min(s + max_blocks, T)
            L = e - s
            obs_t = torch.from_numpy(ep_obs[s:e + 1]).to(device).float().div(255).unsqueeze(0)
            act_t = torch.from_numpy(ep_act[s:e]).to(device).unsqueeze(0)
            lat_t = torch.from_numpy(ep_tok[s:e].astype(np.int64)).to(device).unsqueeze(0)
            frames_emb = wm.frame_cnn(obs_t[:, :L])
            act_emb = wm.act_emb(act_t).unsqueeze(2)
            lat_emb = wm.latents_emb(lat_t)
            seq = torch.cat((frames_emb, act_emb, lat_emb), dim=2)
            seq_flat = rearrange(seq, 'b t p e -> b (t p) e')
            capture.clear()
            _ = wm.transformer(seq_flat, use_kv_cache=False)
            assert block_idx in capture, "block-1 hook did not fire"
            hs = capture[block_idx][0].reshape(L, tokens_per_block, embed_dim)
            summary = hs[:, 2:2 + K].mean(dim=1).cpu().numpy()                 # (L, E)
            assert summary.shape == (L, embed_dim), summary.shape
            feats.append(summary)
            label_idx.extend(range(global_start + s, global_start + e))
            steps_done += L
        if (ep + 1) % max(1, n_ep // 10) == 0 or ep == n_ep - 1:
            print(f"  [probe-extract] ep {ep+1}/{n_ep}  steps {steps_done}  "
                  f"{steps_done/max(time.time()-t0,1):.0f} st/s", flush=True)
    handle.remove()
    feats = np.concatenate(feats, axis=0).astype(np.float32)
    return feats, np.array(label_idx, dtype=np.int64)


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
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-trials-per-set", type=int, default=40)
    ap.add_argument("--window", type=int, default=21)
    ap.add_argument("--gap", type=int, default=3,
                    help="distance between ablation step and target step")
    ap.add_argument("--target-block", type=int, default=1,
                    help="WM transformer block to hook (0-indexed); CAV/SAE space = block 1")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--ranks", default="1,2,4,8,12,16",
                    help="comma-separated subspace ranks to test")
    ap.add_argument("--max-concepts", type=int, default=8,
                    help="cap the number of dominant concepts (by CAV AUROC / detector lift)")
    ap.add_argument("--concepts", default="",
                    help="comma-separated achievement names; empty = auto-pick dominant")
    ap.add_argument("--probe-rep", default="wm_block_1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    run = args.run.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    ranks = sorted({int(x) for x in args.ranks.split(",") if x.strip()})
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=device); gen.manual_seed(args.seed)

    print("loading agent...", flush=True)
    agent = load_agent(run, device)
    decoder, b_dec, sae_cfg = load_sae(args.sae, device)
    D_sae = decoder.shape[0]
    embed_dim = agent.world_model.config.transformer_config.embed_dim
    print(f"SAE: D={D_sae}  n_features={decoder.shape[1]}  layer={sae_cfg.get('layer')}", flush=True)
    assert D_sae == embed_dim, f"SAE D {D_sae} != WM embed_dim {embed_dim}"

    # ----- rollouts (block reused from ablate_features.py) -----------------
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
    print(f"rollouts: {tokens.shape[0]} steps / {ep_lens.shape[0]} episodes / {len(ach_names)} achievements", flush=True)

    # ----- CAVs (block-1 space, exactly as imagine_ablation.cav_direction) --
    cav_npz = np.load(args.cavs, allow_pickle=True)
    cav_names = [str(x) for x in cav_npz["concept_names"]]
    layer_key = "wm_block_1"
    W_cav = cav_npz[f"w_{layer_key}"].astype(np.float32)        # (n_concepts, D)
    sd_cav = cav_npz[f"std_{layer_key}"].astype(np.float32)     # (D,)
    assert W_cav.shape[1] == embed_dim, (W_cav.shape, embed_dim)

    def cav_vec(ach: str) -> np.ndarray | None:
        """CAV direction for just[ach] in raw block-1 activation space (w/std)."""
        name = f"just[{ach}]"
        if name not in cav_names:
            return None
        w = W_cav[cav_names.index(name)]
        if not np.any(w):
            return None
        return (w / sd_cav).astype(np.float32)

    # ----- SAE feature stacks per concept (rank by best_ach lift) ----------
    feats_json = json.loads(args.features.read_text())["features"]
    sae_feats_for: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for f in feats_json:
        a = f.get("best_ach")
        if not a:
            continue
        if f.get("density", 0.0) < 0.001:
            continue
        sae_feats_for[a["name"]].append((float(a["lift"]), int(f["feature"])))
    for k in sae_feats_for:
        sae_feats_for[k].sort(reverse=True, key=lambda x: x[0])   # highest lift first

    # ----- choose dominant concepts ----------------------------------------
    # candidates must have a nonzero CAV (decodable) AND some SAE feature.
    candidates = [a for a in ach_names if cav_vec(a) is not None and len(sae_feats_for.get(a, [])) > 0]
    if args.concepts:
        wanted = [c.strip() for c in args.concepts.split(",") if c.strip()]
        concepts = [c for c in wanted if c in candidates]
    else:
        # rank by best SAE-feature lift (a proxy for concept dominance / decodability)
        concepts = sorted(candidates, key=lambda a: -sae_feats_for[a][0][0])[:args.max_concepts]
    print(f"selected {len(concepts)} dominant concepts: {concepts}", flush=True)
    if not concepts:
        raise RuntimeError("no usable concepts (need nonzero CAV + SAE feature)")

    # ----- tech-tree neighbours via CAV cosine similarity ------------------
    # Data-driven proxy for Crafter's tech tree: a concept's neighbours are the
    # other achievements whose block-1 CAV is most cosine-similar to it.
    all_cav_concepts = [a for a in ach_names if cav_vec(a) is not None]
    cav_mat = {a: cav_vec(a) for a in all_cav_concepts}

    def neighbours(ach: str) -> list[str]:
        base = cav_mat[ach]
        bn = base / (np.linalg.norm(base) + 1e-8)
        sims = []
        for other in all_cav_concepts:
            if other == ach:
                continue
            v = cav_mat[other]
            vn = v / (np.linalg.norm(v) + 1e-8)
            sims.append((float(bn @ vn), other))
        sims.sort(reverse=True)
        return [o for _, o in sims]                              # descending cosine

    # ----- build per-concept bases for the max rank; slice for smaller ranks
    max_rank = max(ranks)

    def matched_cav_basis(ach: str, rank: int) -> torch.Tensor | None:
        cols = [torch.from_numpy(cav_mat[ach]).to(device)]
        for nb in neighbours(ach):
            if len(cols) >= rank:
                break
            cols.append(torch.from_numpy(cav_mat[nb]).to(device))
        return gram_schmidt(cols, rank)

    def sae_stack_basis(ach: str, rank: int) -> torch.Tensor | None:
        feat_ids = [fid for _, fid in sae_feats_for.get(ach, [])][:rank]
        if not feat_ids:
            return None
        cols = [decoder[:, fid].clone() for fid in feat_ids]
        return gram_schmidt(cols, rank)

    def random_basis(rank: int) -> torch.Tensor:
        return random_orthonormal(embed_dim, rank, device, gen)

    # =========================================================================
    # PART A: causal single-step Delta-logP as a function of rank
    # =========================================================================
    print("\n=== PART A: subspace ablation Delta-logP ===", flush=True)
    trials: list[dict] = []
    t0 = time.time()
    for ci, concept in enumerate(concepts):
        concept_idx = ach_names.index(concept)

        # unlock moments with enough history (exactly ablate_features.py logic)
        unlock_global = np.where(aj[:, concept_idx])[0]
        valid_unlock = []
        for g in unlock_global:
            ep = ep_ids[g]
            ep_step = g - ep_start_row[ep]
            if ep_step >= args.window - 1:
                valid_unlock.append(g)
        if len(valid_unlock) < 5:
            print(f"  {concept}: not enough unlock moments with history, skipping", flush=True)
            continue
        unlock_sel = rng.choice(valid_unlock,
                                size=min(args.n_trials_per_set, len(valid_unlock)), replace=False)

        # matched ordinary moments (same episodes, enough history, no unlock at step)
        cand = []
        for ep in np.unique(ep_ids[unlock_sel]):
            ep_start = int(ep_start_row[ep]); ep_end = int(ep_start_row[ep + 1])
            for g in range(ep_start + args.window - 1, ep_end):
                if aj[g, concept_idx]:
                    continue
                cand.append(g)
        if not cand:
            print(f"  {concept}: no ordinary candidates, skipping", flush=True)
            continue
        ordinary_sel = rng.choice(cand, size=min(args.n_trials_per_set, len(cand)), replace=False)

        # precompute bases at the maximum rank, then slice columns for each rank
        cav_full = {rk: matched_cav_basis(concept, rk) for rk in ranks}
        sae_full = {rk: sae_stack_basis(concept, rk) for rk in ranks}
        rand_full = {rk: random_basis(rk) for rk in ranks}
        # report achieved ranks (Gram-Schmidt may drop collinear columns)
        for rk in ranks:
            ach_cav = 0 if cav_full[rk] is None else cav_full[rk].shape[1]
            ach_sae = 0 if sae_full[rk] is None else sae_full[rk].shape[1]
            print(f"    {concept} r={rk:2d}  cav_dim={ach_cav}  sae_dim={ach_sae}", flush=True)

        for moment_type, sel in [("unlock", unlock_sel), ("ordinary", ordinary_sel)]:
            for g in sel:
                ep = int(ep_ids[g]); g_local = int(g - ep_start_row[ep])
                lo = g_local - args.window + 1
                hi = g_local + 1
                ep_lat = tokens[ep_start_row[ep]:ep_start_row[ep + 1]]
                ep_act = actions[ep_start_row[ep]:ep_start_row[ep + 1]]
                ep_obs = all_obs[obs_starts[ep]:obs_starts[ep + 1]]

                lat_chunk = torch.from_numpy(ep_lat[lo:hi].astype(np.int64)).to(device)     # (window, K)
                act_chunk = torch.from_numpy(ep_act[lo:hi]).to(device)                      # (window,)
                obs_chunk = torch.from_numpy(ep_obs[lo:hi + 1]).to(device).float().div(255)
                obs_in = obs_chunk[:args.window].unsqueeze(0)
                act_in = act_chunk.unsqueeze(0)
                lat_in = lat_chunk.unsqueeze(0)

                target_step = args.window - 1
                ablation_step = target_step - args.gap

                # (baseline) once per moment, shared across all bases/ranks
                wm_out = forward_with_subspace_ablation(
                    agent, obs_in, act_in, lat_in,
                    target_block_idx=args.target_block, Q=None,
                    ablation_step=-1, strength=0.0)
                logp_base = log_prob_codes_at_step(wm_out, lat_chunk, target_step)

                for rk in ranks:
                    bases = [("cav", cav_full[rk]), ("sae", sae_full[rk]), ("random", rand_full[rk])]
                    for basis_name, Q in bases:
                        if Q is None:
                            continue
                        assert Q.shape[0] == embed_dim, Q.shape
                        wm_out = forward_with_subspace_ablation(
                            agent, obs_in, act_in, lat_in,
                            target_block_idx=args.target_block, Q=Q,
                            ablation_step=ablation_step, strength=args.strength)
                        logp_abl = log_prob_codes_at_step(wm_out, lat_chunk, target_step)
                        trials.append(dict(
                            concept=concept, basis=basis_name, rank=int(rk),
                            achieved_rank=int(Q.shape[1]),
                            moment_type=moment_type, episode=ep, global_step=int(g),
                            ep_step=g_local, logp_base=logp_base, logp_abl=logp_abl,
                            delta=logp_base - logp_abl,            # >0 if ablation hurts prediction
                        ))

        elapsed = time.time() - t0
        rate = (ci + 1) / max(elapsed, 1e-3)
        eta_min = (len(concepts) - ci - 1) / max(rate, 1e-6) / 60
        n_c = len([t for t in trials if t["concept"] == concept])
        print(f"  done concept {ci+1}/{len(concepts)}: {concept}  ({n_c} trials)  eta {eta_min:.1f} min",
              flush=True)

    print(f"\nPART A: {len(trials)} trials in {(time.time()-t0)/60:.1f} min", flush=True)

    # ----- aggregate Delta-logP --------------------------------------------
    by = defaultdict(list)
    for t in trials:
        by[(t["concept"], t["basis"], t["rank"], t["moment_type"])].append(t["delta"])
    effects = []
    for (concept, basis, rk, mtype), vals in by.items():
        arr = np.asarray(vals, dtype=float)
        lo, hi = bootstrap_ci(arr)
        effects.append(dict(concept=concept, basis=basis, rank=int(rk), moment_type=mtype,
                            n=int(arr.size), mean=float(arr.mean()), std=float(arr.std()),
                            ci_lo=lo, ci_hi=hi))
    (args.out / "subspace_effects.json").write_text(json.dumps(effects, indent=2))
    (args.out / "trials.json").write_text(json.dumps(trials, indent=2))
    print(f"saved {args.out / 'subspace_effects.json'}", flush=True)

    # =========================================================================
    # PART B: probe-AUROC-recovery curve (refit probe after CAV-subspace removal)
    # =========================================================================
    print("\n=== PART B: probe-AUROC recovery ===", flush=True)
    feats_act, label_index = extract_block1_summary(
        agent, all_obs, obs_starts, tokens, actions, ep_ids, ep_lens, device)
    assert feats_act.shape[1] == embed_dim, feats_act.shape
    print(f"  extracted {feats_act.shape[0]} block-1 activations (D={feats_act.shape[1]})", flush=True)

    # episode-disjoint 80/20 split (exactly as probe_layers.py / cav_traces.py)
    sel_ep = ep_ids[label_index]
    rng_split = np.random.default_rng(0)
    uniq = np.unique(sel_ep); rng_split.shuffle(uniq)
    n_tr = int(len(uniq) * 0.8)
    train_eps = set(uniq[:n_tr].tolist())
    is_tr = np.isin(sel_ep, list(train_eps))
    print(f"  split: {is_tr.sum()} train / {(~is_tr).sum()} test", flush=True)

    def fit_auroc(X: np.ndarray, y: np.ndarray) -> float | None:
        ytr, yte = y[is_tr], y[~is_tr]
        if int(yte.sum()) < 5 or int(yte.sum()) == yte.shape[0]:
            return None
        Xtr, Xte = X[is_tr], X[~is_tr]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
        clf = LogisticRegression(max_iter=200, class_weight="balanced", C=1.0, solver="lbfgs")
        clf.fit(Xtr, ytr)
        score = clf.decision_function(Xte)
        try:
            return float(roc_auc_score(yte, score))
        except Exception:
            return None

    recovery = []
    for ci, concept in enumerate(concepts):
        c_idx = ach_names.index(concept)
        y = aj[label_index, c_idx].astype(np.int64)
        # rank 0 baseline (full activations)
        base_auroc = fit_auroc(feats_act, y)
        recovery.append(dict(concept=concept, rank=0, basis="none",
                             auroc=base_auroc, n_pos_test=int(y[~is_tr].sum())))
        # project out the rank-r MATCHED-CAV subspace, then refit
        for rk in ranks:
            Q = matched_cav_basis(concept, rk)
            if Q is None:
                recovery.append(dict(concept=concept, rank=int(rk), basis="cav",
                                     auroc=None, n_pos_test=int(y[~is_tr].sum())))
                continue
            Qn = Q.detach().cpu().numpy().astype(np.float32)       # (D, r) orthonormal
            X_proj = feats_act - (feats_act @ Qn) @ Qn.T           # remove subspace
            au = fit_auroc(X_proj, y)
            recovery.append(dict(concept=concept, rank=int(rk), basis="cav",
                                 achieved_rank=int(Qn.shape[1]), auroc=au,
                                 n_pos_test=int(y[~is_tr].sum())))
        bstr = "  ".join(
            f"r{rr['rank']}={rr['auroc']:.3f}" if rr["auroc"] is not None else f"r{rr['rank']}=NA"
            for rr in recovery if rr["concept"] == concept)
        print(f"  [{ci+1}/{len(concepts)}] {concept}: {bstr}", flush=True)

    (args.out / "auroc_recovery.json").write_text(json.dumps(recovery, indent=2))
    print(f"saved {args.out / 'auroc_recovery.json'}", flush=True)

    # ----- meta ------------------------------------------------------------
    meta = dict(run=str(run), rollouts=str(args.rollouts), obs=str(args.obs),
                sae=str(args.sae), cavs=str(args.cavs),
                ranks=ranks, concepts=concepts, window=args.window, gap=args.gap,
                target_block=args.target_block, strength=args.strength,
                n_trials_per_set=args.n_trials_per_set, seed=args.seed,
                embed_dim=int(embed_dim))
    (args.out / "subspace_meta.json").write_text(json.dumps(meta, indent=2))

    # =========================================================================
    # render
    # =========================================================================
    try:
        render_html(args, effects, recovery, concepts, ranks)
    except Exception as e:                                          # never let plotting kill the run
        print(f"WARN: HTML render failed: {e}", flush=True)

    print("DONE", flush=True)


def render_html(args, effects, recovery, concepts, ranks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    basis_color = {"cav": "#9467bd", "sae": "#d62728", "random": "#999999"}

    def eff_lookup(concept, basis, rk, mtype):
        return next((s for s in effects if s["concept"] == concept and s["basis"] == basis
                     and s["rank"] == rk and s["moment_type"] == mtype), None)

    # --- Figure 1: Delta-logP vs rank (unlock moments), one line/basis per concept
    n_c = len(concepts)
    ncols = min(4, n_c) if n_c else 1
    nrows = int(np.ceil(n_c / ncols)) if n_c else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.0 * nrows), squeeze=False)
    for i, concept in enumerate(concepts):
        ax = axes[i // ncols][i % ncols]
        for basis in ["cav", "sae", "random"]:
            xs, ys, lo, hi = [], [], [], []
            for rk in ranks:
                rec = eff_lookup(concept, basis, rk, "unlock")
                if rec is None:
                    continue
                xs.append(rk); ys.append(rec["mean"])
                lo.append(rec["mean"] - rec["ci_lo"]); hi.append(rec["ci_hi"] - rec["mean"])
            if xs:
                ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", ms=4, lw=1.3, capsize=2,
                            color=basis_color[basis], label=basis)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(concept, fontsize=9)
        ax.set_xlabel("rank r"); ax.grid(alpha=0.3)
        if i % ncols == 0:
            ax.set_ylabel("Δ logP (base − ablated)")
        if i == 0:
            ax.legend(fontsize=8)
    for j in range(n_c, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(f"Subspace ablation: single-step Δ logP vs rank (unlock moments, gap={args.gap}, block {args.target_block})",
                 fontsize=11)
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=130); plt.close(fig)
    img1 = base64.b64encode(buf.getvalue()).decode("ascii")

    # --- Figure 2: AUROC recovery curve
    fig2, ax2 = plt.subplots(figsize=(7.5, 5.0))
    cmap = plt.cm.tab10.colors
    for i, concept in enumerate(concepts):
        recs = sorted([r for r in recovery if r["concept"] == concept], key=lambda r: r["rank"])
        xs = [r["rank"] for r in recs if r["auroc"] is not None]
        ys = [r["auroc"] for r in recs if r["auroc"] is not None]
        if xs:
            ax2.plot(xs, ys, marker="o", ms=4, lw=1.2, color=cmap[i % 10], label=concept)
    ax2.axhline(0.5, color="black", ls="--", lw=0.6)
    ax2.set_xlabel("rank r of CAV subspace projected out")
    ax2.set_ylabel("held-out AUROC for just[concept]")
    ax2.set_ylim(0.4, 1.02); ax2.grid(alpha=0.3)
    ax2.set_title("Probe-AUROC recovery: how many CAV directions until undecodable")
    ax2.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    buf2 = io.BytesIO(); plt.savefig(buf2, format="png", dpi=130); plt.close(fig2)
    img2 = base64.b64encode(buf2.getvalue()).decode("ascii")

    rows = []
    for s in sorted(effects, key=lambda x: (x["concept"], x["basis"], x["rank"], x["moment_type"])):
        rows.append(f"<tr><td class=label>{s['concept']}</td><td>{s['basis']}</td>"
                    f"<td>{s['rank']}</td><td>{s['moment_type']}</td><td>{s['n']}</td>"
                    f"<td>{s['mean']:+.3f}</td><td>[{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}]</td></tr>")
    parts = [
        "<!doctype html><meta charset=utf-8><title>subspace ablation</title>",
        "<style>"
        "body{font:14px ui-monospace,Menlo,Consolas,monospace;background:#0d1117;color:#c9d1d9;padding:18px;}"
        "h1{color:#79c0ff;font-size:18px;margin:0 0 8px;}h2{color:#d2a8ff;font-size:15px;margin:16px 0 6px;}"
        ".dim{color:#8b949e;}img{max-width:100%;background:white;border:1px solid #21262d;border-radius:6px;}"
        "table{width:100%;border-collapse:collapse;font-size:12px;margin-top:8px;}"
        "th,td{padding:4px 8px;border-bottom:1px solid #21262d;text-align:right;}"
        "td.label{text-align:left;}th{color:#8b949e;text-transform:uppercase;font-size:11px;}</style>",
        "<h1>Multi-direction / INLP subspace ablation</h1>",
        f"<div class=dim>{sum(s['n'] for s in effects if s['moment_type']=='unlock' and s['basis']=='cav')} unlock CAV trials · "
        f"projection of rank-r orthonormal basis at the 4 latent positions of step T−{args.gap}, "
        f"WM block {args.target_block} · ranks {ranks} · strength {args.strength}</div>",
        "<h2>Δ logP vs rank (CAV vs SAE-stack vs random)</h2>",
        f"<img src='data:image/png;base64,{img1}'>",
        "<h2>Probe-AUROC recovery (project out rank-r CAV subspace, refit probe)</h2>",
        f"<img src='data:image/png;base64,{img2}'>",
        "<h2>Δ logP summary table</h2>",
        "<table><tr><th class=label>concept</th><th>basis</th><th>rank</th><th>moment</th>"
        "<th>n</th><th>mean Δ logP</th><th>95% CI</th></tr>",
        *rows, "</table>",
    ]
    (args.out / "subspace.html").write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {args.out / 'subspace.html'}", flush=True)


if __name__ == "__main__":
    main()
