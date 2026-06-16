"""SAE robustness sweep for Δ-IRIS: dictionary width × sparsity-k × seed, plus
SAEs at OTHER layers, with feature-stability matching of the headline detectors.

Self-contained extension of analysis/train_sae.py.  We reuse train_sae.py's exact
TopKSAE class + training loop and its block-1 activation extraction, and add:

  1. A small CONFIG GRID of TopK SAEs trained on the existing block-1 summary
     activations (the same (N≈81,919) × 512 tensor train_sae.py builds):
       - widths  M ∈ {1024, 2048, 4096}   at k=16, seed=0
       - sparsity k ∈ {8, 16, 32}         at M=2048, seed=0
       - 2 random seeds {0, 1}            at the default (M=2048, k=16)
     (the M=2048/k=16/seed=0 point is shared across all three axes — trained once.)
     For each config we report FVU (fraction of variance unexplained, on an
     episode-disjoint val split, identical to train_sae.py) and the dead-feature
     count.

  2. ONE SAE each at OTHER layers (default M=2048, k=16, seed=0):
       - frame_emb   : frame_cnn(current frame), 512-d, NO transformer
       - wm_input    : transformer block-0 output (a.k.a. "block-0")
       - wm_block_2  : transformer block-2 output
       - wm_block_3  : transformer block-3 = final post-LN hidden state
     (the default headline layer wm_block_1 is already covered by the grid.)
     All seven representations are extracted in a SINGLE forward pass per window,
     using exactly probe_layers.py's summary-position convention (mean over the 4
     latent positions 2..2+K within each block; frame_emb is the CNN frame token).

  3. FEATURE STABILITY for the headline detector features:
       collect_coal, collect_iron, make_stone_pickaxe, place_table, collect_wood (f187).
     For every trained SAE we (re-)identify the feature that best detects each
     achievement (max P(ach-just-unlocked | feature in top-quartile) lift, exactly
     train_sae.py's "best_ach" criterion).  We then match the canonical reference
     features (from the default wm_block_1 M=2048/k=16/seed=0 SAE) into every other
     same-layer SAE by MAX DECODER COSINE, and report the matching cosine — showing
     the cross-corroborated detectors are stable across width / k / seed.

Outputs under $ANA/sae_robustness/:
  sae_grid.json          config -> {FVU, n_dead, n_alive, train/val sizes, headline feature ids, ...}
  feature_stability.json reference detector features + per-config max-cosine matches

Cost: one 7-layer activation extraction pass (~10-15 min) + ~10 small SAE trainings
(each cheaper than the full 80-epoch run; we cap epochs) ≈ well under the 4h budget.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import OmegaConf


# ===========================================================================
# TopK SAE  --  copied verbatim from analysis/train_sae.py so this script is
# self-contained and trains EXACTLY the same model.
# ===========================================================================
class TopKSAE(nn.Module):
    def __init__(self, d_in: int, d_features: int, k: int):
        super().__init__()
        self.d_in, self.d_features, self.k = d_in, d_features, k
        self.encoder = nn.Linear(d_in, d_features, bias=True)
        self.decoder = nn.Linear(d_features, d_in, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        with torch.no_grad():
            w = torch.randn(d_features, d_in) / (d_in ** 0.5)
            self.decoder.weight.copy_(w.T)
            self.encoder.weight.copy_(w)
            self.encoder.bias.zero_()
            self._normalize_decoder()

    @torch.no_grad()
    def _normalize_decoder(self):
        norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.decoder.weight.div_(norms)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x - self.b_dec)
        vals, idx = z.topk(self.k, dim=-1)
        vals = vals.relu()
        out = torch.zeros_like(z)
        out.scatter_(-1, idx, vals)
        return out

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decoder(z) + self.b_dec
        return x_hat, z


# ===========================================================================
# Agent loader  --  same convention as train_sae.py / probe_layers.py
# ===========================================================================
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
               load_tokenizer=True, load_world_model=True, load_actor_critic=True,
               strict=False)
    return agent


# ===========================================================================
# Multi-layer activation extraction in ONE forward pass.
#
# Mirrors train_sae.py.extract_layer and probe_layers.py exactly:
#   - windows are max_blocks-step chunks within each episode
#   - block hooks capture each transformer block's output
#   - the FINAL post-LN output is treated as wm_block_{num_layers}
#   - per timestep the "summary" is the mean over the 4 latent positions
#       hs[:, 2:2+K, :].mean(1)               (K latent tokens, indices 2..2+K)
#   - frame_emb is the CNN frame token (frames_emb[:, :, 0]) per timestep
#
# Layer names produced (all 512-d here):
#   frame_emb, wm_input(=block-0), wm_block_1, wm_block_2, wm_block_3(=final)
# Returns dict[name] -> (N, D) float32, plus label_idx (N,) global step indices.
# ===========================================================================
def extract_all_layers(agent, rollouts, obs_path, layer_names, device):
    wm, tk = agent.world_model, agent.tokenizer
    num_layers = wm.config.transformer_config.num_layers
    embed_dim = wm.config.transformer_config.embed_dim
    tokens_per_block = wm.config.transformer_config.tokens_per_block
    max_blocks = wm.config.transformer_config.max_blocks

    obs_npz = np.load(obs_path)
    obs = obs_npz["obs"]; obs_starts = obs_npz["episode_starts"]
    actions = rollouts["actions"].astype(np.int64)
    tokens = rollouts["tokens"]
    ep_ids = rollouts["episode_ids"].astype(np.int64)
    ep_lens = rollouts["episode_lengths"].astype(np.int64)
    n_ep = ep_lens.shape[0]; K = tokens.shape[1]

    # block -> layer-name mapping (hook idx 0 == block-0 == wm_input)
    blockidx_to_name = {0: "wm_input"}
    for li in range(1, num_layers + 1):
        blockidx_to_name[li] = f"wm_block_{li}"

    capture: dict[int, torch.Tensor] = {}
    hooks = []
    for i, block in enumerate(wm.transformer.blocks):
        def mk(i=i):
            def h(_m, _i, o): capture[i] = o.detach()
            return h
        hooks.append(block.register_forward_hook(mk()))

    out_arrs: dict[str, list[np.ndarray]] = {n: [] for n in layer_names}
    label_idx: list[int] = []
    t0 = time.time(); steps_done = 0
    for ep in range(n_ep):
        T = int(ep_lens[ep])
        ep_obs = obs[obs_starts[ep]:obs_starts[ep+1]]
        ep_act = actions[ep_ids == ep]
        ep_tok = tokens[ep_ids == ep]
        global_start = int(np.flatnonzero(ep_ids == ep)[0])
        for s in range(0, T, max_blocks):
            e = min(s + max_blocks, T); L = e - s
            obs_t = torch.from_numpy(ep_obs[s:e+1]).to(device).float().div(255).unsqueeze(0)
            act_t = torch.from_numpy(ep_act[s:e]).to(device).unsqueeze(0)
            lat_t = torch.from_numpy(ep_tok[s:e].astype(np.int64)).to(device).unsqueeze(0)
            with torch.no_grad():
                frames_emb = wm.frame_cnn(obs_t[:, :L])               # (1, L, 1, E)
                act_emb = wm.act_emb(act_t).unsqueeze(2)              # (1, L, 1, E)
                lat_emb = wm.latents_emb(lat_t)                       # (1, L, K, E)
                seq = torch.cat((frames_emb, act_emb, lat_emb), dim=2)
                seq_flat = rearrange(seq, 'b t p e -> b (t p) e')
                capture.clear()
                final = wm.transformer(seq_flat, use_kv_cache=False)  # (1, L*tpb, E)
                capture[num_layers] = final.detach()
                # frame_emb summary: the CNN frame token (position 0 in each block)
                if "frame_emb" in out_arrs:
                    fe = frames_emb.squeeze(0).squeeze(1).cpu().numpy()    # (L, E)
                    assert fe.shape == (L, embed_dim), (fe.shape, L, embed_dim)
                    out_arrs["frame_emb"].append(fe)
            # transformer-layer summaries
            for bidx, name in blockidx_to_name.items():
                if name not in out_arrs:
                    continue
                hs = capture[bidx].reshape(1, L, tokens_per_block, embed_dim)
                summary = hs[0, :, 2:2+K].mean(dim=1).cpu().numpy()        # (L, E)
                assert summary.shape == (L, embed_dim), (summary.shape, L, embed_dim)
                out_arrs[name].append(summary)
            label_idx.extend(range(global_start + s, global_start + e))
            steps_done += L
        if (ep + 1) % max(1, n_ep // 10) == 0 or ep == n_ep - 1:
            print(f"  ep {ep+1}/{n_ep}  steps {steps_done}  "
                  f"{steps_done/max(time.time()-t0,1):.0f} st/s", flush=True)
    for h in hooks: h.remove()

    reps = {n: np.concatenate(out_arrs[n], axis=0).astype(np.float32) for n in layer_names}
    label_idx = np.array(label_idx, dtype=np.int64)
    Nref = label_idx.shape[0]
    for n, v in reps.items():
        assert v.shape[0] == Nref, (n, v.shape, Nref)
    return reps, label_idx


# ===========================================================================
# Episode-disjoint train/val split  --  identical 90/10 logic to train_sae.py
# ===========================================================================
def make_split(ep_ids, label_idx, seed=0):
    sel_ep = ep_ids[label_idx]
    rng = np.random.default_rng(seed)
    uniq = np.unique(sel_ep); rng.shuffle(uniq)
    n_tr = int(len(uniq) * 0.9)
    train_eps = set(uniq[:n_tr].tolist())
    is_tr_mask = np.isin(sel_ep, list(train_eps))
    train_idx = np.where(is_tr_mask)[0]
    val_idx = np.where(~is_tr_mask)[0]
    return train_idx, val_idx


# ===========================================================================
# Train one SAE  --  identical loop to train_sae.py (decoder renorm each step,
# Adam, MSE loss, FVU reported on val).  Returns the trained SAE + metrics.
# ===========================================================================
def train_sae(X_t, X_np, train_idx, val_idx, d_in, d_features, k, seed,
              epochs, batch_size, lr, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    sae = TopKSAE(d_in=d_in, d_features=d_features, k=k).to(device)
    sae.b_dec.data.copy_(torch.from_numpy(X_np.mean(0)).to(device))
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    norm_var = float(X_np[train_idx].var())   # variance over all dims & samples (matches train_sae.py)

    t0 = time.time()
    for epoch in range(epochs):
        sae.train()
        perm = rng.permutation(train_idx)
        running_mse = 0.0; n = 0
        for start in range(0, perm.shape[0], batch_size):
            b = perm[start:start+batch_size]
            x = X_t[b]
            x_hat, z = sae(x)
            mse = F.mse_loss(x_hat, x)
            opt.zero_grad(); mse.backward(); opt.step()
            sae._normalize_decoder()
            running_mse += mse.item() * b.shape[0]; n += b.shape[0]
        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            sae.eval()
            with torch.no_grad():
                x = X_t[val_idx]
                x_hat, z = sae(x)
                val_mse = F.mse_loss(x_hat, x).item()
                val_fvu = val_mse / max(norm_var, 1e-8)
            print(f"    epoch {epoch+1:3d}/{epochs}  train MSE={running_mse/n:.5f}  "
                  f"val MSE={val_mse:.5f}  FVU={val_fvu:.3f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # ---- final metrics on val + dead-feature count over ALL samples ----
    sae.eval()
    with torch.no_grad():
        xv = X_t[val_idx]
        xv_hat, _ = sae(xv)
        val_mse = F.mse_loss(xv_hat, xv).item()
        val_fvu = val_mse / max(norm_var, 1e-8)
        # density per feature over the FULL dataset (chunked encode)
        N = X_t.shape[0]
        active_counts = torch.zeros(d_features, device=device)
        for s in range(0, N, batch_size * 4):
            e = min(s + batch_size * 4, N)
            z = sae.encode(X_t[s:e])
            active_counts += (z > 0).float().sum(0)
        density = (active_counts / N).cpu().numpy()
    n_dead = int((active_counts == 0).sum().item())
    n_alive = d_features - n_dead
    metrics = dict(
        val_fvu=float(val_fvu), val_mse=float(val_mse), norm_var=float(norm_var),
        n_dead=int(n_dead), n_alive=int(n_alive),
        n_train=int(train_idx.shape[0]), n_val=int(val_idx.shape[0]),
        median_density=float(np.median(density)),
        mean_density=float(density.mean()),
    )
    return sae, metrics, density


# ===========================================================================
# Headline-feature identification  --  re-derives, per SAE, the feature that
# best detects each target achievement, using train_sae.py's "best_ach" rule:
#   P(ach-just-unlocked | feature in top-quartile of its nonzero activations),
#   lift over the base unlock rate; pick argmax-lift feature for the achievement.
# Also returns the full encoded Z so callers can reuse it.
# ===========================================================================
def headline_features(sae, X_t, label_idx, aj, ach_names, target_achs,
                      batch_size, device, min_active=20, min_mask=5):
    d_features = sae.d_features
    N = X_t.shape[0]
    with torch.no_grad():
        Z = torch.zeros(N, d_features)
        for s in range(0, N, batch_size * 4):
            e = min(s + batch_size * 4, N)
            Z[s:e] = sae.encode(X_t[s:e]).cpu()
    Z_np = Z.numpy()
    n_active = (Z_np > 0).sum(0)
    aj_sel = aj[label_idx]                              # (N, n_ach)
    base_unlock = aj_sel.mean(0)                        # (n_ach,)

    ach_to_idx = {a: i for i, a in enumerate(ach_names)}
    result = {}
    for ach in target_achs:
        if ach not in ach_to_idx:
            result[ach] = dict(feature=None, lift=None, p=None, reason="ach not in names")
            continue
        a = ach_to_idx[ach]
        best_feat, best_lift, best_p = None, -1.0, None
        for f in range(d_features):
            if n_active[f] < min_active:
                continue
            z_f = Z_np[:, f]
            nz = z_f[z_f > 0]
            thr = np.percentile(nz, 75)
            mask = z_f >= max(thr, 1e-6)
            if mask.sum() < min_mask:
                continue
            p = float(aj_sel[mask, a].mean())
            lift = p / max(float(base_unlock[a]), 1e-8)
            if lift > best_lift:
                best_feat, best_lift, best_p = f, lift, p
        result[ach] = dict(feature=(int(best_feat) if best_feat is not None else None),
                           lift=(float(best_lift) if best_feat is not None else None),
                           p=(float(best_p) if best_p is not None else None))
    return result, Z_np


# ===========================================================================
# main
# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--rollouts", required=True, type=Path)
    ap.add_argument("--obs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=60,
                    help="epochs per SAE (capped below full pipeline's 80 to bound runtime)")
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--device", default="cuda:0")
    # headline detectors (collect_wood == f187 in the reference SAE)
    ap.add_argument("--target-achs", nargs="+",
                    default=["collect_coal", "collect_iron", "make_stone_pickaxe",
                             "place_table", "collect_wood"])
    args = ap.parse_args()

    run = args.run.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"loading agent from {run}", flush=True)
    agent = load_agent(run, device)
    wm = agent.world_model
    num_layers = wm.config.transformer_config.num_layers
    embed_dim = wm.config.transformer_config.embed_dim
    print(f"WM: {num_layers} transformer blocks, embed_dim={embed_dim}", flush=True)

    rollouts = np.load(args.rollouts, allow_pickle=True)
    aj = rollouts["ach_just_unlocked"]
    ep_ids = rollouts["episode_ids"].astype(np.int64)
    ach_names = [str(x) for x in rollouts["achievement_names"]]

    # ----- which layers to extract -----
    # headline layer for the grid = wm_block_1; "other" layers per spec =
    # frame_emb, block-0 (wm_input), block-2 (wm_block_2), final (wm_block_3).
    HEADLINE_LAYER = "wm_block_1"
    OTHER_LAYERS = ["frame_emb", "wm_input", "wm_block_2", "wm_block_3"]
    # only keep "other" layers whose block index actually exists in this model
    valid = {"frame_emb", "wm_input"} | {f"wm_block_{i}" for i in range(1, num_layers + 1)}
    OTHER_LAYERS = [l for l in OTHER_LAYERS if l in valid]
    layer_names = [HEADLINE_LAYER] + OTHER_LAYERS
    print(f"extracting layers: {layer_names}", flush=True)

    reps, label_idx = extract_all_layers(agent, rollouts, args.obs, layer_names, device)
    N = label_idx.shape[0]
    print(f"  extracted {N} summary samples per layer; "
          f"dims = {[reps[l].shape[1] for l in layer_names]}", flush=True)
    for l in layer_names:
        assert reps[l].shape == (N, embed_dim), (l, reps[l].shape)

    # episode-disjoint val split (seed 0, identical to train_sae.py). Reused for
    # every config so FVUs are comparable; same split for every layer (same rows).
    train_idx, val_idx = make_split(ep_ids, label_idx, seed=0)
    print(f"  split: train {train_idx.shape[0]}  val {val_idx.shape[0]}", flush=True)

    # ----- pre-move each layer's activations to device once (reused per config) -----
    rep_tensors = {l: torch.from_numpy(reps[l]).to(device) for l in layer_names}

    # =====================================================================
    # CONFIG GRID (on the headline layer wm_block_1)
    # =====================================================================
    DEFAULT_M, DEFAULT_K, DEFAULT_SEED = 2048, 16, 0
    grid_configs = []          # list of dicts: axis, M, k, seed
    seen = set()

    def add_cfg(axis, M, k, seed):
        key = (M, k, seed)
        if key in seen:
            return
        seen.add(key)
        grid_configs.append(dict(axis=axis, M=M, k=k, seed=seed))

    # width axis (k=16, seed=0)
    for M in (1024, 2048, 4096):
        add_cfg("width", M, DEFAULT_K, DEFAULT_SEED)
    # sparsity axis (M=2048, seed=0)
    for k in (8, 16, 32):
        add_cfg("sparsity", DEFAULT_M, k, DEFAULT_SEED)
    # seed axis (M=2048, k=16): seeds 0 and 1
    for seed in (0, 1):
        add_cfg("seed", DEFAULT_M, DEFAULT_K, seed)

    X_head_np = reps[HEADLINE_LAYER]
    X_head_t = rep_tensors[HEADLINE_LAYER]

    grid_results = []
    # cache trained headline SAEs (decoder + headline-feature ids) for stability matching
    head_saes = {}    # (M,k,seed) -> dict(decoder=np(D,M), headline=dict(ach->...))

    for ci, cfg in enumerate(grid_configs):
        M, k, seed = cfg["M"], cfg["k"], cfg["seed"]
        tag = f"{HEADLINE_LAYER}_M{M}_k{k}_s{seed}"
        print(f"\n=== [{ci+1}/{len(grid_configs)}] grid {tag} (axis={cfg['axis']}) ===", flush=True)
        sae, metrics, density = train_sae(
            X_head_t, X_head_np, train_idx, val_idx,
            d_in=embed_dim, d_features=M, k=k, seed=seed,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device)
        head, _Z = headline_features(sae, X_head_t, label_idx, aj, ach_names,
                                     args.target_achs, args.batch_size, device)
        rec = dict(config=tag, layer=HEADLINE_LAYER, axis=cfg["axis"],
                   M=M, k=k, seed=seed, D=embed_dim, **metrics,
                   headline_features={a: head[a] for a in args.target_achs})
        grid_results.append(rec)
        head_saes[(M, k, seed)] = dict(
            decoder=sae.decoder.weight.detach().cpu().numpy().copy(),  # (D, M)
            headline=head)
        print(f"  -> FVU={metrics['val_fvu']:.3f}  dead={metrics['n_dead']}/{M}  "
              f"alive={metrics['n_alive']}", flush=True)

    # =====================================================================
    # OTHER LAYERS (one SAE each at the default M=2048,k=16,seed=0)
    # =====================================================================
    other_results = []
    other_saes = {}      # layer -> dict(decoder, headline)
    for li, layer in enumerate(OTHER_LAYERS):
        tag = f"{layer}_M{DEFAULT_M}_k{DEFAULT_K}_s{DEFAULT_SEED}"
        print(f"\n=== [layer {li+1}/{len(OTHER_LAYERS)}] {tag} ===", flush=True)
        Xn = reps[layer]; Xt = rep_tensors[layer]
        sae, metrics, density = train_sae(
            Xt, Xn, train_idx, val_idx,
            d_in=embed_dim, d_features=DEFAULT_M, k=DEFAULT_K, seed=DEFAULT_SEED,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device)
        head, _Z = headline_features(sae, Xt, label_idx, aj, ach_names,
                                     args.target_achs, args.batch_size, device)
        rec = dict(config=tag, layer=layer, axis="layer",
                   M=DEFAULT_M, k=DEFAULT_K, seed=DEFAULT_SEED, D=embed_dim, **metrics,
                   headline_features={a: head[a] for a in args.target_achs})
        other_results.append(rec)
        other_saes[layer] = dict(
            decoder=sae.decoder.weight.detach().cpu().numpy().copy(),
            headline=head)
        print(f"  -> FVU={metrics['val_fvu']:.3f}  dead={metrics['n_dead']}/{DEFAULT_M}  "
              f"alive={metrics['n_alive']}", flush=True)

    # ----- write sae_grid.json -----
    grid_out = dict(
        run=str(run), layer_headline=HEADLINE_LAYER, other_layers=OTHER_LAYERS,
        N=int(N), D=int(embed_dim), epochs=args.epochs,
        default=dict(M=DEFAULT_M, k=DEFAULT_K, seed=DEFAULT_SEED),
        target_achs=args.target_achs,
        grid=grid_results, other_layers_results=other_results)
    (args.out / "sae_grid.json").write_text(
        json.dumps(grid_out, indent=2,
                   default=lambda o: int(o) if isinstance(o, np.integer) else float(o)))
    print(f"\nwrote {args.out / 'sae_grid.json'}", flush=True)

    # =====================================================================
    # FEATURE STABILITY
    #
    # Reference = default wm_block_1 SAE (M=2048, k=16, seed=0).  Its per-achievement
    # headline feature ids are the canonical detectors (collect_wood should be f187).
    # For every OTHER same-layer (wm_block_1) SAE config we match each reference
    # decoder column into the other SAE's decoder by MAX COSINE and report:
    #   - matched feature id + the matching cosine
    #   - whether the other SAE's OWN re-derived headline feature equals that match
    #     (a stronger statement of stability)
    # We also report each other config's own re-derived headline feature ids/lifts.
    # =====================================================================
    ref_key = (DEFAULT_M, DEFAULT_K, DEFAULT_SEED)
    assert ref_key in head_saes, "default reference SAE was not trained"
    ref_dec = head_saes[ref_key]["decoder"]                       # (D, M_ref)
    ref_dec_n = ref_dec / (np.linalg.norm(ref_dec, axis=0, keepdims=True) + 1e-8)
    ref_head = head_saes[ref_key]["headline"]

    def match_into(ref_feat_id, other_dec):
        """Cosine-match a reference decoder column into other_dec; return (idx, cos)."""
        if ref_feat_id is None:
            return None, None
        v = ref_dec_n[:, ref_feat_id]                            # (D,) unit
        od_n = other_dec / (np.linalg.norm(other_dec, axis=0, keepdims=True) + 1e-8)
        cos = od_n.T @ v                                          # (M_other,)
        j = int(np.argmax(cos))
        return j, float(cos[j])

    # reference detector summary
    ref_detectors = {}
    for ach in args.target_achs:
        ref_detectors[ach] = dict(
            feature=ref_head[ach]["feature"],
            lift=ref_head[ach]["lift"],
            p=ref_head[ach]["p"])

    # per-config matches, only against same-layer (wm_block_1) SAEs (cosine is only
    # meaningful within the same activation space).
    stability_per_config = []
    for (M, k, seed), blob in head_saes.items():
        if (M, k, seed) == ref_key:
            continue
        other_dec = blob["decoder"]
        other_head = blob["headline"]
        per_ach = {}
        for ach in args.target_achs:
            ref_fid = ref_head[ach]["feature"]
            mj, mcos = match_into(ref_fid, other_dec)
            own_fid = other_head[ach]["feature"]
            per_ach[ach] = dict(
                ref_feature=ref_fid,
                matched_feature=mj,
                match_cosine=mcos,
                own_headline_feature=own_fid,
                own_lift=other_head[ach]["lift"],
                match_is_own_headline=(mj is not None and own_fid is not None and mj == own_fid),
            )
        cosines = [per_ach[a]["match_cosine"] for a in args.target_achs
                   if per_ach[a]["match_cosine"] is not None]
        stability_per_config.append(dict(
            config=f"{HEADLINE_LAYER}_M{M}_k{k}_s{seed}",
            layer=HEADLINE_LAYER, M=M, k=k, seed=seed,
            mean_match_cosine=(float(np.mean(cosines)) if cosines else None),
            min_match_cosine=(float(np.min(cosines)) if cosines else None),
            per_ach=per_ach))

    stab_out = dict(
        run=str(run), reference_config=f"{HEADLINE_LAYER}_M{DEFAULT_M}_k{DEFAULT_K}_s{DEFAULT_SEED}",
        layer=HEADLINE_LAYER, target_achs=args.target_achs,
        note=("Cosine matching is computed only WITHIN the same activation space "
              "(wm_block_1). 'collect_wood' reference feature is expected to be f187. "
              "Other-layer SAEs report their own re-derived headline features in "
              "sae_grid.json (other_layers_results) but are not cosine-matched to "
              "block-1 because the spaces differ."),
        reference_detectors=ref_detectors,
        matches=stability_per_config)
    (args.out / "feature_stability.json").write_text(
        json.dumps(stab_out, indent=2,
                   default=lambda o: int(o) if isinstance(o, np.integer) else float(o)))
    print(f"wrote {args.out / 'feature_stability.json'}", flush=True)

    # ----- console summary -----
    print("\n==== reference detectors (wm_block_1, M=2048,k=16,seed=0) ====", flush=True)
    for ach in args.target_achs:
        d = ref_detectors[ach]
        print(f"  {ach:22s} -> f{d['feature']}  lift={d['lift']}", flush=True)
    print("\n==== stability (max decoder cosine vs reference) ====", flush=True)
    for s in stability_per_config:
        print(f"  {s['config']:28s}  mean_cos={s['mean_match_cosine']}  "
              f"min_cos={s['min_match_cosine']}", flush=True)
    print("\ndone.", flush=True)


if __name__ == "__main__":
    main()
