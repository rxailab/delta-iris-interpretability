"""Train a TopK Sparse Autoencoder on Δ-IRIS world-model hidden states and
match the resulting features against our existing concept catalogues.

Pipeline:
  1. Re-extract hidden states from --layer-rep (default wm_block_1) for the
     entire rollouts buffer.
  2. Train a TopK SAE: x ≈ W_dec · TopK(W_enc · (x - b)) + b
     - features = --n-features (default 2048, ≈4× expansion over 512-dim)
     - k        = --k-active (default 16)
  3. For each feature i, compute:
       * n_active across the full dataset (with what mean / max activation)
       * top-K samples that activate it most strongly (with their codes,
         action, achievement labels)
       * cosine similarity of its decoder direction with every CAV in cavs.npz
       * mean activation conditioned on each (slot, code) → strongest code match
       * P(achievement-just-unlocked | feature in top quartile) → "achievement"
         label for the feature
  4. Save:
       sae.pt           encoder/decoder + b
       features.json    per-feature alignment table
       features.html    browsable explorer

Cost: ~10 min activation extraction + ~10-15 min SAE training + ~2 min analysis.
"""
from __future__ import annotations

import argparse, base64, io, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import OmegaConf


# ---------- model -----------------------------------------------------------
class TopKSAE(nn.Module):
    def __init__(self, d_in: int, d_features: int, k: int):
        super().__init__()
        self.d_in, self.d_features, self.k = d_in, d_features, k
        self.encoder = nn.Linear(d_in, d_features, bias=True)
        self.decoder = nn.Linear(d_features, d_in, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        # init: encoder = decoder^T, decoder columns unit-norm
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


# ---------- activation extraction -------------------------------------------
def extract_layer(agent, rollouts, obs_path, layer: str, device):
    """Return (N, embed_dim) np.float32 hidden states at the chosen layer."""
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

    capture: dict[int, torch.Tensor] = {}
    hooks = []
    for i, block in enumerate(wm.transformer.blocks):
        def mk(i=i):
            def h(_m, _i, o): capture[i] = o.detach()
            return h
        hooks.append(block.register_forward_hook(mk()))

    # which captured layer index do we want?
    target = {"wm_input": 0,
              "wm_block_1": 1, "wm_block_2": 2, "wm_block_3": 3}[layer]

    out_arrs: list[np.ndarray] = []
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
                frames_emb = wm.frame_cnn(obs_t[:, :L])
                act_emb = wm.act_emb(act_t).unsqueeze(2)
                lat_emb = wm.latents_emb(lat_t)
                seq = torch.cat((frames_emb, act_emb, lat_emb), dim=2)
                seq_flat = rearrange(seq, 'b t p e -> b (t p) e')
                capture.clear()
                final = wm.transformer(seq_flat, use_kv_cache=False)
                capture[num_layers] = final.detach()
            if target == 0:
                hs = (capture[0] if 0 in capture else seq_flat).reshape(1, L, tokens_per_block, embed_dim)
            else:
                hs = capture[target].reshape(1, L, tokens_per_block, embed_dim)
            summary = hs[0, :, 2:2+K].mean(dim=1).cpu().numpy()       # (L, embed_dim)
            out_arrs.append(summary)
            label_idx.extend(range(global_start + s, global_start + e))
            steps_done += L
        if (ep + 1) % max(1, n_ep // 10) == 0 or ep == n_ep - 1:
            print(f"  ep {ep+1}/{n_ep}  steps {steps_done}  "
                  f"{steps_done/max(time.time()-t0,1):.0f} st/s", flush=True)
    for h in hooks: h.remove()
    return np.concatenate(out_arrs, axis=0).astype(np.float32), np.array(label_idx, dtype=np.int64)


# ---------- main ------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--rollouts", required=True, type=Path)
    ap.add_argument("--obs", required=True, type=Path)
    ap.add_argument("--cavs", type=Path, default=None, help="cavs.npz from cav_traces.py")
    ap.add_argument("--codebook-stats", type=Path, default=None,
                    help="codebook_stats.npz from codebook_stats.py")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--layer", default="wm_block_1",
                    choices=["wm_input", "wm_block_1", "wm_block_2", "wm_block_3"])
    ap.add_argument("--n-features", type=int, default=2048)
    ap.add_argument("--k-active", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--n-top-samples", type=int, default=12)
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

    rollouts = np.load(args.rollouts, allow_pickle=True)
    aj = rollouts["ach_just_unlocked"]
    actions = rollouts["actions"].astype(np.int64)
    tokens = rollouts["tokens"]
    ep_ids = rollouts["episode_ids"].astype(np.int64)
    ach_names = [str(x) for x in rollouts["achievement_names"]]

    # --- extract activations ---------------------------------------------
    print(f"extracting {args.layer} activations from agent…", flush=True)
    X, label_idx = extract_layer(agent, rollouts, args.obs, args.layer, device)
    N, D = X.shape
    print(f"  X = {N} × {D}  ({X.nbytes/1e6:.1f} MB)", flush=True)

    # --- train SAE -------------------------------------------------------
    sae = TopKSAE(d_in=D, d_features=args.n_features, k=args.k_active).to(device)
    sae.b_dec.data.copy_(torch.from_numpy(X.mean(0)).to(device))
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr)
    X_t = torch.from_numpy(X).to(device)

    # episode-disjoint split for val
    sel_ep = ep_ids[label_idx]
    rng = np.random.default_rng(0)
    uniq = np.unique(sel_ep); rng.shuffle(uniq)
    n_tr = int(len(uniq) * 0.9)
    train_eps = set(uniq[:n_tr].tolist())
    is_tr_mask = np.isin(sel_ep, list(train_eps))
    train_idx = np.where(is_tr_mask)[0]
    val_idx   = np.where(~is_tr_mask)[0]
    print(f"  train {train_idx.shape[0]}  val {val_idx.shape[0]}", flush=True)

    norm_var = float(X[train_idx].var())  # used to normalise reconstruction loss to "fraction of variance unexplained"

    print(f"training TopK SAE ({args.n_features} features, k={args.k_active}, lr={args.lr})", flush=True)
    t0 = time.time()
    for epoch in range(args.epochs):
        sae.train()
        perm = rng.permutation(train_idx)
        running_mse = 0.0; n = 0
        for start in range(0, perm.shape[0], args.batch_size):
            b = perm[start:start+args.batch_size]
            x = X_t[b]
            x_hat, z = sae(x)
            mse = F.mse_loss(x_hat, x)
            opt.zero_grad(); mse.backward(); opt.step()
            sae._normalize_decoder()
            running_mse += mse.item() * b.shape[0]; n += b.shape[0]
        sae.eval()
        with torch.no_grad():
            x = X_t[val_idx]
            x_hat, z = sae(x)
            val_mse = F.mse_loss(x_hat, x).item()
            val_fvu = val_mse / max(norm_var, 1e-8)
            # density per feature
            density = (z > 0).float().mean(0)
            n_dead = int((density == 0).sum().item())
            n_used = args.n_features - n_dead
        if (epoch + 1) % max(1, args.epochs // 10) == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch+1:3d}/{args.epochs}  train MSE={running_mse/n:.5f}  "
                  f"val MSE={val_mse:.5f}  FVU={val_fvu:.3f}  alive_features={n_used}/{args.n_features}",
                  flush=True)

    torch.save({"state_dict": sae.state_dict(),
                "config": dict(d_in=D, d_features=args.n_features, k=args.k_active,
                               layer=args.layer)},
               args.out / "sae.pt")
    print(f"saved {args.out / 'sae.pt'}", flush=True)

    # --- analyze features -----------------------------------------------
    print("analyzing features…", flush=True)
    sae.eval()
    with torch.no_grad():
        # encode all data in chunks
        Z = torch.zeros(N, args.n_features, device="cpu")
        for s in range(0, N, args.batch_size * 4):
            e = min(s + args.batch_size * 4, N)
            Z[s:e] = sae.encode(X_t[s:e]).cpu()
    Z_np = Z.numpy()                                  # (N, n_features) sparse

    # per-feature stats
    n_active = (Z_np > 0).sum(0)                       # how many samples each feature fires on
    mean_act = np.where(n_active > 0, Z_np.sum(0) / np.maximum(n_active, 1), 0.0)
    max_act  = Z_np.max(0)

    # achievement alignment: P(any ach just unlocked | feature in top quartile)
    feat_ach_lift = np.full((args.n_features, len(ach_names)), np.nan, dtype=np.float32)
    feat_ach_pmax = np.full((args.n_features, len(ach_names)), np.nan, dtype=np.float32)
    base_unlock = aj[label_idx].mean(0)                # baseline rate per ach
    for f in range(args.n_features):
        if n_active[f] < 20: continue
        z_f = Z_np[:, f]
        # threshold: top quartile of nonzero activations
        nz = z_f[z_f > 0]
        thr = np.percentile(nz, 75)
        mask = z_f >= max(thr, 1e-6)
        if mask.sum() < 5: continue
        for a in range(len(ach_names)):
            p = aj[label_idx][mask, a].mean()
            feat_ach_pmax[f, a] = p
            feat_ach_lift[f, a] = p / max(base_unlock[a], 1e-8)

    # CAV alignment (cosine sim between SAE decoder columns and CAVs)
    cav_aligns = None
    if args.cavs is not None and args.cavs.exists():
        c = np.load(args.cavs, allow_pickle=True)
        rep_key = f"w_{args.layer}"
        if rep_key in c:
            W_cav = c[rep_key]                          # (n_concepts, D) in *standardised* space
            mu = c[f"mean_{args.layer}"]; sd = c[f"std_{args.layer}"]
            # un-standardise CAV: w' = W_cav / sd ; bias absorbed (we only need direction)
            W_cav_raw = W_cav / sd[None, :]
            W_cav_raw = W_cav_raw / (np.linalg.norm(W_cav_raw, axis=1, keepdims=True) + 1e-8)
            dec = sae.decoder.weight.detach().cpu().numpy()     # (D, n_features)
            dec_norm = dec / (np.linalg.norm(dec, axis=0, keepdims=True) + 1e-8)
            cav_aligns = W_cav_raw @ dec_norm                    # (n_concepts, n_features)
            concept_names = [str(x) for x in c["concept_names"]]
            print(f"  CAV alignment matrix: {cav_aligns.shape}", flush=True)
        else:
            concept_names = None
    else:
        concept_names = None

    # codebook alignment: mean activation per feature conditioned on (slot, code)
    codebook_aligns = None
    code_names = None
    if args.codebook_stats is not None and args.codebook_stats.exists():
        cs = np.load(args.codebook_stats)
        K = tokens.shape[1]; C = cs["counts"].shape[1]
        # for each (slot, code), find the feature with highest mean activation among samples with that token
        # only do for codes with at least 50 samples in the rollouts (not the full buffer)
        rollout_token_counts = np.zeros((K, C), dtype=np.int64)
        for k in range(K):
            np.add.at(rollout_token_counts[k], tokens[:, k], 1)
        feat_code_means = np.zeros((args.n_features, K, C), dtype=np.float32)
        for k in range(K):
            for code in np.where(rollout_token_counts[k] >= 50)[0]:
                mask = (tokens[label_idx, k] == code)
                if mask.sum() < 50: continue
                feat_code_means[:, k, code] = Z_np[mask].mean(0)
        codebook_aligns = feat_code_means
        print(f"  codebook alignment shape: {codebook_aligns.shape}", flush=True)

    # build per-feature records
    feats = []
    for f in range(args.n_features):
        n_a = int(n_active[f])
        if n_a < 5: continue
        top_idx = np.argpartition(-Z_np[:, f], min(args.n_top_samples, N))[:args.n_top_samples]
        top_idx = top_idx[np.argsort(-Z_np[top_idx, f])]
        rec = dict(
            feature=int(f),
            n_active=n_a,
            density=float(n_a / N),
            mean_act=float(mean_act[f]),
            max_act=float(max_act[f]),
            top_sample_idx=[int(i) for i in top_idx.tolist()],
            top_sample_act=[float(Z_np[i, f]) for i in top_idx.tolist()],
        )
        # best achievement match
        if not np.isnan(feat_ach_lift[f]).all():
            a_best = int(np.nanargmax(feat_ach_lift[f]))
            rec["best_ach"] = dict(name=ach_names[a_best],
                                   lift=float(feat_ach_lift[f, a_best]),
                                   p=float(feat_ach_pmax[f, a_best]))
        # best CAV match
        if cav_aligns is not None and concept_names is not None:
            c_best = int(np.argmax(np.abs(cav_aligns[:, f])))
            rec["best_cav"] = dict(name=concept_names[c_best],
                                   cos=float(cav_aligns[c_best, f]))
        # best (slot, code) match
        if codebook_aligns is not None:
            flat = codebook_aligns[f].flatten()
            k_best = int(np.argmax(flat))
            slot, code = k_best // codebook_aligns.shape[2], k_best % codebook_aligns.shape[2]
            rec["best_code"] = dict(slot=int(slot), code=int(code),
                                    mean_act=float(codebook_aligns[f, slot, code]))
        feats.append(rec)
    feats.sort(key=lambda r: -r["n_active"])

    (args.out / "features.json").write_text(json.dumps(dict(
        layer=args.layer, n_features=args.n_features, k_active=args.k_active,
        N=N, D=D, features=feats), default=lambda o: int(o) if isinstance(o, np.integer) else float(o)))
    print(f"wrote features.json  ({len(feats)} live features)", flush=True)

    # --- render HTML ----------------------------------------------------
    parts = [
        "<!doctype html><meta charset=utf-8><title>SAE features</title>",
        "<style>"
        "body{font:13px ui-monospace,Menlo,Consolas,monospace;background:#0d1117;color:#c9d1d9;padding:18px;}"
        "h1{color:#79c0ff;font-size:18px;margin:0 0 8px;}"
        "table{width:100%;border-collapse:collapse;font-size:12px;}"
        "th,td{padding:5px 8px;text-align:left;border-bottom:1px solid #21262d;}"
        "th{color:#8b949e;text-transform:uppercase;font-size:11px;cursor:pointer;}"
        ".dim{color:#8b949e;} .hi{color:#3fb950;font-weight:600;}"
        ".controls{margin:12px 0;padding:10px;background:#161b22;border-radius:6px;border:1px solid #30363d;}"
        ".controls input,.controls select{background:#0d1117;color:#c9d1d9;border:1px solid #30363d;padding:4px 8px;font:inherit;border-radius:3px;}"
        "</style>",
        f"<h1>Δ-IRIS SAE features ({args.layer} · {args.n_features} dict · k={args.k_active})</h1>",
        f"<div class=dim>N={N} samples · live features = {len(feats)} · "
        f"sorted by n_active. Click headers to re-sort.</div>",
        "<div class=controls>",
        " min density <input id=minD type=number value=0.001 step=0.001 style='width:5em'>",
        " concept search <input id=q placeholder='e.g. iron' style='width:10em'>",
        " <button id=apply>apply</button>",
        "</div>",
        "<table id=t><thead><tr>"
        "<th>feature</th><th>n_active</th><th>density</th><th>mean_act</th>"
        "<th>best achievement (lift / P)</th>"
        "<th>best CAV (cos)</th>"
        "<th>best (slot,code) (mean_act)</th>"
        "</tr></thead><tbody></tbody></table>",
        "<script>",
        f"const FEATS = {json.dumps(feats)};",
        "let rows=[]; function apply() {",
        " const minD=parseFloat(document.getElementById('minD').value)||0;",
        " const q=document.getElementById('q').value.toLowerCase();",
        " rows = FEATS.filter(f=>f.density>=minD);",
        " if (q) rows = rows.filter(f=>JSON.stringify(f).toLowerCase().includes(q));",
        " render();",
        "} function render() {",
        " const tb=document.querySelector('#t tbody'); tb.innerHTML='';",
        " for (const f of rows.slice(0,500)) {",
        "  const ach=f.best_ach?`${f.best_ach.name} (${f.best_ach.lift.toFixed(1)}×, P=${(f.best_ach.p*100).toFixed(0)}%)`:'—';",
        "  const cav=f.best_cav?`${f.best_cav.name} (cos=${f.best_cav.cos.toFixed(2)})`:'—';",
        "  const code=f.best_code?`s${f.best_code.slot} c${f.best_code.code} (${f.best_code.mean_act.toFixed(2)})`:'—';",
        "  const tr=document.createElement('tr');",
        "  tr.innerHTML=`<td>${f.feature}</td><td>${f.n_active}</td><td>${f.density.toExponential(2)}</td><td>${f.mean_act.toFixed(2)}</td>",
        "    <td class='${f.best_ach&&f.best_ach.lift>5?\"hi\":\"\"}'>${ach}</td>",
        "    <td class='${f.best_cav&&Math.abs(f.best_cav.cos)>0.4?\"hi\":\"\"}'>${cav}</td>",
        "    <td>${code}</td>`;",
        "  tb.appendChild(tr);",
        " }",
        "}",
        "document.getElementById('apply').addEventListener('click',apply);",
        "['minD','q'].forEach(id=>document.getElementById(id).addEventListener('change',apply));",
        "apply();",
        "</script>",
    ]
    (args.out / "features.html").write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {args.out / 'features.html'}", flush=True)


if __name__ == "__main__":
    main()
