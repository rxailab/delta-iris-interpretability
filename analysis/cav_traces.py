"""Extract CAVs (Concept Activation Vectors) for strong probe-concepts and
render their activation traces over time for showcase episodes.

Pipeline:
  1. Reload rollouts + obs (output of rollout_with_info.py --save-obs).
  2. Run the agent's world model on each episode chunk-by-chunk; capture
     hidden states at every transformer block at the "summary position" of
     each timestep (avg of the 4 latent positions).
  3. For each (concept, layer) train a logistic regression on episode-disjoint
     train/test splits; the CAV is the weight vector. Drop probes whose
     test AUROC < --min-auroc.
  4. Save CAVs + standardisation stats as cavs.npz.
  5. For the N most achievement-diverse episodes, project hidden states onto
     each surviving CAV → concept-activation trace; overlay actual unlock
     events; emit traces.html.

Outputs (under --out):
  cavs.npz                        # weights + bias + mean/std, all reps
  cav_metrics.json                # auroc / acc per (concept, rep)
  traces.html                     # showcase episodes with concept traces
"""
from __future__ import annotations

import argparse, base64, io, json, sys, time
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def extract_activations(agent, run, rollouts, obs_path, device):
    """Return dict {rep_name: (N, D) np.float32}, label_index (N,)."""
    wm, tk = agent.world_model, agent.tokenizer
    num_layers = wm.config.transformer_config.num_layers
    embed_dim = wm.config.transformer_config.embed_dim
    tokens_per_block = wm.config.transformer_config.tokens_per_block
    max_blocks = wm.config.transformer_config.max_blocks

    obs_npz = np.load(obs_path)
    obs = obs_npz["obs"]
    obs_starts = obs_npz["episode_starts"]
    actions = rollouts["actions"].astype(np.int64)
    tokens = rollouts["tokens"]
    ep_ids = rollouts["episode_ids"].astype(np.int64)
    ep_lens = rollouts["episode_lengths"].astype(np.int64)
    n_ep = ep_lens.shape[0]
    K = tokens.shape[1]

    capture: dict[int, torch.Tensor] = {}
    hooks = []
    for i, block in enumerate(wm.transformer.blocks):
        def mk(i=i):
            def h(_m, _i, o): capture[i] = o.detach()
            return h
        hooks.append(block.register_forward_hook(mk()))

    rep_names = ["raw_codes", "wm_input"] + [f"wm_block_{i+1}" for i in range(num_layers)]
    rep_arrs: dict[str, list] = {n: [] for n in rep_names}
    label_idx: list[int] = []
    t0 = time.time()
    steps_done = 0
    for ep in range(n_ep):
        T = int(ep_lens[ep])
        ep_obs = obs[obs_starts[ep]:obs_starts[ep+1]]
        ep_act = actions[ep_ids == ep]
        ep_tok = tokens[ep_ids == ep]
        global_start = int(np.flatnonzero(ep_ids == ep)[0])
        for s in range(0, T, max_blocks):
            e = min(s + max_blocks, T)
            L = e - s
            obs_t = torch.from_numpy(ep_obs[s:e+1]).to(device).float().div(255).unsqueeze(0)
            act_t = torch.from_numpy(ep_act[s:e]).to(device).unsqueeze(0)
            lat_t = torch.from_numpy(ep_tok[s:e].astype(np.int64)).to(device).unsqueeze(0)
            with torch.no_grad():
                frames_emb = wm.frame_cnn(obs_t[:, :L])
                act_emb = wm.act_emb(act_t).unsqueeze(2)
                lat_emb = wm.latents_emb(lat_t)
                seq = torch.cat((frames_emb, act_emb, lat_emb), dim=2)
                seq_flat = rearrange(seq, 'b t p e -> b (t p) e')
                cb_emb = tk.quantizer.codebook[lat_t.squeeze(0).reshape(-1)].reshape(L, K, -1)
                raw_codes_feat = cb_emb.mean(dim=1).cpu().numpy()
                capture.clear()
                final = wm.transformer(seq_flat, use_kv_cache=False)
                capture[num_layers] = final.detach()
            for li in range(num_layers + 1):
                hs = capture[li][0].reshape(L, tokens_per_block, embed_dim)
                summary = hs[:, 2:2+K].mean(dim=1).cpu().numpy()
                if li == 0:
                    rep_arrs["wm_input"].append(summary)
                else:
                    rep_arrs[f"wm_block_{li}"].append(summary)
            rep_arrs["raw_codes"].append(raw_codes_feat)
            label_idx.extend(range(global_start + s, global_start + e))
            steps_done += L
        if (ep + 1) % max(1, n_ep // 10) == 0 or ep == n_ep - 1:
            print(f"  ep {ep+1}/{n_ep}  steps {steps_done}  "
                  f"{steps_done/max(time.time()-t0,1):.0f} st/s", flush=True)
    for h in hooks: h.remove()
    reps = {n: np.concatenate(arrs, axis=0).astype(np.float32) for n, arrs in rep_arrs.items()}
    return reps, np.array(label_idx, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--rollouts", required=True, type=Path)
    ap.add_argument("--obs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-auroc", type=float, default=0.85)
    ap.add_argument("--trace-rep", default="wm_block_1",
                    help="which rep to use for the time-series trace plots")
    ap.add_argument("--n-showcase", type=int, default=5)
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

    r = np.load(args.rollouts, allow_pickle=True)
    aj = r["ach_just_unlocked"]; acum = r["ach_cumulative"]
    ach_names = [str(x) for x in r["achievement_names"]]
    ep_ids = r["episode_ids"].astype(np.int64)
    ep_lens = r["episode_lengths"].astype(np.int64)

    print("extracting activations…", flush=True)
    reps, label_idx = extract_activations(agent, run, r, args.obs, device)
    print(f"got {label_idx.shape[0]} samples × {len(reps)} reps", flush=True)

    sel_ep = ep_ids[label_idx]
    # episode-disjoint 80/20 split
    rng = np.random.default_rng(0)
    uniq = np.unique(sel_ep); rng.shuffle(uniq)
    n_tr = int(len(uniq) * 0.8)
    train_eps = set(uniq[:n_tr].tolist())
    is_tr = np.isin(sel_ep, list(train_eps))

    # ----- fit CAVs (only for ach_just / ach_cum, our event/state concepts) ----
    concepts: list[tuple[str, np.ndarray]] = []
    for i, n in enumerate(ach_names):
        concepts.append((f"just[{n}]", aj[label_idx, i].astype(np.int64)))
        concepts.append((f"cum[{n}]",  acum[label_idx, i].astype(np.int64)))

    rep_names = list(reps.keys())
    embed_dim = max(v.shape[1] for v in reps.values())  # 512 for WM reps; codebook_dim=64 for raw_codes
    cav_w = {rep: {} for rep in rep_names}      # rep -> concept -> w
    cav_b = {rep: {} for rep in rep_names}
    rep_mean = {rep: reps[rep].mean(0) for rep in rep_names}
    rep_std  = {rep: reps[rep].std(0) + 1e-6 for rep in rep_names}
    metrics: list[dict] = []
    for rep in rep_names:
        Xtr = (reps[rep][is_tr] - rep_mean[rep]) / rep_std[rep]
        Xte = (reps[rep][~is_tr] - rep_mean[rep]) / rep_std[rep]
        for cname, y in concepts:
            ytr, yte = y[is_tr], y[~is_tr]
            if yte.sum() < 5 or yte.sum() == yte.shape[0]:
                metrics.append(dict(concept=cname, rep=rep, auroc=None, skipped=True,
                                    n_pos_test=int(yte.sum())))
                continue
            clf = LogisticRegression(max_iter=200, class_weight="balanced", C=1.0, solver="lbfgs")
            clf.fit(Xtr, ytr)
            score = clf.decision_function(Xte)
            try:
                au = float(roc_auc_score(yte, score))
            except Exception:
                au = None
            metrics.append(dict(concept=cname, rep=rep, auroc=au,
                                n_pos_test=int(yte.sum())))
            cav_w[rep][cname] = clf.coef_[0].astype(np.float32)
            cav_b[rep][cname] = float(clf.intercept_[0])
    print(f"fit {sum(len(v) for v in cav_w.values())} CAVs across {len(rep_names)} reps")

    (args.out / "cav_metrics.json").write_text(json.dumps(metrics, indent=2))

    # ----- save CAV array -------------------------------------------------
    # Save per-rep separately because embed_dim differs (raw_codes=64, WM=512)
    np.savez_compressed(args.out / "cavs.npz",
        rep_names=np.array(rep_names),
        concept_names=np.array([c[0] for c in concepts]),
        **{f"w_{rep}": np.stack([cav_w[rep].get(cn, np.zeros(reps[rep].shape[1], dtype=np.float32))
                                  for cn, _ in concepts], axis=0) for rep in rep_names},
        **{f"b_{rep}": np.array([cav_b[rep].get(cn, 0.0) for cn, _ in concepts], dtype=np.float32) for rep in rep_names},
        **{f"mean_{rep}": rep_mean[rep] for rep in rep_names},
        **{f"std_{rep}":  rep_std[rep]  for rep in rep_names},
    )
    print(f"saved CAVs → {args.out / 'cavs.npz'}")

    # ----- pick showcase episodes by achievement diversity ----------------
    # per-episode unique achievements count
    ep_div = np.zeros(ep_lens.shape[0], dtype=np.int64)
    for ep in range(ep_lens.shape[0]):
        rows = ep_ids == ep
        ep_div[ep] = aj[rows].any(axis=0).sum()
    # also bias toward long episodes
    score = ep_div * np.log(1 + ep_lens)
    showcase = np.argsort(-score)[:args.n_showcase]
    print(f"showcase episodes: {showcase.tolist()}  (div: {ep_div[showcase].tolist()}  len: {ep_lens[showcase].tolist()})")

    # ----- choose concepts to plot (strong CAVs at trace_rep, deduped just/cum)
    strong = sorted(
        [m for m in metrics if m["rep"] == args.trace_rep and m.get("auroc") and m["auroc"] >= args.min_auroc],
        key=lambda m: -m["auroc"]
    )
    seen, picked = set(), []
    for m in strong:
        base = m["concept"].split("[")[1].rstrip("]")
        if base in seen: continue
        seen.add(base); picked.append(m)
        if len(picked) >= 6: break
    print(f"will plot {len(picked)} concept traces: {[p['concept'] for p in picked]}")

    # ----- render traces --------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rep = args.trace_rep
    H = reps[rep]
    mu, sd = rep_mean[rep], rep_std[rep]

    plots = []
    for ep in showcase:
        rows = np.where(sel_ep == ep)[0]
        if rows.size == 0: continue
        T = rows.size
        Hep = (H[rows] - mu) / sd                # (T, D)
        # achievements within this episode and the steps at which they fired
        ep_label_steps = label_idx[rows]
        ep_aj = aj[ep_label_steps]                # (T, 22)
        fired_ach = [(t, ach_names[i]) for t, i in zip(*np.where(ep_aj)) if True]

        fig, ax = plt.subplots(figsize=(12, 1.0 * len(picked) + 1))
        colors = plt.cm.tab10.colors
        # vertical lines for every actual unlock event
        for t, name in fired_ach:
            ax.axvline(t, color="#888", ls=":", lw=0.7, alpha=0.6)
            ax.text(t, len(picked) + 0.2, name, rotation=90, fontsize=7, color="#999",
                    ha="right", va="bottom")
        for i, m in enumerate(picked):
            cn = m["concept"]
            w = cav_w[rep][cn]; b = cav_b[rep][cn]
            trace = Hep @ w + b
            # min-max normalise to [0,1] for stacking, offset by i
            tr = (trace - trace.min()) / (trace.ptp() + 1e-6)
            ax.plot(np.arange(T), tr + i, color=colors[i % 10], lw=1.0, label=cn)
        ax.set_yticks(np.arange(len(picked)) + 0.5)
        ax.set_yticklabels([p["concept"] for p in picked], fontsize=9)
        ax.set_xlabel("step"); ax.set_ylim(-0.1, len(picked) + 1.0)
        ax.set_title(f"episode {ep} · length {T} · {ep_div[ep]} achievements unlocked")
        ax.grid(axis="x", alpha=0.2); ax.spines["right"].set_visible(False); ax.spines["top"].set_visible(False)
        plt.tight_layout()
        buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=110); plt.close(fig)
        plots.append((int(ep), base64.b64encode(buf.getvalue()).decode("ascii")))

    # ----- HTML -----------------------------------------------------------
    parts = [
        "<!doctype html><meta charset=utf-8><title>CAV traces</title>",
        "<style>"
        "body{font:14px ui-monospace,Menlo,Consolas,monospace;background:#0d1117;color:#c9d1d9;padding:18px;}"
        "h1{color:#79c0ff;font-size:18px;margin:0 0 12px;}"
        "h2{color:#d2a8ff;font-size:15px;margin:18px 0 6px;}"
        ".dim{color:#8b949e;}"
        "img{max-width:100%;border:1px solid #21262d;border-radius:4px;background:white;}"
        "</style>",
        f"<h1>Δ-IRIS CAV activation traces (rep={rep})</h1>",
        f"<div class=dim>CAVs fit on episode-disjoint train split, then projected on "
        f"hidden states from held-out showcase episodes. Vertical dotted lines = "
        f"actual Crafter achievement unlocks at that step.</div>",
    ]
    for ep, img in plots:
        parts.append(f"<h2>episode {ep}</h2>"
                     f"<img src='data:image/png;base64,{img}'>")

    (args.out / "traces.html").write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote → {args.out / 'traces.html'}")


if __name__ == "__main__":
    main()
