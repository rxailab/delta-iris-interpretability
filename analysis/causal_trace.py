"""Causal tracing (activation patching) over the Δ-IRIS world-model transformer.

For each target concept:
  clean window    = 21 real steps ending at an unlock moment T
                    (target = the 4 real codes at step T, which include the
                     concept's detector code)
  corrupt window  = 21 real steps from the SAME episode ending at an ordinary
                    moment (concept not unlocking at its final step)

  1. Clean forward: record every block's output (3 blocks x 126 positions),
     and logP_clean(target codes at final step).
  2. Corrupt forward: logP_corrupt(same target codes) -- should be low.
  3. For each cell (block L, step s, token-group g in {frame, act, latents}):
     run the corrupt input, but at block L's output overwrite the positions of
     (s, g) with the CLEAN activations. Measure logP_patched.
     restoration = (logP_patched - logP_corrupt) / (logP_clean - logP_corrupt)

A cell with high restoration carries the causal signal for predicting the
unlock codes. Batched over trials: one forward per cell with batch = n trials.

Outputs under --out:
  trace_results.json   raw mean restoration per (concept, block, step, group)
  causal_trace.html    heatmaps per concept
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import OmegaConf


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


def build_seq(wm, obs_t, act_t, lat_t):
    """(B, W, ...) -> flat WM input (B, W*tpb, D)."""
    frames_emb = wm.frame_cnn(obs_t)
    a_emb = wm.act_emb(act_t).unsqueeze(2)
    l_emb = wm.latents_emb(lat_t)
    seq = torch.cat([frames_emb, a_emb, l_emb], dim=2)
    return rearrange(seq, "b t p d -> b (t p) d")


def logp_codes_final_step(wm_out, target_codes: torch.Tensor, window: int) -> torch.Tensor:
    """Per-sample logP of the 4 target codes at the final step. target_codes: (B, K)."""
    logits = wm_out.logits_latents
    B, K = target_codes.shape
    if logits.dim() == 4:
        per_slot = logits[:, window - 1]                      # (B, K, V)
    else:
        T = logits.size(1) // K
        per_slot = logits.view(B, T, K, -1)[:, window - 1]    # (B, K, V)
    lp = F.log_softmax(per_slot.float(), dim=-1)
    return lp.gather(-1, target_codes.long().unsqueeze(-1)).squeeze(-1).sum(-1)   # (B,)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--rollouts", required=True, type=Path)
    ap.add_argument("--obs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--window", type=int, default=21)
    ap.add_argument("--n-trials", type=int, default=24)
    ap.add_argument("--concepts", default="wake_up,collect_coal,defeat_skeleton,collect_sapling")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    run = args.run.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    print("loading agent...", flush=True)
    agent = load_agent(run, device)
    wm = agent.world_model
    tpb = wm.config.transformer_config.tokens_per_block        # 6
    n_blocks = wm.config.transformer_config.num_layers          # 3
    W = args.window

    r = np.load(args.rollouts, allow_pickle=True)
    obs_npz = np.load(args.obs)
    all_obs = obs_npz["obs"]; obs_starts = obs_npz["episode_starts"]
    tokens = r["tokens"]; actions = r["actions"].astype(np.int64)
    aj = r["ach_just_unlocked"]; ep_ids = r["episode_ids"].astype(np.int64)
    ep_lens = r["episode_lengths"].astype(np.int64)
    ach_names = [str(x) for x in r["achievement_names"]]
    ep_start_row = np.concatenate(([0], np.cumsum(ep_lens)))

    groups = {"frame": [0], "act": [1], "latents": [2, 3, 4, 5]}
    concepts = [c.strip() for c in args.concepts.split(",") if c.strip()]
    results = {}

    # hooks: capture clean activations / patch positions
    capture: dict[int, torch.Tensor] = {}
    patch_cfg = {"block": None, "positions": None, "clean": None}

    def make_capture_hook(i):
        def h(_m, _inp, out):
            capture[i] = out.detach()
            return out
        return h

    def make_patch_hook(i):
        def h(_m, _inp, out):
            if patch_cfg["block"] != i: return out
            pos = patch_cfg["positions"]
            new = out.clone()
            new[:, pos] = patch_cfg["clean"][i][:, pos]
            return new
        return h

    t0 = time.time()
    for concept in concepts:
        a_idx = ach_names.index(concept)

        # gather trial windows
        clean_windows, corrupt_windows, target_codes_list = [], [], []
        unlock_global = [g for g in np.where(aj[:, a_idx])[0]
                         if (g - ep_start_row[ep_ids[g]]) >= W - 1]
        rng.shuffle(unlock_global)
        for g in unlock_global:
            if len(clean_windows) >= args.n_trials: break
            ep = int(ep_ids[g]); g_local = int(g - ep_start_row[ep])
            T_ep = int(ep_lens[ep])
            # corrupt: ordinary end-step in same episode, >= W-1 history, no unlock of this concept at end
            cand = [s for s in range(W - 1, T_ep)
                    if not aj[ep_start_row[ep] + s, a_idx] and abs(s - g_local) >= W]
            if not cand: continue
            s_cor = int(rng.choice(cand))

            def window(ep, end_local):
                lo = end_local - W + 1
                o = all_obs[obs_starts[ep] + lo: obs_starts[ep] + end_local + 1]   # W frames
                a = actions[ep_start_row[ep] + lo: ep_start_row[ep] + end_local + 1]
                l = tokens[ep_start_row[ep] + lo: ep_start_row[ep] + end_local + 1]
                return o, a, l

            o_c, a_c, l_c = window(ep, g_local)
            o_x, a_x, l_x = window(ep, s_cor)
            clean_windows.append((o_c, a_c, l_c))
            corrupt_windows.append((o_x, a_x, l_x))
            target_codes_list.append(l_c[-1])          # real codes at unlock step

        B = len(clean_windows)
        if B < 5:
            print(f"  {concept}: only {B} usable trials, skipping"); continue
        print(f"[{concept}] {B} trials", flush=True)

        def stack(ws):
            o = torch.from_numpy(np.stack([w[0] for w in ws])).to(device).float().div(255)
            a = torch.from_numpy(np.stack([w[1] for w in ws])).to(device)
            l = torch.from_numpy(np.stack([w[2] for w in ws]).astype(np.int64)).to(device)
            return o, a, l

        o_clean, a_clean, l_clean = stack(clean_windows)
        o_cor, a_cor, l_cor = stack(corrupt_windows)
        tgt = torch.from_numpy(np.stack(target_codes_list).astype(np.int64)).to(device)

        cap_handles = [blk.register_forward_hook(make_capture_hook(i))
                       for i, blk in enumerate(wm.transformer.blocks)]
        with torch.no_grad():
            out_clean = wm(build_seq(wm, o_clean, a_clean, l_clean))
        clean_acts = {i: capture[i].clone() for i in range(n_blocks)}
        for h in cap_handles: h.remove()
        lp_clean = logp_codes_final_step(out_clean, tgt, W)

        with torch.no_grad():
            out_cor = wm(build_seq(wm, o_cor, a_cor, l_cor))
        lp_cor = logp_codes_final_step(out_cor, tgt, W)
        denom = (lp_clean - lp_cor)
        print(f"  logp clean {lp_clean.mean():.2f}  corrupt {lp_cor.mean():.2f}  "
              f"gap {denom.mean():.2f}", flush=True)

        patch_handles = [blk.register_forward_hook(make_patch_hook(i))
                         for i, blk in enumerate(wm.transformer.blocks)]
        patch_cfg["clean"] = clean_acts
        grid = np.zeros((len(groups), n_blocks, W), dtype=np.float64)
        seq_cor = build_seq(wm, o_cor, a_cor, l_cor)
        for gi, (gname, offsets) in enumerate(groups.items()):
            for blk in range(n_blocks):
                for step in range(W):
                    patch_cfg["block"] = blk
                    patch_cfg["positions"] = [step * tpb + off for off in offsets]
                    with torch.no_grad():
                        out_p = wm(seq_cor)
                    lp_p = logp_codes_final_step(out_p, tgt, W)
                    restore = ((lp_p - lp_cor) / denom.clamp(min=1e-3)).clamp(-1, 2)
                    grid[gi, blk, step] = float(restore.mean().item())
        patch_cfg["block"] = None
        for h in patch_handles: h.remove()

        results[concept] = dict(
            n=B,
            lp_clean=float(lp_clean.mean()), lp_corrupt=float(lp_cor.mean()),
            groups=list(groups.keys()),
            grid=grid.tolist(),
        )
        print(f"  done ({(time.time()-t0)/60:.1f} min total)", flush=True)

    (args.out / "trace_results.json").write_text(json.dumps(results, indent=2))

    # ----- render ------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    imgs = []
    for concept, res in results.items():
        grid = np.array(res["grid"])               # (3 groups, n_blocks, W)
        fig, axes = plt.subplots(1, len(groups), figsize=(16, 2.6), sharey=True)
        for gi, gname in enumerate(res["groups"]):
            ax = axes[gi]
            im = ax.imshow(grid[gi], aspect="auto", cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"{gname} tokens", fontsize=10)
            ax.set_xlabel("step in window (20 = unlock step)")
            ax.set_yticks(range(n_blocks)); ax.set_yticklabels([f"block {i}" for i in range(n_blocks)])
        fig.colorbar(im, ax=axes, label="restoration", fraction=0.02)
        fig.suptitle(f"{concept}  (n={res['n']}, logp clean {res['lp_clean']:.1f} vs corrupt {res['lp_corrupt']:.1f})",
                     fontsize=11)
        buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=130, bbox_inches="tight"); plt.close(fig)
        imgs.append((concept, base64.b64encode(buf.getvalue()).decode("ascii")))

    html = ("<!doctype html><meta charset=utf-8><title>causal tracing</title>"
            "<style>body{font:14px ui-monospace,Menlo,Consolas,monospace;background:#0d1117;"
            "color:#c9d1d9;padding:18px;}h1{color:#79c0ff;font-size:18px;}h2{color:#d2a8ff;font-size:15px;}"
            ".dim{color:#8b949e;}img{max-width:100%;background:white;border-radius:6px;margin-bottom:10px;}</style>"
            "<h1>Causal tracing: which (block, step, token-group) carries the unlock signal?</h1>"
            "<div class=dim>Activation patching: corrupt window from same episode, patch clean "
            "activations per cell, measure restoration of logP(real unlock codes at final step). "
            "restoration 1.0 = patching that single cell fully recovers the clean prediction.</div>"
            + "".join(f"<h2>{c}</h2><img src='data:image/png;base64,{i}'>" for c, i in imgs))
    (args.out / "causal_trace.html").write_text(html, encoding="utf-8")
    print(f"wrote {args.out/'causal_trace.html'}", flush=True)


if __name__ == "__main__":
    main()
