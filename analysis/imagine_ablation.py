"""Multi-step causal ablation IN IMAGINATION.

For each target concept (achievement) with a strong SAE feature:
  1. Find unlock moments T in the rollout buffer (with >= burn-in history).
  2. Burn the world model in on the REAL history [T-8 .. T] (frames + actions).
  3. Force the REAL action a_T at the first imagined step, then let the
     actor-critic act for the remaining steps. Imagine `horizon` steps,
     sampling the 4 latent codes per step from head_latents and decoding
     frames with the tokenizer.
  4. Conditions (hook on WM transformer block 1, active ONLY during imagination):
       baseline : no intervention
       sae      : project out the SAE feature's decoder direction at every position
       cav      : project out the concept's CAV direction (w/std, renormed)
       random   : project out a random unit direction (specificity control)
  5. Outcome per rollout: was any of the concept's top-2 detector codes
     (from the MI table) sampled at the first imagined step / within horizon?
     Plus sum of predicted rewards.

Outputs under --out:
  imagine_trials.json      raw per-rollout records
  imagine_effects.json     aggregated P(detector) per (concept, condition) + CIs
  imagination.html         bar chart + table
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
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.distributions.categorical import Categorical


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


def detector_codes_from_mi(mi_table: Path, top_n: int = 2) -> dict[str, list[tuple[int, int]]]:
    """achievement -> top-N (slot, code) detectors by lift."""
    records = json.loads(mi_table.read_text())
    per: dict[str, list] = defaultdict(list)
    for r in records:
        per[r["achievement"]].append((r["lift"], r["slot"], r["code"]))
    out = {}
    for ach, rows in per.items():
        rows.sort(reverse=True)
        out[ach] = [(s, c) for _, s, c in rows[:top_n]]
    return out


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
    ap.add_argument("--burn-in", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--n-moments", type=int, default=20)
    ap.add_argument("--target-block", type=int, default=1)
    ap.add_argument("--concepts", default="",
                    help="comma-separated achievement names; empty = all with strong SAE feature")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    run = args.run.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    print("loading agent...", flush=True)
    agent = load_agent(run, device)
    wm, tk, ac = agent.world_model, agent.tokenizer, agent.actor_critic
    max_blocks = wm.config.transformer_config.max_blocks
    assert args.burn_in + args.horizon <= max_blocks, \
        f"burn_in+horizon must fit the {max_blocks}-block KV cache without a flush"

    from envs.world_model_env import WorldModelEnv

    class RecordingWMEnv(WorldModelEnv):
        """WorldModelEnv that also returns the 4 sampled codes per step."""
        @torch.no_grad()
        def step_recorded(self, action):
            assert self.world_model.transformer.num_blocks_left_in_kv_cache > 1, \
                "KV cache about to flush mid-imagination; reduce horizon"
            if not isinstance(action, torch.Tensor):
                action = torch.tensor(action, dtype=torch.long).reshape(-1, 1).to(self.device)
            a = self.world_model.act_emb(action)
            if self.last_latent_token_emb is None:
                inp = a if self.x is None else torch.cat((self.x, a), dim=1)
            else:
                inp = torch.cat((self.last_latent_token_emb, self.x, a), dim=1)
            outputs_wm = self.world_model(inp, use_kv_cache=True)

            from utils import compute_softmax_over_buckets, symexp
            reward = symexp(compute_softmax_over_buckets(outputs_wm.logits_rewards)) \
                if self.world_model.config.two_hot_rews \
                else Categorical(logits=outputs_wm.logits_rewards).sample().float() - 1
            reward = float(reward.flatten()[0].item())

            latent_tokens = []
            latent_token = Categorical(logits=outputs_wm.logits_latents).sample()
            latent_tokens.append(latent_token)
            for _ in range(self.tokenizer.config.num_tokens - 1):
                emb = self.world_model.latents_emb(latent_token)
                outputs_wm = self.world_model(emb, use_kv_cache=True)
                latent_token = Categorical(logits=outputs_wm.logits_latents).sample()
                latent_tokens.append(latent_token)
            self.last_latent_token_emb = self.world_model.latents_emb(latent_token)

            from einops import rearrange as rea
            q = self.tokenizer.quantizer.embed_tokens(torch.stack(latent_tokens, dim=-1))
            self.obs = self.tokenizer.decode(
                self.obs, action,
                rea(q, 'b t (h w) (k l e) -> b t e (h k) (w l)',
                    h=self.tokenizer.tokens_grid_res,
                    k=self.tokenizer.token_res, l=self.tokenizer.token_res),
                should_clamp=True)
            self.x = rea(self.world_model.frame_cnn(self.obs), 'b 1 k e -> b k e')
            codes = [int(t.flatten()[0].item()) for t in latent_tokens]  # [c0, c1, c2, c3]
            return codes, reward, self.obs

    wm_env = RecordingWMEnv(tk, wm, device)

    # ----- directions -------------------------------------------------------
    sae_blob = torch.load(args.sae, map_location=device, weights_only=False)
    decoder = sae_blob["state_dict"]["decoder.weight"].to(device).float()   # (D, n_feat)
    feats = json.loads(args.features.read_text())["features"]
    best_feat: dict[str, dict] = {}
    for f in feats:
        a = f.get("best_ach")
        if not a or f["density"] < 0.001: continue
        cur = best_feat.get(a["name"])
        if cur is None or a["lift"] > cur["lift"]:
            best_feat[a["name"]] = dict(feat=f["feature"], lift=a["lift"], p=a["p"])

    cav_npz = np.load(args.cavs, allow_pickle=True)
    cav_names = [str(x) for x in cav_npz["concept_names"]]
    layer_key = "wm_block_1"
    W_cav = cav_npz[f"w_{layer_key}"]; sd_cav = cav_npz[f"std_{layer_key}"]

    def cav_direction(ach: str) -> torch.Tensor | None:
        name = f"just[{ach}]"
        if name not in cav_names: return None
        w = W_cav[cav_names.index(name)]
        if not np.any(w): return None
        d = w / sd_cav
        return torch.from_numpy(d).to(device).float()

    detectors = detector_codes_from_mi(args.mi_table, top_n=2)

    # ----- rollout data -----------------------------------------------------
    r = np.load(args.rollouts, allow_pickle=True)
    obs_npz = np.load(args.obs)
    all_obs = obs_npz["obs"]; obs_starts = obs_npz["episode_starts"]
    tokens = r["tokens"]; actions = r["actions"].astype(np.int64)
    aj = r["ach_just_unlocked"]; ep_ids = r["episode_ids"].astype(np.int64)
    ep_lens = r["episode_lengths"].astype(np.int64)
    ach_names = [str(x) for x in r["achievement_names"]]
    ep_start_row = np.concatenate(([0], np.cumsum(ep_lens)))

    if args.concepts:
        wanted = [c.strip() for c in args.concepts.split(",") if c.strip()]
    else:
        wanted = [c for c in best_feat if c in detectors]
    targets = []
    for c in wanted:
        if c not in best_feat or c not in detectors:
            print(f"  skipping {c}: no strong SAE feature or no detector codes"); continue
        targets.append(c)
    print(f"target concepts ({len(targets)}): {targets}", flush=True)

    # ----- the intervention hook -------------------------------------------
    hook_state = {"direction": None}

    def hook(_m, _i, output):
        d = hook_state["direction"]
        if d is None: return output
        dn = F.normalize(d, dim=-1)
        scores = output @ dn                         # (..., T)
        return output - scores.unsqueeze(-1) * dn

    handle = wm.transformer.blocks[args.target_block].register_forward_hook(hook)

    # ----- run --------------------------------------------------------------
    trials = []
    t0 = time.time()
    torch.manual_seed(args.seed)
    for ci, concept in enumerate(targets):
        a_idx = ach_names.index(concept)
        det = detectors[concept]                     # [(slot, code), ...]
        d_sae = decoder[:, best_feat[concept]["feat"]].clone()
        d_cav = cav_direction(concept)
        d_rand = torch.randn_like(d_sae)

        unlock_global = [g for g in np.where(aj[:, a_idx])[0]
                         if (g - ep_start_row[ep_ids[g]]) >= args.burn_in]
        if len(unlock_global) < 3:
            print(f"  {concept}: too few unlock moments, skipping"); continue
        sel = rng.choice(unlock_global, size=min(args.n_moments, len(unlock_global)), replace=False)

        conditions = [("baseline", None), ("sae", d_sae), ("random", d_rand)]
        if d_cav is not None:
            conditions.insert(2, ("cav", d_cav))

        for g in sel:
            ep = int(ep_ids[g]); g_local = int(g - ep_start_row[ep])
            ep_obs = all_obs[obs_starts[ep]:obs_starts[ep+1]]
            ep_act = actions[ep_start_row[ep]:ep_start_row[ep+1]]
            lo = g_local - args.burn_in
            # burn-in: frames lo..g_local (burn_in+1 frames), actions lo..g_local-1
            burn_obs = torch.from_numpy(ep_obs[lo:g_local+1]).to(device).float().div(255).unsqueeze(0)
            burn_act = torch.from_numpy(ep_act[lo:g_local]).to(device).unsqueeze(0)
            real_a_T = int(ep_act[g_local])

            for cond_name, direction in conditions:
                hook_state["direction"] = None        # burn-in is always clean
                wm_env.reset_from_past(burn_obs.clone(), burn_act.clone())
                ac.reset(n=1)
                with torch.no_grad():
                    for t in range(args.burn_in):
                        _ = ac.act(burn_obs[:, t])
                hook_state["direction"] = direction   # intervention ON for imagination

                step_codes, rewards = [], []
                action = real_a_T                     # force real action at unlock step
                with torch.no_grad():
                    for h in range(args.horizon):
                        codes, rew, frame = wm_env.step_recorded(action)
                        step_codes.append(codes); rewards.append(rew)
                        act_tok, _ = ac.act(frame[:, 0], should_sample=True, temperature=1.0)
                        action = int(act_tok.item())
                hook_state["direction"] = None

                hit_steps = [h for h, codes in enumerate(step_codes)
                             if any(codes[s] == c for s, c in det)]
                trials.append(dict(
                    concept=concept, condition=cond_name,
                    episode=ep, ep_step=g_local,
                    detector_first=bool(hit_steps and hit_steps[0] == 0),
                    detector_within=bool(hit_steps),
                    first_hit_step=(hit_steps[0] if hit_steps else None),
                    reward_sum=float(np.sum(rewards)),
                ))

        n_done = len([t for t in trials if t["concept"] == concept])
        print(f"  [{ci+1}/{len(targets)}] {concept}: {n_done} rollouts  "
              f"({(time.time()-t0)/60:.1f} min elapsed)", flush=True)

    handle.remove()
    print(f"\n{len(trials)} imagination rollouts in {(time.time()-t0)/60:.1f} min", flush=True)
    (args.out / "imagine_trials.json").write_text(json.dumps(trials, indent=2))

    # ----- aggregate ----------------------------------------------------------
    def boot_ci(arr, B=1000):
        if len(arr) == 0: return 0.0, 0.0
        rng2 = np.random.default_rng(0)
        means = [rng2.choice(arr, size=len(arr), replace=True).mean() for _ in range(B)]
        return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    summary = []
    for concept in {t["concept"] for t in trials}:
        for cond in ["baseline", "sae", "cav", "random"]:
            sub = [t for t in trials if t["concept"] == concept and t["condition"] == cond]
            if not sub: continue
            first = np.array([t["detector_first"] for t in sub], dtype=float)
            within = np.array([t["detector_within"] for t in sub], dtype=float)
            rew = np.array([t["reward_sum"] for t in sub], dtype=float)
            lo_f, hi_f = boot_ci(first); lo_w, hi_w = boot_ci(within)
            summary.append(dict(concept=concept, condition=cond, n=len(sub),
                                p_first=float(first.mean()), p_first_lo=lo_f, p_first_hi=hi_f,
                                p_within=float(within.mean()), p_within_lo=lo_w, p_within_hi=hi_w,
                                reward_mean=float(rew.mean())))
    (args.out / "imagine_effects.json").write_text(json.dumps(summary, indent=2))

    # ----- render --------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    concepts = sorted({s["concept"] for s in summary})
    conds = ["baseline", "sae", "cav", "random"]
    colors = {"baseline": "#2ca02c", "sae": "#d62728", "cav": "#9467bd", "random": "#aaaaaa"}
    fig, axes = plt.subplots(1, 2, figsize=(2.2 + 1.1 * len(concepts), 5.2), sharey=True)
    for ax, metric, title in [
        (axes[0], "p_first", "P(detector code sampled at FIRST imagined step)"),
        (axes[1], "p_within", f"P(detector code within {args.horizon} imagined steps)"),
    ]:
        x = np.arange(len(concepts)); width = 0.2
        for j, cond in enumerate(conds):
            ys, ylo, yhi = [], [], []
            for c in concepts:
                rec = next((s for s in summary if s["concept"] == c and s["condition"] == cond), None)
                if rec:
                    ys.append(rec[metric])
                    ylo.append(rec[metric] - rec[f"{metric}_lo"])
                    yhi.append(rec[f"{metric}_hi"] - rec[metric])
                else:
                    ys.append(0); ylo.append(0); yhi.append(0)
            ax.bar(x + (j - 1.5) * width, ys, width, label=cond, color=colors[cond],
                   yerr=[ylo, yhi], capsize=2, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(concepts, rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10); ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("probability"); axes[0].legend(fontsize=9)
    fig.suptitle(f"Sustained concept-direction ablation during imagination "
                 f"(burn-in {args.burn_in} real steps, block {args.target_block})", fontsize=11)
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=140); plt.close(fig)
    img = base64.b64encode(buf.getvalue()).decode("ascii")

    rows = []
    for s in sorted(summary, key=lambda x: (x["concept"], conds.index(x["condition"]))):
        rows.append(f"<tr><td class=label>{s['concept']}</td><td>{s['condition']}</td>"
                    f"<td>{s['n']}</td>"
                    f"<td>{s['p_first']:.2f} [{s['p_first_lo']:.2f},{s['p_first_hi']:.2f}]</td>"
                    f"<td>{s['p_within']:.2f} [{s['p_within_lo']:.2f},{s['p_within_hi']:.2f}]</td>"
                    f"<td>{s['reward_mean']:+.2f}</td></tr>")
    html = ("<!doctype html><meta charset=utf-8><title>imagination ablation</title>"
            "<style>body{font:14px ui-monospace,Menlo,Consolas,monospace;background:#0d1117;"
            "color:#c9d1d9;padding:18px;}h1{color:#79c0ff;font-size:18px;}"
            ".dim{color:#8b949e;}img{max-width:100%;background:white;border-radius:6px;}"
            "table{border-collapse:collapse;font-size:12.5px;margin-top:10px;width:100%;}"
            "th,td{padding:4px 8px;border-bottom:1px solid #21262d;text-align:right;}"
            "td.label{text-align:left;}th{color:#8b949e;font-size:11px;text-transform:uppercase;}</style>"
            f"<h1>Concept ablation in imagination</h1>"
            f"<div class=dim>{len(trials)} rollouts; ablation = sustained projection-removal at WM block "
            f"{args.target_block} during imagination only (burn-in clean); real action forced at unlock step; "
            f"detector codes = top-2 (slot,code) per achievement from the MI table.</div>"
            f"<img src='data:image/png;base64,{img}'>"
            "<table><tr><th>concept</th><th>condition</th><th>n</th>"
            "<th>P(first step) [CI]</th><th>P(within horizon) [CI]</th><th>mean reward sum</th></tr>"
            + "".join(rows) + "</table>")
    (args.out / "imagination.html").write_text(html, encoding="utf-8")
    print(f"wrote {args.out/'imagination.html'}", flush=True)


if __name__ == "__main__":
    main()
