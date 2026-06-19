"""Behavioural cascade through the Crafter tech tree under a residual-stream lesion.

Closes the chain from "which codes get sampled" to "what the agent does".
Delta-IRIS is an agent: inside imagination the actor-critic acts on the decoded
frames, so a residual-stream lesion can change the agent's behaviour, not just
the next-code distribution. We exploit Crafter's tech tree as a pre-specified
ground-truth causal graph: collect_wood gates place_table / make_wood_pickaxe /
make_wood_sword, which gate collect_stone, which gates the stone tier, etc.

Design (frozen checkpoint, no retraining):
  * Burn the world model in on the real history ending at a collect_wood unlock.
  * Force the real action at the unlock step, then let the actor-critic act for
    `horizon` imagined steps (closed loop: codes sampled -> frame decoded ->
    policy picks next action).
  * Conditions (sustained projection-removal at WM block 1 during imagination):
      baseline | sae (ablate the WOOD SAE feature f187) | cav (wood CAV) | random
  * Outcomes per rollout: imagined return (sum of head_rewards), and for EVERY
    achievement whether its detector code(s) (top-2 from the MI table) are
    sampled within the horizon.

Prediction: ablating the wood feature should not merely suppress the wood code
(near-tautological) but CASCADE - suppress the wood-tree achievements downstream
of wood in tech-tree order and lower imagined return - while the CAV (and random)
leave imagined behaviour intact, and off-tree achievements are unaffected (a
built-in specificity control).

Outputs under --out: cascade_trials.json, cascade_effects.json, cascade.html
"""
from __future__ import annotations
import argparse, base64, io, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.distributions.categorical import Categorical

# Crafter tech tree: achievement -> depth from collect_wood (None = off the wood tree).
WOOD_TREE_DEPTH = {
    "collect_wood": 0,
    "place_table": 1, "make_wood_pickaxe": 1, "make_wood_sword": 1,
    "collect_stone": 2,
    "place_stone": 3, "make_stone_pickaxe": 3, "make_stone_sword": 3, "place_furnace": 3,
    "collect_coal": 4, "collect_iron": 4,
    "make_iron_pickaxe": 5, "make_iron_sword": 5,
}
OFF_TREE = ["collect_drink", "collect_sapling", "place_plant", "eat_cow",
            "wake_up", "defeat_zombie", "defeat_skeleton"]


def load_agent(run: Path, device):
    sys.path.insert(0, str(run / "src"))
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    cfg = OmegaConf.load(run / ".hydra" / "config.yaml")
    for k in ("tokenizer", "world_model"):
        if cfg.params[k].num_actions is None: cfg.params[k].num_actions = 17
    if cfg.params.actor_critic.model.num_actions is None: cfg.params.actor_critic.model.num_actions = 17
    OmegaConf.resolve(cfg)
    from models.tokenizer import Tokenizer
    from models.world_model import WorldModel
    from models.actor_critic import ActorCritic
    from agent import Agent
    agent = Agent(Tokenizer(instantiate(cfg.params.tokenizer)),
                  WorldModel(instantiate(cfg.params.world_model)),
                  ActorCritic(instantiate(cfg.params.actor_critic))).to(device).eval()
    agent.load(run / "checkpoints" / "last.pt", device=device,
               load_tokenizer=True, load_world_model=True, load_actor_critic=True, strict=False)
    return agent


def detector_codes_from_mi(mi_table: Path, top_n: int = 2):
    per = defaultdict(list)
    for r in json.loads(mi_table.read_text()):
        per[r["achievement"]].append((r["lift"], r["slot"], r["code"]))
    out = {}
    for ach, rows in per.items():
        rows.sort(reverse=True)
        out[ach] = [(s, c) for _, s, c in rows[:top_n]]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--rollouts", required=True, type=Path)
    ap.add_argument("--obs", required=True, type=Path)
    ap.add_argument("--sae", required=True, type=Path)
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--cavs", required=True, type=Path)
    ap.add_argument("--mi-table", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--focus", default="collect_wood", help="gating concept to lesion")
    ap.add_argument("--burn-in", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--n-moments", type=int, default=40)
    ap.add_argument("--target-block", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    run = args.run.resolve(); args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device); rng = np.random.default_rng(args.seed)

    print("loading agent...", flush=True)
    agent = load_agent(run, device)
    wm, tk, ac = agent.world_model, agent.tokenizer, agent.actor_critic
    max_blocks = wm.config.transformer_config.max_blocks
    assert args.burn_in + args.horizon <= max_blocks, f"burn+horizon must fit {max_blocks} blocks"

    from envs.world_model_env import WorldModelEnv

    class RecordingWMEnv(WorldModelEnv):
        @torch.no_grad()
        def step_recorded(self, action):
            assert self.world_model.transformer.num_blocks_left_in_kv_cache > 1, "KV cache about to flush"
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
                    h=self.tokenizer.tokens_grid_res, k=self.tokenizer.token_res, l=self.tokenizer.token_res),
                should_clamp=True)
            self.x = rea(self.world_model.frame_cnn(self.obs), 'b 1 k e -> b k e')
            codes = [int(t.flatten()[0].item()) for t in latent_tokens]
            return codes, reward, self.obs

    wm_env = RecordingWMEnv(tk, wm, device)

    # directions
    sae_blob = torch.load(args.sae, map_location=device, weights_only=False)
    decoder = sae_blob["state_dict"]["decoder.weight"].to(device).float()
    feats = json.loads(args.features.read_text())["features"]
    best_feat = {}
    for f in feats:
        a = f.get("best_ach")
        if not a or f["density"] < 0.001: continue
        cur = best_feat.get(a["name"])
        if cur is None or a["lift"] > cur["lift"]:
            best_feat[a["name"]] = dict(feat=f["feature"], lift=a["lift"])
    assert args.focus in best_feat, f"no strong SAE feature for {args.focus}"
    feat_id = best_feat[args.focus]["feat"]
    d_sae = decoder[:, feat_id].clone()

    cav_npz = np.load(args.cavs, allow_pickle=True)
    cav_names = [str(x) for x in cav_npz["concept_names"]]
    W_cav, sd_cav = cav_npz["w_wm_block_1"], cav_npz["std_wm_block_1"]
    def cav_direction(ach):
        name = f"just[{ach}]"
        if name not in cav_names: return None
        w = W_cav[cav_names.index(name)]
        if not np.any(w): return None
        return torch.from_numpy(w / sd_cav).to(device).float()
    d_cav = cav_direction(args.focus)
    d_rand = torch.randn_like(d_sae)
    print(f"focus={args.focus}  SAE feat #{feat_id}  CAV={'yes' if d_cav is not None else 'MISSING'}", flush=True)

    detectors = detector_codes_from_mi(args.mi_table, top_n=2)
    meas_achs = [a for a in detectors]  # achievements with detector codes

    # rollouts
    r = np.load(args.rollouts, allow_pickle=True); obs_npz = np.load(args.obs)
    all_obs = obs_npz["obs"]; obs_starts = obs_npz["episode_starts"]
    actions = r["actions"].astype(np.int64); aj = r["ach_just_unlocked"]
    ep_ids = r["episode_ids"].astype(np.int64); ep_lens = r["episode_lengths"].astype(np.int64)
    ach_names = [str(x) for x in r["achievement_names"]]
    ep_start_row = np.concatenate(([0], np.cumsum(ep_lens)))
    focus_idx = ach_names.index(args.focus)

    unlock_global = [g for g in np.where(aj[:, focus_idx])[0]
                     if (g - ep_start_row[ep_ids[g]]) >= args.burn_in]
    sel = rng.choice(unlock_global, size=min(args.n_moments, len(unlock_global)), replace=False)
    print(f"{len(sel)} burn-in moments at {args.focus} unlocks", flush=True)

    hook_state = {"direction": None}
    def hook(_m, _i, output):
        d = hook_state["direction"]
        if d is None: return output
        dn = F.normalize(d, dim=-1)
        return output - (output @ dn).unsqueeze(-1) * dn
    handle = wm.transformer.blocks[args.target_block].register_forward_hook(hook)

    conditions = [("baseline", None), ("sae", d_sae)]
    if d_cav is not None: conditions.append(("cav", d_cav))
    conditions.append(("random", d_rand))

    trials = []; t0 = time.time(); torch.manual_seed(args.seed)
    for mi, g in enumerate(sel):
        ep = int(ep_ids[g]); g_local = int(g - ep_start_row[ep])
        ep_obs = all_obs[obs_starts[ep]:obs_starts[ep+1]]
        ep_act = actions[ep_start_row[ep]:ep_start_row[ep+1]]
        lo = g_local - args.burn_in
        burn_obs = torch.from_numpy(ep_obs[lo:g_local+1]).to(device).float().div(255).unsqueeze(0)
        burn_act = torch.from_numpy(ep_act[lo:g_local]).to(device).unsqueeze(0)
        real_a_T = int(ep_act[g_local])
        for cond_name, direction in conditions:
            hook_state["direction"] = None
            wm_env.reset_from_past(burn_obs.clone(), burn_act.clone())
            ac.reset(n=1)
            with torch.no_grad():
                for t in range(args.burn_in): _ = ac.act(burn_obs[:, t])
            hook_state["direction"] = direction
            seen = {a: None for a in meas_achs}      # ach -> first hit step
            rewards = []; action = real_a_T
            with torch.no_grad():
                for h in range(args.horizon):
                    codes, rew, frame = wm_env.step_recorded(action)
                    rewards.append(rew)
                    for a in meas_achs:
                        if seen[a] is None and any(codes[s] == c for s, c in detectors[a]):
                            seen[a] = h
                    act_tok, _ = ac.act(frame[:, 0], should_sample=True, temperature=1.0)
                    action = int(act_tok.item())
            hook_state["direction"] = None
            trials.append(dict(condition=cond_name, episode=ep, ep_step=g_local,
                               reward_sum=float(np.sum(rewards)),
                               first_hit={a: seen[a] for a in meas_achs}))
        if (mi + 1) % 10 == 0:
            print(f"  {mi+1}/{len(sel)} moments  ({(time.time()-t0)/60:.1f} min)", flush=True)
    handle.remove()
    (args.out / "cascade_trials.json").write_text(json.dumps(trials, indent=2))
    print(f"\n{len(trials)} rollouts in {(time.time()-t0)/60:.1f} min", flush=True)

    # aggregate
    def boot_ci(arr, B=1000):
        if len(arr) == 0: return 0.0, 0.0
        rng2 = np.random.default_rng(0)
        m = [rng2.choice(arr, len(arr), replace=True).mean() for _ in range(B)]
        return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
    conds = [c for c, _ in conditions]
    summary = {"focus": args.focus, "feat": feat_id, "horizon": args.horizon,
               "burn_in": args.burn_in, "wood_tree_depth": WOOD_TREE_DEPTH, "return": {}, "per_ach": []}
    for cond in conds:
        sub = [t for t in trials if t["condition"] == cond]
        rew = np.array([t["reward_sum"] for t in sub], float)
        lo, hi = boot_ci(rew)
        summary["return"][cond] = dict(n=len(sub), mean=float(rew.mean()), lo=lo, hi=hi)
    for a in meas_achs:
        row = {"achievement": a, "depth": WOOD_TREE_DEPTH.get(a, None),
               "on_wood_tree": a in WOOD_TREE_DEPTH}
        for cond in conds:
            sub = [t for t in trials if t["condition"] == cond]
            within = np.array([1.0 if t["first_hit"][a] is not None else 0.0 for t in sub])
            lo, hi = boot_ci(within)
            row[cond] = dict(p=float(within.mean()), lo=lo, hi=hi)
        summary["per_ach"].append(row)
    (args.out / "cascade_effects.json").write_text(json.dumps(summary, indent=2))

    # headline print: imagined return + wood-tree cascade
    print("\n=== imagined return ===")
    for cond in conds:
        rr = summary["return"][cond]; print(f"  {cond:9} {rr['mean']:+.2f}  [{rr['lo']:+.2f},{rr['hi']:+.2f}]  n={rr['n']}")
    print("\n=== wood-tree imagined-unlock probability (baseline -> sae | cav | random) ===")
    tree_rows = sorted([r for r in summary["per_ach"] if r["on_wood_tree"]], key=lambda r: (r["depth"], r["achievement"]))
    for r in tree_rows:
        b = r["baseline"]["p"]; s = r["sae"]["p"]; c = r.get("cav", {}).get("p", float("nan")); rnd = r["random"]["p"]
        print(f"  d{r['depth']} {r['achievement']:20} {b:.2f} -> sae {s:.2f} | cav {c:.2f} | rand {rnd:.2f}")
    off = [r for r in summary["per_ach"] if not r["on_wood_tree"]]
    if off:
        import statistics as st
        db = st.mean(r["baseline"]["p"] for r in off); ds = st.mean(r["sae"]["p"] for r in off)
        print(f"\n  off-tree (specificity) mean P: baseline {db:.2f} -> sae {ds:.2f}  ({len(off)} achievements)")

    # figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 2.2]})
    # (A) imagined return
    cc = {"baseline": "#2ca02c", "sae": "#1B4F8A", "cav": "#B23A8E", "random": "#9aa0a8"}
    xs = np.arange(len(conds))
    ax[0].bar(xs, [summary["return"][c]["mean"] for c in conds],
              yerr=[[summary["return"][c]["mean"]-summary["return"][c]["lo"] for c in conds],
                    [summary["return"][c]["hi"]-summary["return"][c]["mean"] for c in conds]],
              color=[cc[c] for c in conds], capsize=3, edgecolor="black", lw=0.5)
    ax[0].set_xticks(xs); ax[0].set_xticklabels(conds); ax[0].set_ylabel("imagined return (sum of $\\hat r$)")
    ax[0].set_title("(A) Imagined return", fontsize=11, fontweight="bold"); ax[0].grid(axis="y", alpha=0.25)
    # (B) cascade: per wood-tree achievement, baseline vs sae vs cav
    rows = tree_rows
    y = np.arange(len(rows)); h = 0.26
    for j, cond in enumerate(["baseline", "sae", "cav"]):
        if cond not in conds: continue
        ax[1].barh(y + (1-j)*h, [r[cond]["p"] for r in rows], height=h,
                   color=cc[cond], edgecolor="black", lw=0.4, label=cond)
    ax[1].set_yticks(y); ax[1].set_yticklabels([f"d{r['depth']} {r['achievement']}" for r in rows], fontsize=8)
    ax[1].invert_yaxis(); ax[1].set_xlabel("P(detector code sampled within horizon)")
    ax[1].set_title("(B) Cascade through the wood tech tree", fontsize=11, fontweight="bold")
    ax[1].legend(fontsize=8.5, loc="lower right"); ax[1].grid(axis="x", alpha=0.25); ax[1].set_xlim(0, 1.02)
    for s in ("top", "right"):
        ax[0].spines[s].set_visible(False); ax[1].spines[s].set_visible(False)
    fig.suptitle(f"Behavioural cascade under a residual-stream lesion of the {args.focus} feature "
                 f"(f{feat_id}); closed-loop imagination, n={summary['return']['baseline']['n']}", fontsize=10)
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=140); plt.close(fig)
    img = base64.b64encode(buf.getvalue()).decode("ascii")
    (args.out / "cascade.html").write_text(
        "<!doctype html><meta charset=utf-8><title>behavioural cascade</title>"
        "<style>body{font:14px ui-monospace,monospace;background:#0d1117;color:#c9d1d9;padding:18px}"
        "img{max-width:100%;background:#fff;border-radius:6px}</style>"
        f"<h1>Behavioural cascade: lesion of the {args.focus} feature (f{feat_id})</h1>"
        f"<img src='data:image/png;base64,{img}'>", encoding="utf-8")
    print(f"\nwrote {args.out/'cascade.html'}", flush=True)


if __name__ == "__main__":
    main()
