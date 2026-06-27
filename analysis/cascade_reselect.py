"""E3 - causal re-selection. The cascade picks the wood SAE feature by *detector
lift* (decodability). Question: does a DIFFERENT feature gate wood *generation*?

For each agent, sweep the top-K features by collect_wood detector lift; for each,
run TEACHER-FORCED imagination ablation (replay the baseline action sequence so
the effect is the world model's content generation, not policy divergence) and
measure P(wood detector code sampled within horizon). The lift-best feature is
the cascade default; we report whether any candidate suppresses the wood code.

  selection artifact (H3) -> some non-top-lift feature gates wood in the seeds.
  genuinely distributed   -> no single feature gates wood in the seeds.

Outputs <out>/reselect.json.
"""
from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from torch.distributions import Categorical
sys.path.insert(0, str(Path(__file__).resolve().parent))
from imagine_cascade import load_agent, detector_codes_from_mi


def main():
    ap = argparse.ArgumentParser()
    for a in ("run", "rollouts", "obs", "sae", "features", "mi-table", "out"):
        ap.add_argument("--" + a, required=True, type=Path)
    ap.add_argument("--focus", default="collect_wood")
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--burn-in", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--n-moments", type=int, default=30)
    ap.add_argument("--target-block", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    run = args.run.resolve(); args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device); rng = np.random.default_rng(args.seed)
    print("loading agent...", flush=True)
    agent = load_agent(run, device)
    wm, tk, ac = agent.world_model, agent.tokenizer, agent.actor_critic
    from envs.world_model_env import WorldModelEnv
    from utils import compute_softmax_over_buckets, symexp
    from einops import rearrange as rea

    class Env(WorldModelEnv):
        @torch.no_grad()
        def step_rec(self, action):
            if not isinstance(action, torch.Tensor):
                action = torch.tensor(action, dtype=torch.long).reshape(-1, 1).to(self.device)
            a = self.world_model.act_emb(action)
            inp = a if self.x is None and self.last_latent_token_emb is None else (
                torch.cat((self.x, a), dim=1) if self.last_latent_token_emb is None
                else torch.cat((self.last_latent_token_emb, self.x, a), dim=1))
            o = self.world_model(inp, use_kv_cache=True)
            rew = symexp(compute_softmax_over_buckets(o.logits_rewards)) if self.world_model.config.two_hot_rews \
                else Categorical(logits=o.logits_rewards).sample().float() - 1
            lt = Categorical(logits=o.logits_latents).sample(); toks = [lt]
            for _ in range(self.tokenizer.config.num_tokens - 1):
                o = self.world_model(self.world_model.latents_emb(lt), use_kv_cache=True)
                lt = Categorical(logits=o.logits_latents).sample(); toks.append(lt)
            self.last_latent_token_emb = self.world_model.latents_emb(lt)
            q = self.tokenizer.quantizer.embed_tokens(torch.stack(toks, dim=-1))
            self.obs = self.tokenizer.decode(self.obs, action,
                rea(q, 'b t (h w) (k l e) -> b t e (h k) (w l)',
                    h=self.tokenizer.tokens_grid_res, k=self.tokenizer.token_res, l=self.tokenizer.token_res),
                should_clamp=True)
            self.x = rea(self.world_model.frame_cnn(self.obs), 'b 1 k e -> b k e')
            return [int(t.flatten()[0].item()) for t in toks], float(rew.flatten()[0].item()), self.obs
    env = Env(tk, wm, device)

    # candidate features: top-K by collect_wood detector lift
    feats = json.loads(args.features.read_text())["features"]
    wood = sorted([(f["feature"], f["best_ach"]["lift"]) for f in feats
                   if f.get("best_ach") and f["best_ach"]["name"] == args.focus], key=lambda x: -x[1])
    cands = wood[:args.top_k]
    sae_blob = torch.load(args.sae, map_location=device, weights_only=False)
    decoder = sae_blob["state_dict"]["decoder.weight"].to(device).float()
    dirs = {fid: F.normalize(decoder[:, fid], dim=-1) for fid, _ in cands}
    detectors = detector_codes_from_mi(args.mi_table, top_n=2); wood_codes = detectors[args.focus]
    print(f"focus={args.focus}  {len(cands)} candidate wood features (lift {cands[0][1]:.0f}..{cands[-1][1]:.0f})  "
          f"det={['s%dc%d' % wc for wc in wood_codes]}", flush=True)

    # moments
    r = np.load(args.rollouts, allow_pickle=True); obs_npz = np.load(args.obs)
    all_obs = obs_npz["obs"]; obs_starts = obs_npz["episode_starts"]
    actions = r["actions"].astype(np.int64); aj = r["ach_just_unlocked"]
    ep_ids = r["episode_ids"].astype(np.int64); ep_lens = r["episode_lengths"].astype(np.int64)
    ach_names = [str(x) for x in r["achievement_names"]]
    ep_start_row = np.concatenate(([0], np.cumsum(ep_lens)))
    fidx = ach_names.index(args.focus)
    unlock = [gi for gi in np.where(aj[:, fidx])[0] if (gi - ep_start_row[ep_ids[gi]]) >= args.burn_in]
    sel = rng.choice(unlock, size=min(args.n_moments, len(unlock)), replace=False)
    print(f"{len(sel)} moments", flush=True)

    hook_state = {"d": None}
    def hook(_m, _i, out):
        d = hook_state["d"]
        return out if d is None else out - (out @ d).unsqueeze(-1) * d
    handle = wm.transformer.blocks[args.target_block].register_forward_hook(hook)

    def roll(burn_obs, burn_act, direction, first_action, forced=None):
        hook_state["d"] = None
        env.reset_from_past(burn_obs.clone(), burn_act.clone()); ac.reset(n=1)
        with torch.no_grad():
            for t in range(args.burn_in): _ = ac.act(burn_obs[:, t])
        hook_state["d"] = direction
        action = forced[0] if forced is not None else int(first_action); acts = []; hit = 0
        with torch.no_grad():
            for h in range(args.horizon):
                codes, _, frame = env.step_rec(action)
                if any(codes[s] == c for s, c in wood_codes): hit = 1
                acts.append(int(action))
                if forced is not None:
                    action = forced[h + 1] if h + 1 < len(forced) else action
                else:
                    act_tok, _ = ac.act(frame[:, 0], should_sample=True, temperature=1.0); action = int(act_tok.item())
        hook_state["d"] = None
        return hit, acts

    within = {fid: [] for fid, _ in cands}; base_within = []
    t0 = time.time(); torch.manual_seed(args.seed)
    for mi, gi in enumerate(sel):
        ep = int(ep_ids[gi]); gl = int(gi - ep_start_row[ep])
        ep_obs = all_obs[obs_starts[ep]:obs_starts[ep+1]]; ep_act = actions[ep_start_row[ep]:ep_start_row[ep+1]]
        lo = gl - args.burn_in
        burn_obs = torch.from_numpy(ep_obs[lo:gl+1]).to(device).float().div(255).unsqueeze(0)
        burn_act = torch.from_numpy(ep_act[lo:gl]).to(device).unsqueeze(0)
        real_aT = int(ep_act[gl])
        bhit, bacts = roll(burn_obs, burn_act, None, real_aT)
        base_within.append(bhit)
        for fid, _ in cands:
            hit, _ = roll(burn_obs, burn_act, dirs[fid], real_aT, forced=bacts)
            within[fid].append(hit)
        if (mi + 1) % 10 == 0: print(f"  {mi+1}/{len(sel)}  ({(time.time()-t0)/60:.1f} min)", flush=True)
    handle.remove()

    lift = {fid: lf for fid, lf in cands}
    rows = [dict(feature=fid, lift=lift[fid], within=float(np.mean(within[fid])),
                 rank_by_lift=i + 1) for i, (fid, _) in enumerate(cands)]
    rows_by_supp = sorted(rows, key=lambda r: r["within"])
    out = {"focus": args.focus, "baseline_within": float(np.mean(base_within)),
           "lift_best": rows[0], "most_suppressing": rows_by_supp[0],
           "n_moments": int(len(sel)), "candidates": rows}
    (args.out / "reselect.json").write_text(json.dumps(out, indent=2))
    print(f"\nbaseline within-horizon: {out['baseline_within']:.2f}")
    print(f"{'feat':>6} {'lift':>7} {'liftRank':>9} {'within(TF)':>11}")
    for r in rows_by_supp:
        tag = "  <-- cascade default" if r["rank_by_lift"] == 1 else ""
        print(f"{r['feature']:>6} {r['lift']:>7.0f} {r['rank_by_lift']:>9} {r['within']:>11.2f}{tag}")
    mb = rows_by_supp[0]
    print(f"\nmost-suppressing feature: #{mb['feature']} (lift rank {mb['rank_by_lift']}) -> within {mb['within']:.2f}")
    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
