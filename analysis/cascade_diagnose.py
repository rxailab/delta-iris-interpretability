"""Diagnose why the imagination cascade's wood-code suppression replicates in the
original (1.00->0.03) but not in seeds 1-2 (->1.00, 0.97).

For each collect_wood burn-in moment we run, under continuous projection of a
direction out of block-1:
  - baseline (closed-loop): record the sampled actions + per-step wood-code logP.
  - each ablation (sae / cav / random) in TWO modes:
      * teacher-forced: replay baseline's actions -> isolates the WORLD MODEL's
        direct response to the lesion (gating), trajectory held fixed.
      * closed-loop: actor samples its own actions -> full effect incl. policy
        divergence (steering).

Per step we log, detector-independently:
  - wood-code logP (graded; sensitive where the binary "code sampled" saturates)  [E1]
  - binary: top-lift wood detector code sampled                                    [reproduce]
  - frame MAE vs the baseline trajectory (0-255)                                   [E2 divergence]
  - next-code KL(baseline || ablated) summed over the 4 slots                      [E2 divergence]

Reading:
  gating  -> wood logP drops under TEACHER-FORCED ablation.
  steering-> teacher-forced ~flat, suppression appears only CLOSED-LOOP (policy diverges).
  readout artifact -> binary saturates (->1.00) but logP separates conditions.

Outputs <out>/cascade_diagnose.json (per-step curves + scalars).
"""
from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imagine_cascade import load_agent, detector_codes_from_mi  # module-level helpers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--rollouts", required=True, type=Path)
    ap.add_argument("--obs", required=True, type=Path)
    ap.add_argument("--sae", required=True, type=Path)
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--cavs", required=True, type=Path)
    ap.add_argument("--mi-table", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--focus", default="collect_wood")
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
    from envs.world_model_env import WorldModelEnv
    from utils import compute_softmax_over_buckets, symexp
    from einops import rearrange as rea

    class DiagEnv(WorldModelEnv):
        @torch.no_grad()
        def step_diag(self, action):
            if not isinstance(action, torch.Tensor):
                action = torch.tensor(action, dtype=torch.long).reshape(-1, 1).to(self.device)
            a = self.world_model.act_emb(action)
            if self.last_latent_token_emb is None:
                inp = a if self.x is None else torch.cat((self.x, a), dim=1)
            else:
                inp = torch.cat((self.last_latent_token_emb, self.x, a), dim=1)
            outputs_wm = self.world_model(inp, use_kv_cache=True)
            reward = symexp(compute_softmax_over_buckets(outputs_wm.logits_rewards)) \
                if self.world_model.config.two_hot_rews \
                else Categorical(logits=outputs_wm.logits_rewards).sample().float() - 1
            reward = float(reward.flatten()[0].item())
            slot_logits = []
            lt = Categorical(logits=outputs_wm.logits_latents).sample()
            slot_logits.append(outputs_wm.logits_latents.reshape(-1).float())
            toks = [lt]
            for _ in range(self.tokenizer.config.num_tokens - 1):
                emb = self.world_model.latents_emb(lt)
                outputs_wm = self.world_model(emb, use_kv_cache=True)
                lt = Categorical(logits=outputs_wm.logits_latents).sample()
                slot_logits.append(outputs_wm.logits_latents.reshape(-1).float())
                toks.append(lt)
            self.last_latent_token_emb = self.world_model.latents_emb(lt)
            q = self.tokenizer.quantizer.embed_tokens(torch.stack(toks, dim=-1))
            self.obs = self.tokenizer.decode(
                self.obs, action,
                rea(q, 'b t (h w) (k l e) -> b t e (h k) (w l)',
                    h=self.tokenizer.tokens_grid_res, k=self.tokenizer.token_res, l=self.tokenizer.token_res),
                should_clamp=True)
            self.x = rea(self.world_model.frame_cnn(self.obs), 'b 1 k e -> b k e')
            codes = [int(t.flatten()[0].item()) for t in toks]
            logits4 = torch.stack(slot_logits, dim=0)          # [4, 1024]
            return codes, reward, self.obs, logits4

    env = DiagEnv(tk, wm, device)

    # ---- direction: wood SAE feature (top detector lift), CAV, random ----
    sae_blob = torch.load(args.sae, map_location=device, weights_only=False)
    decoder = sae_blob["state_dict"]["decoder.weight"].to(device).float()   # [d, n_feat]
    feats = json.loads(args.features.read_text())["features"]
    best_feat = {}
    for f in feats:
        a = f.get("best_ach")
        if a and (a["name"] not in best_feat or a["lift"] > best_feat[a["name"]]["lift"]):
            best_feat[a["name"]] = dict(feat=f["feature"], lift=a["lift"])
    feat_id = best_feat[args.focus]["feat"]
    d_sae = F.normalize(decoder[:, feat_id], dim=-1)
    cav_npz = np.load(args.cavs, allow_pickle=True)
    cav_names = [str(x) for x in cav_npz["concept_names"]]
    name = f"just[{args.focus}]"
    if name in cav_names:
        w = cav_npz["w_wm_block_1"][cav_names.index(name)]
        d_cav = F.normalize(torch.from_numpy(w / cav_npz["std_wm_block_1"]).to(device).float(), dim=-1)
    else:
        d_cav = None
    g = torch.Generator(device=device).manual_seed(args.seed)
    d_rand = F.normalize(torch.randn(d_sae.shape, generator=g, device=device), dim=-1)

    # ---- wood detector code (top lift) ----
    detectors = detector_codes_from_mi(args.mi_table, top_n=2)
    wood_codes = detectors[args.focus]          # list of (slot, code)
    wslot, wcode = wood_codes[0]
    print(f"focus={args.focus} SAE#{feat_id} CAV={'y' if d_cav is not None else 'n'} "
          f"wooddet=s{wslot}c{wcode}", flush=True)

    # ---- moments ----
    r = np.load(args.rollouts, allow_pickle=True); obs_npz = np.load(args.obs)
    all_obs = obs_npz["obs"]; obs_starts = obs_npz["episode_starts"]
    actions = r["actions"].astype(np.int64); aj = r["ach_just_unlocked"]
    ep_ids = r["episode_ids"].astype(np.int64); ep_lens = r["episode_lengths"].astype(np.int64)
    ach_names = [str(x) for x in r["achievement_names"]]
    ep_start_row = np.concatenate(([0], np.cumsum(ep_lens)))
    focus_idx = ach_names.index(args.focus)
    unlock_global = [gi for gi in np.where(aj[:, focus_idx])[0]
                     if (gi - ep_start_row[ep_ids[gi]]) >= args.burn_in]
    sel = rng.choice(unlock_global, size=min(args.n_moments, len(unlock_global)), replace=False)
    print(f"{len(sel)} moments", flush=True)

    hook_state = {"direction": None}
    def hook(_m, _i, output):
        d = hook_state["direction"]
        if d is None: return output
        return output - (output @ d).unsqueeze(-1) * d
    handle = wm.transformer.blocks[args.target_block].register_forward_hook(hook)

    H = args.horizon
    def wood_logp(logits4):
        return float(F.log_softmax(logits4[wslot], dim=-1)[wcode].item())
    def kl_base_abl(lb, la):  # sum over slots KL(base||abl)
        pb = F.softmax(lb, dim=-1); lpb = F.log_softmax(lb, dim=-1); lpa = F.log_softmax(la, dim=-1)
        return float((pb * (lpb - lpa)).sum().item())

    def rollout(burn_obs, burn_act, direction, first_action, forced_actions=None):
        """Returns per-step dicts. forced_actions: list[int] for teacher-forcing, else closed-loop."""
        hook_state["direction"] = None
        env.reset_from_past(burn_obs.clone(), burn_act.clone()); ac.reset(n=1)
        with torch.no_grad():
            for t in range(args.burn_in): _ = ac.act(burn_obs[:, t])
        hook_state["direction"] = direction
        steps = []; action = forced_actions[0] if forced_actions is not None else int(first_action)
        acts = []
        with torch.no_grad():
            for h in range(H):
                codes, rew, frame, logits4 = env.step_diag(action)
                acts.append(int(action))
                steps.append(dict(codes=codes, frame=frame.detach().float(), logits4=logits4.detach(),
                                  wlogp=wood_logp(logits4),
                                  wsamp=1.0 if any(codes[s] == c for s, c in wood_codes) else 0.0))
                if forced_actions is not None:
                    action = forced_actions[h + 1] if h + 1 < len(forced_actions) else action
                else:
                    act_tok, _ = ac.act(frame[:, 0], should_sample=True, temperature=1.0)
                    action = int(act_tok.item())
        hook_state["direction"] = None
        return steps, acts

    conds = [("sae", d_sae), ("random", d_rand)] + ([("cav", d_cav)] if d_cav is not None else [])
    # accumulators: metric[cond][mode] -> list over moments of per-step arrays
    acc = {"baseline": {"wlogp": [], "wsamp": []}}
    for c, _ in conds:
        acc[c] = {m: {"wlogp": [], "wsamp": [], "mae": [], "kl": []} for m in ("tf", "cl")}

    t0 = time.time(); torch.manual_seed(args.seed)
    for mi, gi in enumerate(sel):
        ep = int(ep_ids[gi]); g_local = int(gi - ep_start_row[ep])
        ep_obs = all_obs[obs_starts[ep]:obs_starts[ep+1]]
        ep_act = actions[ep_start_row[ep]:ep_start_row[ep+1]]
        lo = g_local - args.burn_in
        burn_obs = torch.from_numpy(ep_obs[lo:g_local+1]).to(device).float().div(255).unsqueeze(0)
        burn_act = torch.from_numpy(ep_act[lo:g_local]).to(device).unsqueeze(0)
        real_a_T = int(ep_act[g_local])

        base, base_acts = rollout(burn_obs, burn_act, None, real_a_T)
        forced = base_acts                      # baseline's realized action sequence (len H)
        acc["baseline"]["wlogp"].append([s["wlogp"] for s in base])
        acc["baseline"]["wsamp"].append([s["wsamp"] for s in base])
        base_frames = [s["frame"] for s in base]; base_logits = [s["logits4"] for s in base]

        for cname, d in conds:
            tf, _ = rollout(burn_obs, burn_act, d, real_a_T, forced_actions=forced)
            cl, _ = rollout(burn_obs, burn_act, d, real_a_T, forced_actions=None)
            for mode, steps in (("tf", tf), ("cl", cl)):
                acc[cname][mode]["wlogp"].append([s["wlogp"] for s in steps])
                acc[cname][mode]["wsamp"].append([s["wsamp"] for s in steps])
                acc[cname][mode]["mae"].append(
                    [float((s["frame"] - bf).abs().mean().item() * 255) for s, bf in zip(steps, base_frames)])
                acc[cname][mode]["kl"].append(
                    [kl_base_abl(bl, s["logits4"]) for s, bl in zip(steps, base_logits)])
        if (mi + 1) % 10 == 0:
            print(f"  {mi+1}/{len(sel)}  ({(time.time()-t0)/60:.1f} min)", flush=True)
    handle.remove()

    def curve(x): return np.array(x, float).mean(0).tolist()      # mean over moments -> per-step
    def scal(x): return float(np.array(x, float).mean())
    out = {"focus": args.focus, "feat": feat_id, "wood_det": f"s{wslot}c{wcode}",
           "horizon": H, "n_moments": int(len(sel)),
           "baseline": {"wlogp_step": curve(acc["baseline"]["wlogp"]),
                        "wsamp_step": curve(acc["baseline"]["wsamp"]),
                        "wlogp_mean": scal(acc["baseline"]["wlogp"]),
                        "wsamp_within": scal([max(r) for r in acc["baseline"]["wsamp"]])}}
    for c, _ in conds:
        out[c] = {}
        for mode in ("tf", "cl"):
            a = acc[c][mode]
            out[c][mode] = {
                "wlogp_step": curve(a["wlogp"]), "wsamp_step": curve(a["wsamp"]),
                "mae_step": curve(a["mae"]), "kl_step": curve(a["kl"]),
                "wlogp_mean": scal(a["wlogp"]),
                "wsamp_within": scal([max(r) for r in a["wsamp"]]),   # P(code sampled within horizon)
                "mae_mean": scal(a["mae"]), "kl_mean": scal(a["kl"]),
                "dwlogp_vs_base": scal(acc["baseline"]["wlogp"]) - scal(a["wlogp"]),  # suppression
            }
    (args.out / "cascade_diagnose.json").write_text(json.dumps(out, indent=2))
    # console summary
    print(f"\n=== {args.focus}  feat#{feat_id}  det s{wslot}c{wcode}  ({len(sel)} moments) ===")
    print(f"baseline: wood logP {out['baseline']['wlogp_mean']:+.2f}  within-horizon sampled {out['baseline']['wsamp_within']:.2f}")
    for c, _ in conds:
        tf, cl = out[c]["tf"], out[c]["cl"]
        print(f"{c:>7}  TEACHER-FORCED: within {tf['wsamp_within']:.2f}  logP {tf['wlogp_mean']:+.2f} (drop {tf['dwlogp_vs_base']:+.2f})  MAE {tf['mae_mean']:.1f}  KL {tf['kl_mean']:.2f}")
        print(f"{c:>7}  CLOSED-LOOP   : within {cl['wsamp_within']:.2f}  logP {cl['wlogp_mean']:+.2f} (drop {cl['dwlogp_vs_base']:+.2f})  MAE {cl['mae_mean']:.1f}  KL {cl['kl_mean']:.2f}")
    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
