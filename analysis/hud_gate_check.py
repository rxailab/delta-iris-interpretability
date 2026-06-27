"""Path-1 mechanism check. Under each agent's wood GENERATIVE GATE (E3), does the
decoded HUD inventory still render wood (frame-carried -> no cascade, the original's
smoking gun) or is the HUD wood also removed (-> cascade, the seeds)?

Teacher-forced imagination (replay baseline actions so frames are step-aligned and
the only difference is the lesion). Per step we split the 64x64 decoded frame into
the HUD strip (rows y>=HUD_Y, the inventory/vitals) and the world region (y<HUD_Y),
and report mean-abs pixel difference baseline-vs-ablated (0-255 scale).

  frame-carried (no cascade): HUD MAE << world MAE  (HUD wood persists; f187/original)
  wood removed (cascade)    : HUD MAE ~ large       (HUD wood blanked; seed gates)
"""
from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from torch.distributions import Categorical
sys.path.insert(0, str(Path(__file__).resolve().parent))
from imagine_cascade import load_agent

HUD_Y = 49


def main():
    ap = argparse.ArgumentParser()
    for a in ("run", "rollouts", "obs", "sae", "features", "out"):
        ap.add_argument("--" + a, required=True, type=Path)
    ap.add_argument("--feat-id", required=True, type=int)
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
            return [int(t.flatten()[0].item()) for t in toks], self.obs

    env = Env(tk, wm, device)
    sae_blob = torch.load(args.sae, map_location=device, weights_only=False)
    decoder = sae_blob["state_dict"]["decoder.weight"].to(device).float()
    d = F.normalize(decoder[:, args.feat_id], dim=-1)
    print(f"focus={args.focus} gate feat#{args.feat_id}", flush=True)

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
        dd = hook_state["d"]
        return out if dd is None else out - (out @ dd).unsqueeze(-1) * dd
    handle = wm.transformer.blocks[args.target_block].register_forward_hook(hook)

    def roll(burn_obs, burn_act, direction, forced):
        hook_state["d"] = None
        env.reset_from_past(burn_obs.clone(), burn_act.clone()); ac.reset(n=1)
        with torch.no_grad():
            for t in range(args.burn_in): _ = ac.act(burn_obs[:, t])
        hook_state["d"] = direction
        frames = []; acts = []; action = forced[0]
        with torch.no_grad():
            for h in range(args.horizon):
                _, frame = env.step_rec(action)
                frames.append(frame.detach().float())     # [1,1,3,64,64], 0..1
                acts.append(int(action))
                if forced is not None and h + 1 < len(forced): action = forced[h + 1]
        hook_state["d"] = None
        return frames, acts

    hud_maes, world_maes = [], []
    t0 = time.time(); torch.manual_seed(args.seed)
    for mi, gi in enumerate(sel):
        ep = int(ep_ids[gi]); gl = int(gi - ep_start_row[ep])
        ep_obs = all_obs[obs_starts[ep]:obs_starts[ep+1]]; ep_act = actions[ep_start_row[ep]:ep_start_row[ep+1]]
        lo = gl - args.burn_in
        burn_obs = torch.from_numpy(ep_obs[lo:gl+1]).to(device).float().div(255).unsqueeze(0)
        burn_act = torch.from_numpy(ep_act[lo:gl]).to(device).unsqueeze(0)
        # baseline closed-loop (sample actions) -> record frames + the action sequence
        hook_state["d"] = None; env.reset_from_past(burn_obs.clone(), burn_act.clone()); ac.reset(n=1)
        with torch.no_grad():
            for t in range(args.burn_in): _ = ac.act(burn_obs[:, t])
        base_f = []; acts = []; action = int(ep_act[gl])
        with torch.no_grad():
            for h in range(args.horizon):
                _, frame = env.step_rec(action); base_f.append(frame.detach().float())
                acts.append(int(action))
                at, _ = ac.act(frame[:, 0], should_sample=True, temperature=1.0); action = int(at.item())
        abl_f, _ = roll(burn_obs, burn_act, d, acts)         # teacher-forced ablation, same actions
        for bf, af in zip(base_f, abl_f):
            diff = (bf - af).abs() * 255.0                   # [1,1,3,64,64]
            hud_maes.append(float(diff[..., HUD_Y:, :].mean().item()))
            world_maes.append(float(diff[..., :HUD_Y, :].mean().item()))
        if (mi + 1) % 10 == 0: print(f"  {mi+1}/{len(sel)}  ({(time.time()-t0)/60:.1f} min)", flush=True)
    handle.remove()

    hud = float(np.mean(hud_maes)); world = float(np.mean(world_maes))
    out = {"focus": args.focus, "feat": args.feat_id, "hud_y": HUD_Y, "n_moments": int(len(sel)),
           "hud_mae": hud, "world_mae": world, "ratio_world_over_hud": world / max(hud, 1e-6)}
    (args.out / "hud_gate.json").write_text(json.dumps(out, indent=2))
    print(f"\ngate f#{args.feat_id}: HUD MAE {hud:.2f}  world MAE {world:.2f}  "
          f"ratio world/HUD {out['ratio_world_over_hud']:.1f}")
    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
