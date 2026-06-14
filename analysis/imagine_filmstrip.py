"""Paired imagined trajectories: baseline vs SAE-feature-ablated, decoded to frames.

For a target concept, we burn the world model in on real history ending just
before an achievement unlock, then roll the agent's imagination forward
`horizon` steps TWICE from the same starting point and the same random seed:

  * baseline : no intervention
  * ablated  : the concept's best SAE decoder direction is projected out of the
               block-1 residual stream at every position, every step

Because the two runs share burn-in and RNG seed, any divergence is attributable
to the ablation, not to sampling noise. At each imagined step we decode the
frame with the tokenizer, record the 4 sampled codes, the predicted reward, and
whether the concept's detector codes (top-2 from the MI table) were sampled.

Output (one .npz per concept under --out):
  frames_baseline : (M, T, 64, 64, 3) uint8
  frames_ablated  : (M, T, 64, 64, 3) uint8
  burnin_last     : (M, 64, 64, 3) uint8   last real frame before imagination
  hit_baseline    : (M, T) bool            detector code sampled at step t
  hit_ablated     : (M, T) bool
  reward_baseline : (M, T) float32
  reward_ablated  : (M, T) float32
  moments         : (M,) int32             global step index of each unlock
  meta            : json string            concept, feature id, detector codes, params

Run via runs/imagine_filmstrip.sbatch.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.distributions.categorical import Categorical


def load_agent(run: Path, device):
    sys.path.insert(0, str(run / "src"))
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    cfg = OmegaConf.load(run / ".hydra" / "config.yaml")
    for k in ("tokenizer", "world_model"):
        if cfg.params[k].num_actions is None:
            cfg.params[k].num_actions = 17
    if cfg.params.actor_critic.model.num_actions is None:
        cfg.params.actor_critic.model.num_actions = 17
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


def detector_codes(mi_table: Path, concept: str, top_n: int = 2):
    per = defaultdict(list)
    for r in json.loads(mi_table.read_text()):
        per[r["achievement"]].append((r["lift"], r["slot"], r["code"]))
    rows = sorted(per[concept], reverse=True)[:top_n]
    return [(s, c) for _, s, c in rows]


def best_feature(features_json: Path, concept: str) -> int:
    feats = json.loads(features_json.read_text())["features"]
    best, best_lift = None, -1
    for f in feats:
        a = f.get("best_ach")
        if a and a["name"] == concept and f["density"] >= 0.001 and a["lift"] > best_lift:
            best, best_lift = f["feature"], a["lift"]
    if best is None:
        raise SystemExit(f"no SAE feature found for {concept}")
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--rollouts", required=True, type=Path)
    ap.add_argument("--obs", required=True, type=Path)
    ap.add_argument("--sae", required=True, type=Path)
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--mi-table", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--concept", default="collect_wood")
    ap.add_argument("--offtarget-concept", default="collect_iron",
                    help="concept whose best SAE feature serves as the off-target (live) control")
    ap.add_argument("--n-moments", type=int, default=6)
    ap.add_argument("--burn-in", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--target-block", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    run = args.run.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    agent = load_agent(run, device)
    wm, tk, ac = agent.world_model, agent.tokenizer, agent.actor_critic
    max_blocks = wm.config.transformer_config.max_blocks
    assert args.burn_in + args.horizon < max_blocks, \
        f"burn_in+horizon ({args.burn_in}+{args.horizon}) must stay under the {max_blocks}-block KV cache"

    from envs.world_model_env import WorldModelEnv
    from utils import compute_softmax_over_buckets, symexp
    from einops import rearrange as rea

    feat_id = best_feature(args.features, args.concept)
    sae = torch.load(args.sae, map_location=device, weights_only=False)
    dec = sae["state_dict"]["decoder.weight"]
    direction = dec[:, feat_id].clone().to(device).float()
    # fixed random unit direction (direction-norm specificity control)
    rgen = torch.Generator(device=device).manual_seed(12345)
    rand_dir = torch.randn(direction.shape, generator=rgen, device=device)
    # off-target SAE feature: a strong, live direction for an UNRELATED concept.
    # This is the magnitude-matched specificity control the random unit vector
    # cannot provide (a real dictionary direction, not a near-inert random one).
    off_concept = args.offtarget_concept
    if off_concept == args.concept:
        off_concept = "collect_wood" if args.concept != "collect_wood" else "collect_iron"
    off_feat = best_feature(args.features, off_concept)
    off_dir = dec[:, off_feat].clone().to(device).float()
    dets = detector_codes(args.mi_table, args.concept)
    print(f"concept={args.concept}  SAE feature f{feat_id}  detector codes={dets}  "
          f"off-target={off_concept} f{off_feat}", flush=True)

    class FilmEnv(WorldModelEnv):
        @torch.no_grad()
        def step_film(self, action, direction=None):
            if not isinstance(action, torch.Tensor):
                action = torch.tensor(action, dtype=torch.long).reshape(-1, 1).to(self.device)
            a = self.world_model.act_emb(action)
            if self.last_latent_token_emb is None:
                inp = a if self.x is None else torch.cat((self.x, a), dim=1)
            else:
                inp = torch.cat((self.last_latent_token_emb, self.x, a), dim=1)
            handle = None
            if direction is not None:
                d = F.normalize(direction, dim=-1)
                def hook(_m, _i, out):
                    return out - (out @ d).unsqueeze(-1) * d
                handle = self.world_model.transformer.blocks[args.target_block].register_forward_hook(hook)
            try:
                out = self.world_model(inp, use_kv_cache=True)
                rew = float(symexp(compute_softmax_over_buckets(out.logits_rewards)).flatten()[0].item())
                toks = [Categorical(logits=out.logits_latents).sample()]
                for _ in range(self.tokenizer.config.num_tokens - 1):
                    emb = self.world_model.latents_emb(toks[-1])
                    out = self.world_model(emb, use_kv_cache=True)
                    toks.append(Categorical(logits=out.logits_latents).sample())
            finally:
                if handle is not None:
                    handle.remove()
            self.last_latent_token_emb = self.world_model.latents_emb(toks[-1])
            q = self.tokenizer.quantizer.embed_tokens(torch.stack(toks, dim=-1))
            self.obs = self.tokenizer.decode(
                self.obs, action,
                rea(q, 'b t (h w) (k l e) -> b t e (h k) (w l)',
                    h=self.tokenizer.tokens_grid_res, k=self.tokenizer.token_res, l=self.tokenizer.token_res),
                should_clamp=True)
            self.x = rea(self.world_model.frame_cnn(self.obs), 'b 1 k e -> b k e')
            codes = [int(t.flatten()[0].item()) for t in toks]
            frame = (self.obs[0, 0].permute(1, 2, 0).clamp(0, 1).mul(255)
                     .round().byte().cpu().numpy())  # (64,64,3) uint8
            return codes, rew, frame

    env = FilmEnv(tk, wm, device)

    # data
    r = np.load(args.rollouts, allow_pickle=True)
    obs_npz = np.load(args.obs)
    all_obs, obs_starts = obs_npz["obs"], obs_npz["episode_starts"]
    actions = r["actions"].astype(np.int64)
    aj = r["ach_just_unlocked"]
    ep_ids = r["episode_ids"].astype(np.int64)
    ep_lens = r["episode_lengths"].astype(np.int64)
    ach_names = [str(x) for x in r["achievement_names"]]
    ep_start_row = np.concatenate(([0], np.cumsum(ep_lens)))
    a_idx = ach_names.index(args.concept)

    rng = np.random.default_rng(args.seed)
    unlock = [g for g in np.where(aj[:, a_idx])[0]
              if (g - ep_start_row[ep_ids[g]]) >= args.burn_in]
    sel = rng.choice(unlock, size=min(args.n_moments, len(unlock)), replace=False)
    sel.sort()
    print(f"{len(unlock)} eligible unlock moments; using {len(sel)}", flush=True)

    T, M = args.horizon, len(sel)
    CONDS = ("baseline", "ablated", "offtarget", "random")
    FR = {c: np.zeros((M, T, 64, 64, 3), np.uint8) for c in CONDS}
    Hh = {c: np.zeros((M, T), bool) for c in CONDS}
    Rr = {c: np.zeros((M, T), np.float32) for c in CONDS}
    burn = np.zeros((M, 64, 64, 3), np.uint8)

    def hit(codes):
        return any(codes[s] == c for s, c in dets)

    conds = [("baseline", None), ("ablated", direction),
             ("offtarget", off_dir), ("random", rand_dir)]
    for mi, g in enumerate(sel):
        ep = int(ep_ids[g]); gl = int(g - ep_start_row[ep])
        lo = gl - args.burn_in
        ep_obs = all_obs[obs_starts[ep]:obs_starts[ep + 1]]
        ep_act = actions[ep_start_row[ep]:ep_start_row[ep + 1]]
        burn_obs = torch.from_numpy(ep_obs[lo:gl + 1]).to(device).float().div(255).unsqueeze(0)
        burn_act = torch.from_numpy(ep_act[lo:gl]).to(device).unsqueeze(0)
        real_a = int(ep_act[gl])
        burn[mi] = ep_obs[gl].transpose(1, 2, 0)

        for cond, d in conds:
            torch.manual_seed(args.seed * 1000 + mi)   # paired seed across all conditions
            env.reset_from_past(burn_obs.clone(), burn_act.clone())
            ac.reset(n=1)
            with torch.no_grad():
                for tt in range(args.burn_in):
                    _ = ac.act(burn_obs[:, tt])
            action = real_a
            with torch.no_grad():
                for tstep in range(T):
                    codes, rew, frame = env.step_film(action, direction=d)
                    FR[cond][mi, tstep] = frame
                    Hh[cond][mi, tstep] = hit(codes)
                    Rr[cond][mi, tstep] = rew
                    act_tok, _ = ac.act(
                        torch.from_numpy(frame).to(device).float().div(255)
                        .permute(2, 0, 1).unsqueeze(0))
                    action = int(act_tok.item())
        print(f"  moment {mi+1}/{M} (ep {ep}, step {gl}): hits "
              f"baseline {int(Hh['baseline'][mi].sum())}, "
              f"ablated {int(Hh['ablated'][mi].sum())}, "
              f"offtarget {int(Hh['offtarget'][mi].sum())}, "
              f"random {int(Hh['random'][mi].sum())}", flush=True)

    meta = dict(concept=args.concept, feature=int(feat_id), detector_codes=dets,
                offtarget_concept=off_concept, offtarget_feature=int(off_feat),
                burn_in=args.burn_in, horizon=args.horizon, target_block=args.target_block,
                n_moments=int(M), seed=args.seed, moments=[int(x) for x in sel])
    np.savez_compressed(args.out / f"filmstrip_{args.concept}.npz",
                        frames_baseline=FR["baseline"], frames_ablated=FR["ablated"],
                        frames_offtarget=FR["offtarget"], frames_random=FR["random"], burnin_last=burn,
                        hit_baseline=Hh["baseline"], hit_ablated=Hh["ablated"],
                        hit_offtarget=Hh["offtarget"], hit_random=Hh["random"],
                        reward_baseline=Rr["baseline"], reward_ablated=Rr["ablated"],
                        reward_offtarget=Rr["offtarget"], reward_random=Rr["random"],
                        moments=np.array(sel, np.int32), meta=json.dumps(meta))
    print(f"saved {args.out / ('filmstrip_' + args.concept + '.npz')}", flush=True)


if __name__ == "__main__":
    main()
