"""Roll the trained Δ-IRIS agent in Crafter and log per-step tokens + achievements.

Output (single .npz under --out):
  - tokens:           (N_steps, 4)     int16 — per-step (slot0..3, code) from tokenizer(prev, action, next)
  - actions:          (N_steps,)       int8
  - rewards:          (N_steps,)       float32
  - ach_just_unlocked:(N_steps, 22)    bool  — True at the *step* where achievement i flipped 0→1
  - ach_cumulative:   (N_steps, 22)    bool  — episodic cumulative state AFTER the step
  - episode_ids:      (N_steps,)       int32
  - episode_returns:  (N_episodes,)    float32
  - episode_lengths:  (N_episodes,)    int32
  - achievement_names: list[str]       — Crafter's 22 achievement names, in order

Usage:
  python rollout_with_info.py \
    --run /mmfs1/scratch/.../delta-iris-full-21531393/hydra \
    --out /mmfs1/scratch/.../analysis-21531393/rollouts.npz \
    --n-episodes 500 --temperature 1.0 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import OmegaConf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-episodes", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=10000,
                    help="cap per-episode steps (Crafter's default is 10000)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--save-obs", type=Path, default=None,
                    help="if given, also dump every observation to this .npy "
                         "(uint8, shape=(N_steps+N_eps, 3, 64, 64); first obs of each ep + post-step obs)")
    args = ap.parse_args()

    run = args.run.resolve()
    sys.path.insert(0, str(run / "src"))
    args.out.parent.mkdir(parents=True, exist_ok=True)

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

    tokenizer = Tokenizer(instantiate(cfg.params.tokenizer))
    world_model = WorldModel(instantiate(cfg.params.world_model))
    actor_critic = ActorCritic(instantiate(cfg.params.actor_critic))
    agent = Agent(tokenizer, world_model, actor_critic).to(device).eval()
    agent.load(run / "checkpoints" / "last.pt", device=device,
               load_tokenizer=True, load_world_model=True, load_actor_critic=True, strict=False)

    # Build env directly so we keep info dict (the repo's wrappers drop it).
    import crafter, gym
    env = gym.wrappers.TimeLimit(crafter.Env(reward=True), max_episode_steps=args.max_steps)

    # Crafter's 22 achievements (the order Crafter uses internally — stable across versions).
    ach_names = sorted(crafter.constants.achievements)
    n_ach = len(ach_names)
    print(f"agent loaded. {n_ach} achievements: {', '.join(ach_names)}")

    # ----- accumulators ----------------------------------------------------
    all_tokens, all_actions, all_rewards = [], [], []
    all_ach_just, all_ach_cum = [], []
    all_ep_ids = []
    ep_returns, ep_lengths = [], []
    all_obs: list[np.ndarray] = []          # one (T+1, 3, 64, 64) uint8 per episode (only used if --save-obs)
    obs_episode_starts: list[int] = []     # global index where each episode's obs begin

    t0 = time.time()
    for ep_idx in range(args.n_episodes):
        agent.actor_critic.reset(n=1)
        obs = env.reset()                          # (64, 64, 3) uint8
        obs_list = [obs]
        act_list = []
        rew_list = []
        ach_cum_list = [[False] * n_ach]           # before step 0 nothing unlocked

        prev_ach = np.zeros(n_ach, dtype=bool)
        done = False
        steps = 0

        while not done:
            obs_t = torch.from_numpy(obs).float().div(255).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 3, 64, 64)
            with torch.no_grad():
                act_token, _ = agent.actor_critic.act(obs_t, should_sample=True, temperature=args.temperature)
            action = int(act_token.item())

            next_obs, reward, done, info = env.step(action)
            ach_dict = info.get("achievements", {})
            ach_now = np.array([bool(ach_dict.get(n, False)) for n in ach_names], dtype=bool)

            obs_list.append(next_obs)
            act_list.append(action)
            rew_list.append(float(reward))
            ach_cum_list.append(ach_now.tolist())

            obs = next_obs
            prev_ach = ach_now
            steps += 1

        agent.actor_critic.clear()

        # ----- tokenise the whole episode in one GPU shot ------------------
        if steps < 1:
            continue
        obs_arr = np.stack(obs_list, axis=0).astype(np.float32) / 255.0       # (T+1, 64, 64, 3)
        obs_t = torch.from_numpy(obs_arr).permute(0, 3, 1, 2).to(device)       # (T+1, 3, 64, 64)
        act_t = torch.tensor(act_list, dtype=torch.long, device=device)        # (T,)
        x1 = obs_t[:-1].unsqueeze(0)                                           # (1, T, 3, 64, 64)
        x2 = obs_t[1:].unsqueeze(0)
        a = act_t.unsqueeze(0)                                                 # (1, T)
        # call forward to also exercise the same code path as training
        with torch.no_grad():
            out = agent.tokenizer(x1, a, x2)
        tokens = out.tokens.squeeze(0).cpu().numpy().astype(np.int16)          # (T, 4)

        ach_cum = np.array(ach_cum_list, dtype=bool)                           # (T+1, n_ach)
        # achievement was JUST unlocked at step t if cum[t+1] != cum[t]
        ach_just = ach_cum[1:] & (~ach_cum[:-1])                               # (T, n_ach)
        ach_cum_post = ach_cum[1:]                                             # (T, n_ach) state after step

        all_tokens.append(tokens)
        all_actions.append(np.asarray(act_list, dtype=np.int8))
        all_rewards.append(np.asarray(rew_list, dtype=np.float32))
        all_ach_just.append(ach_just)
        all_ach_cum.append(ach_cum_post)
        all_ep_ids.append(np.full(len(act_list), ep_idx, dtype=np.int32))
        ep_returns.append(sum(rew_list))
        ep_lengths.append(len(act_list))
        if args.save_obs is not None:
            obs_episode_starts.append(sum(o.shape[0] for o in all_obs))
            all_obs.append(np.stack(obs_list, axis=0).transpose(0, 3, 1, 2).astype(np.uint8))

        elapsed = time.time() - t0
        if (ep_idx + 1) % max(1, args.n_episodes // 20) == 0 or ep_idx == args.n_episodes - 1:
            so_far = sum(ep_lengths)
            print(f"  ep {ep_idx+1}/{args.n_episodes}  "
                  f"steps {so_far}  return={np.mean(ep_returns):.2f}±{np.std(ep_returns):.2f}  "
                  f"{so_far/max(elapsed,1):.0f} steps/s  "
                  f"eta {(args.n_episodes - ep_idx - 1) * elapsed/(ep_idx+1)/60:.1f} min",
                  flush=True)

    print(f"\ncollected {len(ep_returns)} episodes, {sum(ep_lengths)} steps in {(time.time()-t0)/60:.1f} min")
    print(f"mean return: {np.mean(ep_returns):.3f} ± {np.std(ep_returns):.3f}")
    print(f"achievement unlock totals (out of {len(ep_returns)} episodes):")
    cum_per_ep = np.array([c[-1] for c in [np.concatenate([[np.zeros(n_ach, dtype=bool)], aj.cumsum(axis=0).astype(bool)]) for aj in all_ach_just]])
    unlocked = np.stack([aj.any(axis=0) for aj in all_ach_just])  # (N_ep, n_ach)
    for i, name in enumerate(ach_names):
        pct = unlocked[:, i].mean() * 100
        print(f"  {name:24s}  {unlocked[:, i].sum():4d}/{len(ep_returns)}  ({pct:5.1f}%)")

    np.savez_compressed(args.out,
        tokens=np.concatenate(all_tokens, axis=0),
        actions=np.concatenate(all_actions, axis=0),
        rewards=np.concatenate(all_rewards, axis=0),
        ach_just_unlocked=np.concatenate(all_ach_just, axis=0),
        ach_cumulative=np.concatenate(all_ach_cum, axis=0),
        episode_ids=np.concatenate(all_ep_ids, axis=0),
        episode_returns=np.asarray(ep_returns, dtype=np.float32),
        episode_lengths=np.asarray(ep_lengths, dtype=np.int32),
        achievement_names=np.asarray(ach_names),
    )
    print(f"\nsaved -> {args.out}")
    if args.save_obs is not None and all_obs:
        obs_concat = np.concatenate(all_obs, axis=0)        # (sum(T+1), 3, 64, 64) uint8
        np.savez_compressed(args.save_obs,
            obs=obs_concat,
            episode_starts=np.asarray(obs_episode_starts + [obs_concat.shape[0]], dtype=np.int64),
        )
        print(f"saved obs -> {args.save_obs}  ({obs_concat.nbytes / 1e9:.2f} GB raw)")


if __name__ == "__main__":
    main()
