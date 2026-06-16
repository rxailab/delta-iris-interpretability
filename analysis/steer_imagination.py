"""Activation-ADDITION / steering INSIDE imagination (the induce dual of ablation).

This is the suppress-AND-induce counterpart to analysis/imagine_ablation.py.
Where imagine_ablation.py PROJECTS OUT a concept direction at WM block 1 during
imagination (and seeds from UNLOCK moments) to *suppress* a concept, this script
ADDS alpha * d_hat at the same block-1 latent positions during imagination only
(and seeds from ORDINARY, non-unlock moments where the concept's detector codes
are baseline-rare) to *induce* the concept.

Pipeline (per target concept, reusing imagine_ablation verbatim where possible):
  1. Pick ORDINARY (non-unlock) burn-in moments in the rollout buffer where the
     concept's detector codes are baseline-rare (matched count, enough history,
     no unlock of the concept at the seed step).
  2. Burn the world model in on the REAL history [T-burn_in .. T] (frames+actions).
  3. Force the REAL action a_T at the first imagined step, then let the
     actor-critic act for the remaining steps. Imagine `horizon` steps, sampling
     the 4 latent codes per step and decoding frames with the tokenizer.
  4. Directions (block-1 summary space, the SAME space the ablation hook uses):
       sae    : matched SAE feature decoder column (unit-normalised, then added)
       cav    : matched CAV direction (w / std, unit-normalised, then added)
       random : random unit direction (specificity control)
     Each is scaled to alpha units of the residual-stream RMS norm: we add
     alpha * rms * d_hat where rms is the per-concept RMS of block-1 activations
     measured on the clean burn-in frames.
  5. Sweep alpha in {2,4,8}. Plus a baseline (no steering) per concept.
  6. Outcome per rollout: was any of the concept's top-2 detector codes (from the
     MI table) SAMPLED within the `horizon` imagined steps (induction success)?
  7. Induce-vs-suppress comparison: pair this experiment's per-(concept) SAE
     induction success against the SAE suppression success already measured by
     imagine_ablation (read from --suppress-effects) to test the cited
     "SAE features are easier to activate than to suppress" asymmetry.

Outputs under --out:
  steer_trials.json        raw per-rollout records
  steer_effects.json       aggregated P(detector) per (concept, direction, alpha) + CIs
  steer_summary.json       induce-vs-suppress comparison (SAE) + headline numbers
  steering.html            bar chart + induce-vs-suppress scatter + table
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


# --------------------------------------------------------------------------- #
# Reused verbatim from analysis/imagine_ablation.py                            #
# --------------------------------------------------------------------------- #
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


def boot_ci(arr, B=1000):
    """Bootstrap 95% CI for the mean (verbatim from imagine_ablation.py)."""
    if len(arr) == 0:
        return 0.0, 0.0
    rng2 = np.random.default_rng(0)
    means = [rng2.choice(arr, size=len(arr), replace=True).mean() for _ in range(B)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# --------------------------------------------------------------------------- #
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
    ap.add_argument("--suppress-effects", type=Path, default=None,
                    help="imagine_ablation's imagine_effects.json for the induce-vs-suppress comparison")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--burn-in", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--n-moments", type=int, default=20)
    ap.add_argument("--target-block", type=int, default=1)
    ap.add_argument("--alphas", default="2,4,8",
                    help="comma-separated steering strengths in units of residual-stream RMS")
    ap.add_argument("--baseline-rare-max", type=float, default=0.10,
                    help="max baseline detector rate for a seed step to count as 'rare'")
    ap.add_argument("--concepts", default="",
                    help="comma-separated achievement names; empty = all with strong SAE feature")
    ap.add_argument("--max-concepts", type=int, default=12,
                    help="cap number of concepts to keep runtime bounded")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    run = args.run.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    assert len(alphas) > 0, "need at least one alpha"
    print(f"alphas (RMS units): {alphas}", flush=True)

    print("loading agent...", flush=True)
    agent = load_agent(run, device)
    wm, tk, ac = agent.world_model, agent.tokenizer, agent.actor_critic
    max_blocks = wm.config.transformer_config.max_blocks
    embed_dim = wm.config.transformer_config.embed_dim
    assert args.burn_in + args.horizon <= max_blocks, \
        f"burn_in+horizon must fit the {max_blocks}-block KV cache without a flush"
    assert 0 <= args.target_block < wm.config.transformer_config.num_layers, "bad target block"

    from envs.world_model_env import WorldModelEnv

    class RecordingWMEnv(WorldModelEnv):
        """WorldModelEnv that also returns the 4 sampled codes per step.
        (Imagination loop copied verbatim from imagine_ablation.py.)"""
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

    # ----- directions (same selection logic as imagine_ablation) -----------
    sae_blob = torch.load(args.sae, map_location=device, weights_only=False)
    decoder = sae_blob["state_dict"]["decoder.weight"].to(device).float()   # (D, n_feat)
    assert decoder.shape[0] == embed_dim, \
        f"SAE D={decoder.shape[0]} != WM embed_dim={embed_dim}"
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

    def cav_direction(ach: str):
        name = f"just[{ach}]"
        if name not in cav_names: return None
        w = W_cav[cav_names.index(name)]
        if not np.any(w): return None
        d = w / sd_cav
        return torch.from_numpy(d).to(device).float()

    detectors = detector_codes_from_mi(args.mi_table, top_n=2)

    # ----- rollout data (verbatim block from imagine_ablation) -------------
    r = np.load(args.rollouts, allow_pickle=True)
    obs_npz = np.load(args.obs)
    all_obs = obs_npz["obs"]; obs_starts = obs_npz["episode_starts"]
    tokens = r["tokens"]; actions = r["actions"].astype(np.int64)
    aj = r["ach_just_unlocked"]; ep_ids = r["episode_ids"].astype(np.int64)
    ep_lens = r["episode_lengths"].astype(np.int64)
    ach_names = [str(x) for x in r["achievement_names"]]
    ep_start_row = np.concatenate(([0], np.cumsum(ep_lens)))
    assert tokens.ndim == 2, f"tokens expected (steps,K), got {tokens.shape}"
    K_codes = tokens.shape[1]
    print(f"rollouts: {tokens.shape[0]} steps, {ep_lens.shape[0]} episodes, K={K_codes}", flush=True)

    if args.concepts:
        wanted = [c.strip() for c in args.concepts.split(",") if c.strip()]
    else:
        wanted = [c for c in best_feat if c in detectors]
    targets = []
    for c in wanted:
        if c not in best_feat or c not in detectors:
            print(f"  skipping {c}: no strong SAE feature or no detector codes"); continue
        targets.append(c)
    targets = targets[:args.max_concepts]
    print(f"target concepts ({len(targets)}): {targets}", flush=True)

    # ----- per-step baseline detector rate (to find 'rare' ordinary seeds) --
    # detector fires at step g if any of the concept's top-2 (slot,code) match the
    # REAL codes at that step. baseline rate over the buffer tells us 'rare'.
    def concept_detector_hits(concept):
        det = detectors[concept]
        hit = np.zeros(tokens.shape[0], dtype=bool)
        for s, c in det:
            if 0 <= s < K_codes:
                hit |= (tokens[:, s] == c)
        return hit, det

    # ----- the steering hook (ADD instead of project-out) ------------------
    # During imagination only (burn-in is clean), ADD step_vec to EVERY position
    # of the block-1 output. step_vec already encodes alpha * rms * d_hat.
    # Matches imagine_ablation's hook, which operates on all positions of the
    # per-WM-call block-1 output during imagination.
    hook_state = {"vec": None}

    def hook(_m, _i, output):
        v = hook_state["vec"]
        if v is None:
            return output
        return output + v   # broadcast (D,) over (..., T, D)

    handle = wm.transformer.blocks[args.target_block].register_forward_hook(hook)

    # ----- RMS probe: capture block-1 output RMS on clean burn-in ----------
    rms_state = {"vals": []}

    def rms_hook(_m, _i, output):
        # output: (B, T, D). RMS over the feature dim D, averaged over positions.
        rms_state["vals"].append(
            output.detach().float().pow(2).mean(dim=-1).sqrt().mean().item())
        return output

    # ----- run --------------------------------------------------------------
    trials = []
    t0 = time.time()
    torch.manual_seed(args.seed)
    for ci, concept in enumerate(targets):
        a_idx = ach_names.index(concept)
        hit_real, det = concept_detector_hits(concept)
        d_sae = decoder[:, best_feat[concept]["feat"]].clone()
        d_cav = cav_direction(concept)
        d_rand = torch.randn(embed_dim, device=device)
        # unit-normalise all steering directions (scaling is via alpha * rms)
        d_sae_hat = F.normalize(d_sae, dim=-1)
        d_rand_hat = F.normalize(d_rand, dim=-1)
        d_cav_hat = F.normalize(d_cav, dim=-1) if d_cav is not None else None

        # ORDINARY (non-unlock) seed moments where detector codes are baseline-rare.
        # Use the same episodes that actually unlock this concept (so the seed
        # context is on-distribution for the concept), matching imagine_ablation's
        # ordinary-moment selection; require enough history and that the concept
        # is NOT just-unlocked at the seed step.
        ep_with_unlock = np.unique(ep_ids[np.where(aj[:, a_idx])[0]])
        cand = []
        for ep in ep_with_unlock:
            ep_start = int(ep_start_row[ep]); ep_end = int(ep_start_row[ep + 1])
            for g in range(ep_start + args.burn_in, ep_end):
                if aj[g, a_idx]:
                    continue                       # not an unlock step
                # baseline-rare: detector not already firing at/just-before seed
                lo = max(ep_start, g - 1)
                if hit_real[lo:g + 1].any():
                    continue
                cand.append(g)
        if len(cand) < 3:
            print(f"  {concept}: too few ordinary baseline-rare seeds, skipping", flush=True)
            continue
        # baseline rarity sanity: overall hit rate at the candidate seeds is low
        cand_arr = np.asarray(cand, dtype=np.int64)
        seed_rate = float(hit_real[cand_arr].mean()) if cand_arr.size else 0.0
        sel = rng.choice(cand_arr, size=min(args.n_moments, cand_arr.size), replace=False)

        # conditions: baseline once, then (direction, alpha) grid
        dir_specs = [("sae", d_sae_hat), ("random", d_rand_hat)]
        if d_cav_hat is not None:
            dir_specs.insert(1, ("cav", d_cav_hat))

        for g in sel:
            ep = int(ep_ids[g]); g_local = int(g - ep_start_row[ep])
            ep_obs = all_obs[obs_starts[ep]:obs_starts[ep + 1]]
            ep_act = actions[ep_start_row[ep]:ep_start_row[ep + 1]]
            lo = g_local - args.burn_in
            assert lo >= 0, "seed lacks burn-in history (should be filtered)"
            # burn-in: frames lo..g_local (burn_in+1 frames), actions lo..g_local-1
            burn_obs = torch.from_numpy(ep_obs[lo:g_local + 1]).to(device).float().div(255).unsqueeze(0)
            burn_act = torch.from_numpy(ep_act[lo:g_local]).to(device).unsqueeze(0)
            assert burn_obs.shape[1] == args.burn_in + 1, f"burn frames {burn_obs.shape}"
            assert burn_act.shape[1] == args.burn_in, f"burn acts {burn_act.shape}"
            real_a_T = int(ep_act[g_local])

            # ---- measure block-1 RMS on this clean burn-in (probe hook) ----
            rms_state["vals"] = []
            rms_handle = wm.transformer.blocks[args.target_block].register_forward_hook(rms_hook)
            hook_state["vec"] = None
            wm_env.reset_from_past(burn_obs.clone(), burn_act.clone())
            rms_handle.remove()
            rms = float(np.mean(rms_state["vals"])) if rms_state["vals"] else 1.0
            assert rms > 0 and np.isfinite(rms), f"bad rms {rms}"

            # build the (condition_name, vec) list for this seed
            conditions = [("baseline", "baseline", 0.0, None)]
            for dname, dhat in dir_specs:
                for alpha in alphas:
                    vec = (alpha * rms) * dhat        # (D,)
                    cond_name = f"{dname}@a{alpha:g}"
                    conditions.append((cond_name, dname, alpha, vec))

            for cond_name, dname, alpha, vec in conditions:
                hook_state["vec"] = None             # burn-in is always clean
                wm_env.reset_from_past(burn_obs.clone(), burn_act.clone())
                ac.reset(n=1)
                with torch.no_grad():
                    for t in range(args.burn_in):
                        _ = ac.act(burn_obs[:, t])
                hook_state["vec"] = vec               # steering ON for imagination

                step_codes, rewards = [], []
                action = real_a_T                     # force real action at seed step
                with torch.no_grad():
                    for h in range(args.horizon):
                        codes, rew, frame = wm_env.step_recorded(action)
                        assert len(codes) == K_codes, f"got {len(codes)} codes"
                        step_codes.append(codes); rewards.append(rew)
                        act_tok, _ = ac.act(frame[:, 0], should_sample=True, temperature=1.0)
                        action = int(act_tok.item())
                hook_state["vec"] = None

                hit_steps = [h for h, codes in enumerate(step_codes)
                             if any(codes[s] == c for s, c in det)]
                trials.append(dict(
                    concept=concept, condition=cond_name,
                    direction=dname, alpha=float(alpha),
                    episode=ep, ep_step=g_local, rms=rms, seed_rate=seed_rate,
                    detector_first=bool(hit_steps and hit_steps[0] == 0),
                    detector_within=bool(hit_steps),
                    first_hit_step=(hit_steps[0] if hit_steps else None),
                    reward_sum=float(np.sum(rewards)),
                ))

        n_done = len([t for t in trials if t["concept"] == concept])
        print(f"  [{ci + 1}/{len(targets)}] {concept}: {n_done} rollouts  "
              f"(seed_rate={seed_rate:.3f}, {(time.time() - t0) / 60:.1f} min elapsed)", flush=True)

    handle.remove()
    print(f"\n{len(trials)} imagination rollouts in {(time.time() - t0) / 60:.1f} min", flush=True)
    (args.out / "steer_trials.json").write_text(json.dumps(trials, indent=2))

    # ----- aggregate: P(detector within horizon) per (concept,dir,alpha) ----
    summary = []
    cond_keys = sorted({(t["direction"], t["alpha"]) for t in trials})
    for concept in sorted({t["concept"] for t in trials}):
        for dname, alpha in cond_keys:
            sub = [t for t in trials if t["concept"] == concept
                   and t["direction"] == dname and t["alpha"] == alpha]
            if not sub: continue
            first = np.array([t["detector_first"] for t in sub], dtype=float)
            within = np.array([t["detector_within"] for t in sub], dtype=float)
            rew = np.array([t["reward_sum"] for t in sub], dtype=float)
            lo_f, hi_f = boot_ci(first); lo_w, hi_w = boot_ci(within)
            summary.append(dict(concept=concept, direction=dname, alpha=float(alpha),
                                condition=(f"{dname}@a{alpha:g}" if dname != "baseline" else "baseline"),
                                n=len(sub),
                                p_first=float(first.mean()), p_first_lo=lo_f, p_first_hi=hi_f,
                                p_within=float(within.mean()), p_within_lo=lo_w, p_within_hi=hi_w,
                                reward_mean=float(rew.mean())))
    (args.out / "steer_effects.json").write_text(json.dumps(summary, indent=2))

    # ----- induce-vs-suppress comparison (SAE) ------------------------------
    # induction = best SAE-steering p_within across the alpha sweep (per concept).
    # baseline = the no-steering p_within for that concept (this experiment).
    # suppression = how far imagine_ablation's SAE condition drops p_within below
    #   its own baseline, i.e. suppress_drop = base_sup - sae_sup.
    induce_best = {}
    base_within = {}
    for concept in sorted({s["concept"] for s in summary}):
        recs = [s for s in summary if s["concept"] == concept and s["direction"] == "sae"]
        if recs:
            best = max(recs, key=lambda s: s["p_within"])
            induce_best[concept] = dict(alpha=best["alpha"], p_within=best["p_within"],
                                        p_within_lo=best["p_within_lo"], p_within_hi=best["p_within_hi"])
        brec = next((s for s in summary if s["concept"] == concept and s["direction"] == "baseline"), None)
        if brec:
            base_within[concept] = brec["p_within"]

    suppress = {}
    if args.suppress_effects is not None and args.suppress_effects.exists():
        sup = json.loads(args.suppress_effects.read_text())
        sup_base = {r["concept"]: r["p_within"] for r in sup if r["condition"] == "baseline"}
        sup_sae = {r["concept"]: r["p_within"] for r in sup if r["condition"] == "sae"}
        for c in sup_base:
            if c in sup_sae:
                suppress[c] = dict(base=sup_base[c], sae=sup_sae[c],
                                   drop=sup_base[c] - sup_sae[c])
    else:
        print("WARN: no suppress-effects file; induce-vs-suppress comparison will be partial", flush=True)

    comparison = []
    for c in sorted(induce_best):
        ind = induce_best[c]
        b_ind = base_within.get(c, 0.0)
        # induction gain = best SAE steering p_within minus this experiment's baseline
        induce_gain = ind["p_within"] - b_ind
        rec = dict(concept=c,
                   induce_baseline=b_ind,
                   induce_best_alpha=ind["alpha"],
                   induce_p_within=ind["p_within"],
                   induce_gain=induce_gain)
        if c in suppress:
            rec.update(suppress_baseline=suppress[c]["base"],
                       suppress_p_within=suppress[c]["sae"],
                       suppress_drop=suppress[c]["drop"],
                       asymmetry_gain_minus_drop=induce_gain - suppress[c]["drop"])
        comparison.append(rec)

    paired = [r for r in comparison if "suppress_drop" in r]
    headline = dict(
        n_concepts=len(comparison),
        n_paired=len(paired),
        mean_induce_gain=float(np.mean([r["induce_gain"] for r in comparison])) if comparison else 0.0,
        mean_suppress_drop=float(np.mean([r["suppress_drop"] for r in paired])) if paired else 0.0,
        mean_asymmetry=float(np.mean([r["asymmetry_gain_minus_drop"] for r in paired])) if paired else 0.0,
        note=("positive mean_asymmetry => SAE features are easier to ACTIVATE (induce) "
              "than to SUPPRESS, consistent with the cited asymmetry"),
        alphas=alphas,
    )
    (args.out / "steer_summary.json").write_text(
        json.dumps(dict(headline=headline, comparison=comparison), indent=2))
    print(f"headline: {json.dumps(headline)}", flush=True)

    # ----- render -----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    concepts = sorted({s["concept"] for s in summary})
    # bar chart: P(within) for baseline + sae@each-alpha + cav@best + random@best
    dirs_plot = ["baseline", "sae", "cav", "random"]
    colors = {"baseline": "#2ca02c", "sae": "#d62728", "cav": "#9467bd", "random": "#aaaaaa"}

    fig, axes = plt.subplots(1, 2, figsize=(2.4 + 1.2 * len(concepts), 5.4))
    ax = axes[0]
    x = np.arange(len(concepts)); width = 0.2
    for j, dname in enumerate(dirs_plot):
        ys, ylo, yhi = [], [], []
        for c in concepts:
            if dname == "baseline":
                rec = next((s for s in summary if s["concept"] == c and s["direction"] == "baseline"), None)
            else:
                recs = [s for s in summary if s["concept"] == c and s["direction"] == dname]
                rec = max(recs, key=lambda s: s["p_within"]) if recs else None
            if rec:
                ys.append(rec["p_within"])
                ylo.append(rec["p_within"] - rec["p_within_lo"])
                yhi.append(rec["p_within_hi"] - rec["p_within"])
            else:
                ys.append(0); ylo.append(0); yhi.append(0)
        ax.bar(x + (j - 1.5) * width, ys, width, label=dname, color=colors[dname],
               yerr=[ylo, yhi], capsize=2, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(concepts, rotation=30, ha="right", fontsize=8)
    ax.set_title(f"P(detector within {args.horizon} imagined steps)\n"
                 f"best over alpha in {alphas}", fontsize=10)
    ax.set_ylabel("induction probability"); ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)

    # scatter: induce gain (x) vs suppress drop (y) per concept (SAE)
    ax2 = axes[1]
    if paired:
        gx = [r["induce_gain"] for r in paired]
        dy = [r["suppress_drop"] for r in paired]
        ax2.scatter(gx, dy, c="#d62728", s=40, edgecolor="black", linewidth=0.5, zorder=3)
        for r in paired:
            ax2.annotate(r["concept"], (r["induce_gain"], r["suppress_drop"]),
                         fontsize=6.5, xytext=(3, 3), textcoords="offset points")
        lim = max(0.05, max([abs(v) for v in gx + dy]) * 1.15)
        ax2.plot([-lim, lim], [-lim, lim], "--", color="#888", lw=0.8, zorder=1)
        ax2.set_xlim(-0.05, lim); ax2.set_ylim(-0.05, lim)
    ax2.axhline(0, color="black", lw=0.5); ax2.axvline(0, color="black", lw=0.5)
    ax2.set_xlabel("induction gain (SAE add - baseline)")
    ax2.set_ylabel("suppression drop (baseline - SAE project-out)")
    ax2.set_title("Induce vs suppress (SAE)\nbelow diagonal => easier to activate", fontsize=10)
    ax2.grid(alpha=0.3)
    fig.suptitle(f"Activation-addition steering in imagination "
                 f"(burn-in {args.burn_in}, block {args.target_block}, ordinary seeds)", fontsize=11)
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=140); plt.close(fig)
    img = base64.b64encode(buf.getvalue()).decode("ascii")

    rows = []
    for s in sorted(summary, key=lambda x: (x["concept"], x["direction"], x["alpha"])):
        rows.append(f"<tr><td class=label>{s['concept']}</td><td>{s['condition']}</td>"
                    f"<td>{s['n']}</td>"
                    f"<td>{s['p_first']:.2f} [{s['p_first_lo']:.2f},{s['p_first_hi']:.2f}]</td>"
                    f"<td>{s['p_within']:.2f} [{s['p_within_lo']:.2f},{s['p_within_hi']:.2f}]</td>"
                    f"<td>{s['reward_mean']:+.2f}</td></tr>")
    crows = []
    for r in comparison:
        sd = f"{r['suppress_drop']:+.2f}" if "suppress_drop" in r else "—"
        asym = f"{r['asymmetry_gain_minus_drop']:+.2f}" if "asymmetry_gain_minus_drop" in r else "—"
        crows.append(f"<tr><td class=label>{r['concept']}</td>"
                     f"<td>{r['induce_baseline']:.2f}</td>"
                     f"<td>a{r['induce_best_alpha']:g}</td>"
                     f"<td>{r['induce_p_within']:.2f}</td>"
                     f"<td>{r['induce_gain']:+.2f}</td>"
                     f"<td>{sd}</td><td>{asym}</td></tr>")
    html = ("<!doctype html><meta charset=utf-8><title>steering in imagination</title>"
            "<style>body{font:14px ui-monospace,Menlo,Consolas,monospace;background:#0d1117;"
            "color:#c9d1d9;padding:18px;}h1{color:#79c0ff;font-size:18px;}"
            "h2{color:#d2a8ff;font-size:15px;margin:14px 0 6px;}"
            ".dim{color:#8b949e;}img{max-width:100%;background:white;border-radius:6px;}"
            "table{border-collapse:collapse;font-size:12.5px;margin-top:10px;width:100%;}"
            "th,td{padding:4px 8px;border-bottom:1px solid #21262d;text-align:right;}"
            "td.label{text-align:left;}th{color:#8b949e;font-size:11px;text-transform:uppercase;}</style>"
            f"<h1>Activation-addition steering in imagination (induce dual of ablation)</h1>"
            f"<div class=dim>{len(trials)} rollouts; steering = ADD alpha*rms*d_hat at WM block "
            f"{args.target_block} during imagination only (burn-in clean); ordinary baseline-rare seeds; "
            f"real action forced at seed step; alpha in {alphas} (residual-stream RMS units); "
            f"detector codes = top-2 (slot,code) per achievement from the MI table.</div>"
            f"<div class=dim>headline: mean induce gain {headline['mean_induce_gain']:+.3f}; "
            f"mean suppress drop {headline['mean_suppress_drop']:+.3f}; "
            f"mean asymmetry (gain - drop) {headline['mean_asymmetry']:+.3f} "
            f"over {headline['n_paired']} paired concepts.</div>"
            f"<img src='data:image/png;base64,{img}'>"
            "<h2>Induce-vs-suppress (SAE)</h2>"
            "<table><tr><th>concept</th><th>induce base</th><th>best alpha</th>"
            "<th>induce P(within)</th><th>induce gain</th><th>suppress drop</th>"
            "<th>asymmetry</th></tr>" + "".join(crows) + "</table>"
            "<h2>Full effects table</h2>"
            "<table><tr><th>concept</th><th>condition</th><th>n</th>"
            "<th>P(first step) [CI]</th><th>P(within horizon) [CI]</th><th>mean reward sum</th></tr>"
            + "".join(rows) + "</table>")
    (args.out / "steering.html").write_text(html, encoding="utf-8")
    print(f"wrote {args.out / 'steering.html'}", flush=True)


if __name__ == "__main__":
    main()
