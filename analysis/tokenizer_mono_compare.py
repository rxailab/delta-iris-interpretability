"""Unified code -> achievement monosemanticity analysis for one tokenizer.

Runs the SAME analysis used for Δ-IRIS on any of three tokenizer kinds so the
results are directly comparable. For each step in the 150-episode rollout set we
encode the relevant frame(s) into discrete codes, then correlate every
(slot, code) with the per-step just-unlocked achievement labels.

  kind = deltairis : encode (x_t, a_t, x_{t+1}) with the trained Δ-IRIS tokenizer
                     (4 codes, codebook 1024) -- the transition/delta codes
  kind = frameonly : encode x_{t+1} with the frame-only ablation tokenizer
                     (4 codes, codebook 1024)
  kind = iris      : encode x_{t+1} with the original IRIS tokenizer
                     (16 codes, codebook 512)

All three are correlated with ach_just_unlocked at step t (the event that the
transition x_t -> x_{t+1} produced).

Outputs <out>/mono_<kind>.json with:
  per_slot: active codes, entropy (bits), max share
  per_achievement: best detector (slot, code, P(a|c), lift, n_co, n_code)
  monosemanticity: mean/median best-P(a|c) and best-lift over achievements;
                   purity (usage-weighted mean over codes of max_a P(a|c));
                   n achievements with a strong detector (P(a|c) >= 0.5)
  meta
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


def load_deltairis(run, device):
    sys.path.insert(0, str(run / "src"))
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    cfg = OmegaConf.load(run / ".hydra" / "config.yaml")
    if cfg.params.tokenizer.num_actions is None:
        cfg.params.tokenizer.num_actions = 17
    OmegaConf.resolve(cfg)
    from models.tokenizer import Tokenizer
    tok = Tokenizer(instantiate(cfg.params.tokenizer)).to(device).eval()
    ckpt = torch.load(run / "checkpoints" / "last.pt", map_location=device, weights_only=False)
    tok.load_state_dict({k[len("tokenizer."):]: v for k, v in ckpt.items()
                         if k.startswith("tokenizer.")}, strict=False)
    return tok


@torch.no_grad()
def encode_deltairis(tok, x1, a, x2):
    z = tok.encode(x1, a, x2)
    z = rearrange(z, 'b t c (h k) (w l) -> b t (h w) (k l c)',
                  h=tok.tokens_grid_res, w=tok.tokens_grid_res)
    return tok.quantizer(z).tokens.squeeze(0).cpu().numpy()   # (B, 4)


def build_frameonly(run, ckpt_path, device):
    sys.path.insert(0, str(run / "src"))
    import torch.nn as nn
    from models.convnet import FrameEncoder, FrameDecoder, FrameCnnConfig
    from models.tokenizer.quantizer import Quantizer
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = blob["config"]
    grid, tr, ld, cd = c["grid"], c["token_res"], c["latent"], c["code_dim"]

    class FrameOnlyTokenizer(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = FrameEncoder(FrameCnnConfig(3, ld, c["num_ch"], c["mult"], c["down"]))
            self.decoder = FrameDecoder(FrameCnnConfig(3, cd, c["num_ch"], c["mult"], c["down"]))
            self.quantizer = Quantizer(c["codebook"], cd, input_dim=c["input_dim"],
                                       max_codebook_updates_with_revival=0)

        @torch.no_grad()
        def encode_tokens(self, x):  # x (b,1,3,64,64)
            z = self.encoder(x)
            z = rearrange(z, 'b t c (h k) (w l) -> b t (h w) (k l c)', h=grid, w=grid)
            return self.quantizer(z).tokens.squeeze(0).cpu().numpy()  # (b,4)

    tok = FrameOnlyTokenizer().to(device).eval()
    tok.load_state_dict(blob["state_dict"], strict=True)
    return tok


def build_iris(iris_src, ckpt_path, device):
    sys.path.insert(0, str(iris_src))
    from models.tokenizer.nets import Encoder, Decoder, EncoderDecoderConfig
    from models.tokenizer.tokenizer import Tokenizer
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = EncoderDecoderConfig(resolution=64, in_channels=3, z_channels=512, ch=64,
                               ch_mult=[1, 1, 1, 1, 1], num_res_blocks=2,
                               attn_resolutions=[8, 16], out_ch=3, dropout=0.0)
    tok = Tokenizer(vocab_size=512, embed_dim=512,
                    encoder=Encoder(cfg), decoder=Decoder(cfg), with_lpips=False).to(device).eval()
    sd = {k: v for k, v in blob["state_dict"].items() if not k.startswith("lpips.")}
    tok.load_state_dict(sd, strict=False)
    return tok


@torch.no_grad()
def encode_iris(tok, x2):
    # x2: (B,3,64,64) in [0,1]; preprocess to [-1,1] inside encode via should_preprocess
    out = tok.encode(x2, should_preprocess=True)
    return out.tokens.cpu().numpy()   # (B, 16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["deltairis", "frameonly", "iris"])
    ap.add_argument("--deltairis-src", required=True, type=Path)
    ap.add_argument("--iris-src", type=Path, default=None)
    ap.add_argument("--ckpt", type=Path, default=None, help="frameonly/iris tokenizer .pt")
    ap.add_argument("--rollouts", required=True, type=Path)
    ap.add_argument("--obs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    run = args.deltairis_src.resolve()

    if args.kind == "deltairis":
        tok = load_deltairis(run, device); K, C = 4, 1024
    elif args.kind == "frameonly":
        tok = build_frameonly(run, args.ckpt, device); K, C = 4, 1024
    else:
        tok = build_iris(args.iris_src.resolve(), args.ckpt, device); K, C = 16, 512
    print(f"kind={args.kind}  K_slots={K}  codebook={C}", flush=True)

    r = np.load(args.rollouts, allow_pickle=True)
    obs_npz = np.load(args.obs)
    all_obs, obs_starts = obs_npz["obs"], obs_npz["episode_starts"]
    actions = r["actions"].astype(np.int64)
    aj = r["ach_just_unlocked"]
    ep_ids = r["episode_ids"].astype(np.int64)
    ep_lens = r["episode_lengths"].astype(np.int64)
    ach_names = [str(x) for x in r["achievement_names"]]
    ep_start_row = np.concatenate(([0], np.cumsum(ep_lens)))
    n_ep = ep_lens.shape[0]; A = aj.shape[1]

    counts = np.zeros((K, C), dtype=np.int64)
    # co-occurrence of (slot, code) with each just-unlocked achievement
    co = np.zeros((K, C, A), dtype=np.int64)
    n_steps = 0
    t0 = time.time()

    for ep in range(n_ep):
        T = int(ep_lens[ep])
        ep_obs = all_obs[obs_starts[ep]:obs_starts[ep + 1]]      # (T+1,3,64,64) uint8
        ep_act = actions[ep_start_row[ep]:ep_start_row[ep + 1]]  # (T,)
        ep_aj = aj[ep_start_row[ep]:ep_start_row[ep + 1]]        # (T, A)
        x1 = torch.from_numpy(ep_obs[:-1]).to(device).float().div(255)   # (T,3,64,64)
        x2 = torch.from_numpy(ep_obs[1:]).to(device).float().div(255)
        a = torch.from_numpy(ep_act).to(device)
        for s in range(0, T, args.batch):
            e = min(s + args.batch, T)
            if args.kind == "deltairis":
                toks = encode_deltairis(tok, x1[s:e].unsqueeze(0), a[s:e].unsqueeze(0), x2[s:e].unsqueeze(0))
            elif args.kind == "frameonly":
                toks = tok.encode_tokens(x2[s:e].unsqueeze(0))
            else:
                toks = encode_iris(tok, x2[s:e])
            lab = ep_aj[s:e]                                     # (B, A)
            for k in range(K):
                np.add.at(counts[k], toks[:, k], 1)
                # co-occurrence: for each achievement, add token histogram over rows where it fired
                for ai in range(A):
                    rows = np.where(lab[:, ai])[0]
                    if rows.size:
                        np.add.at(co[k, :, ai], toks[rows, k], 1)
            n_steps += (e - s)
        if (ep + 1) % max(1, n_ep // 10) == 0:
            print(f"  ep {ep+1}/{n_ep}  steps {n_steps}  {n_steps/max(time.time()-t0,1):.0f}/s", flush=True)

    base_rate = aj[ : ep_start_row[n_ep]].mean(axis=0)   # P(achievement just-unlocks) per step
    N = n_steps

    # ----- metrics --------------------------------------------------------
    per_slot = []
    for k in range(K):
        ck = counts[k]; tot = ck.sum()
        p = ck[ck > 0] / max(tot, 1)
        H = float(-(p * np.log2(p)).sum()) if tot > 0 else 0.0
        per_slot.append(dict(slot=k, active=int((ck > 0).sum()),
                             entropy_bits=H, max_share=float(ck.max() / max(tot, 1))))

    # per (slot, code, ach): P(a|c) = co / counts ; lift = P(a|c)/base_rate
    per_ach = []
    # also code purity: max_a P(a|c) per (slot, code)
    purity = np.zeros((K, C))
    for ai, name in enumerate(ach_names):
        if base_rate[ai] <= 0:
            continue
        best = None
        for k in range(K):
            with np.errstate(divide="ignore", invalid="ignore"):
                p_a_c = np.where(counts[k] >= 5, co[k, :, ai] / np.maximum(counts[k], 1), 0.0)
            lift = p_a_c / base_rate[ai]
            ci = int(np.argmax(lift))
            if best is None or lift[ci] > best["lift"]:
                best = dict(achievement=name, slot=k, code=ci,
                            n_co=int(co[k, ci, ai]), n_code=int(counts[k, ci]),
                            p_a_given_c=float(p_a_c[ci]), lift=float(lift[ci]))
        if best:
            per_ach.append(best)

    # purity (max P(a|c) per code, over achievements with support)
    for k in range(K):
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(counts[k][:, None] >= 5, co[k] / np.maximum(counts[k][:, None], 1), 0.0)  # (C, A)
        purity[k] = p.max(axis=1)
    active_mask = counts > 0
    w = counts[active_mask].astype(float)
    weighted_purity = float((purity[active_mask] * w).sum() / max(w.sum(), 1))

    best_p = np.array([b["p_a_given_c"] for b in per_ach])
    best_lift = np.array([b["lift"] for b in per_ach])
    mono = dict(
        n_achievements=len(per_ach),
        mean_best_p=float(best_p.mean()), median_best_p=float(np.median(best_p)),
        mean_best_lift=float(best_lift.mean()), median_best_lift=float(np.median(best_lift)),
        n_strong_detectors_p50=int((best_p >= 0.5).sum()),
        n_strong_detectors_p30=int((best_p >= 0.3).sum()),
        usage_weighted_code_purity=weighted_purity,
        total_active_codes=int((counts.sum(axis=0) > 0).sum()),
    )

    out = dict(kind=args.kind, K_slots=K, codebook=C, n_steps=int(N),
               per_slot=per_slot, per_achievement=sorted(per_ach, key=lambda b: -b["lift"]),
               monosemanticity=mono, achievement_base_rate={n: float(base_rate[i]) for i, n in enumerate(ach_names)})
    (args.out / f"mono_{args.kind}.json").write_text(json.dumps(out, indent=2))
    print(f"\n[{args.kind}] mean best P(a|c) = {mono['mean_best_p']:.3f}  "
          f"mean best lift = {mono['mean_best_lift']:.1f}  "
          f"strong detectors (P>=0.5) = {mono['n_strong_detectors_p50']}/{len(per_ach)}  "
          f"weighted purity = {weighted_purity:.3f}", flush=True)
    print(f"saved {args.out / ('mono_' + args.kind + '.json')}", flush=True)


if __name__ == "__main__":
    main()
