"""Train a FRAME-ONLY ablation of the Δ-IRIS tokenizer on the Crafter replay buffer.

This is the clean ablation that isolates the "delta trick": it reuses the exact
Δ-IRIS tokenizer backbone (FrameEncoder / Quantizer / FrameDecoder, codebook
1024, 4 tokens on a 2x2 grid) but removes the delta + action conditioning ---
the encoder sees only x_{t+1} (3 channels, not the stacked x_t|a|x_{t+1} = 7),
and the decoder reconstructs x_{t+1} from its 4 codes alone (no frame_cnn(x_t),
no action embedding). Everything else (data, codebook size, token count, conv
stack, optimiser) is held identical so the only changed variable is whether the
codes describe the transition (Δ-IRIS) or the absolute frame (this ablation).

Saves: <out>/frameonly_tokenizer.pt  (state_dict + config) and a small log.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltairis-src", required=True, type=Path,
                    help="path to a finished run's hydra dir (for src/ + config + dataset)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    run = args.deltairis_src.resolve()
    sys.path.insert(0, str(run / "src"))
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)

    from models.convnet import FrameEncoder, FrameDecoder, FrameCnnConfig
    from models.tokenizer.quantizer import Quantizer
    from data.dataset import EpisodeDataset

    # ----- frame-only tokenizer (mirrors Δ-IRIS shapes exactly) ---------------
    CODEBOOK, CODE_DIM, LATENT, NUMCH = 1024, 64, 64, 64
    MULT, DOWN = [1, 1, 2, 2, 4], [1, 0, 1, 1, 0]   # identical to Δ-IRIS
    IMG, NUM_TOKENS = 64, 4
    latent_res = IMG // (2 ** sum(DOWN))             # 8
    grid = int(NUM_TOKENS ** 0.5)                    # 2
    token_res = latent_res // grid                   # 4
    input_dim = LATENT * token_res * token_res       # 1024

    class FrameOnlyTokenizer(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = FrameEncoder(FrameCnnConfig(3, LATENT, NUMCH, MULT, DOWN))
            # decoder input = quantized codes reshaped to (b,t,code_dim,latent_res,latent_res)
            self.decoder = FrameDecoder(FrameCnnConfig(3, CODE_DIM, NUMCH, MULT, DOWN))
            self.quantizer = Quantizer(CODEBOOK, CODE_DIM, input_dim=input_dim,
                                       max_codebook_updates_with_revival=0)

        def encode(self, x):  # x: (b,1,3,64,64) in [0,1]
            z = self.encoder(x)                                           # (b,1,64,8,8)
            z = rearrange(z, 'b t c (h k) (w l) -> b t (h w) (k l c)', h=grid, w=grid)
            return self.quantizer(z)

        def forward(self, x):
            qo = self.encode(x)
            q = rearrange(qo.q, 'b t (h w) (k l e) -> b t e (h k) (w l)',
                          h=grid, k=token_res, l=token_res)
            r = self.decoder(q)
            return qo, r

    tok = FrameOnlyTokenizer().to(device).train()
    n_params = sum(p.numel() for p in tok.parameters())
    print(f"frame-only tokenizer: {n_params} params; latent_res={latent_res}, "
          f"{NUM_TOKENS} tokens, codebook {CODEBOOK}", flush=True)

    opt = torch.optim.Adam(tok.parameters(), lr=args.lr, weight_decay=0.01)

    ds = EpisodeDataset(run / "checkpoints" / "dataset" / "train", name="train")
    n_ep = int(ds.num_episodes)
    print(f"dataset: {n_ep} episodes, {int(ds.num_steps)} steps", flush=True)
    rng = np.random.default_rng(args.seed)

    # Preload a representative frame pool onto the GPU (uint8) to remove the
    # per-step disk bottleneck; sample minibatches from it.
    N_POOL = 300_000
    print(f"preloading ~{N_POOL} frames into a GPU pool...", flush=True)
    pool = []
    eps = rng.permutation(n_ep)
    pi = 0
    while sum(p.shape[0] for p in pool) < N_POOL and pi < n_ep:
        try:
            e = ds.load_episode(int(eps[pi]))
            obs = e.observations.numpy()
            take = min(obs.shape[0], 60)
            sel = rng.choice(obs.shape[0], take, replace=False)
            pool.append(obs[sel])
        except FileNotFoundError:
            pass
        pi += 1
    pool = np.concatenate(pool, axis=0)[:N_POOL]
    pool_t = torch.from_numpy(pool).to(device)        # (N,3,64,64) uint8
    n_frames = pool_t.shape[0]
    print(f"  pool: {n_frames} frames from {pi} episodes ({pool_t.element_size()*pool_t.nelement()/1e9:.1f} GB)", flush=True)

    def sample_batch(bs):
        idx = torch.randint(0, n_frames, (bs,), device=device)
        return pool_t[idx].float().div(255).unsqueeze(1)   # (bs,1,3,64,64)

    t0 = time.time(); logs = []
    for step in range(1, args.steps + 1):
        x = sample_batch(args.batch)
        qo, r = tok(x)
        recon_l1 = (x - r).abs().mean()
        recon_l2 = (x - r).pow(2).mean()
        commit = qo.loss["commitment_loss"]
        loss = 0.1 * recon_l1 + recon_l2 + commit
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(tok.parameters(), 10.0)
        opt.step()
        if step % 500 == 0 or step == 1:
            ent = tok.quantizer.compute_codebook_entropy()
            msg = (f"step {step}/{args.steps}  l1 {recon_l1.item():.4f}  l2 {recon_l2.item():.5f}  "
                   f"commit {commit.item():.5f}  cb_entropy {ent:.2f}/10  "
                   f"{step/max(time.time()-t0,1):.1f} it/s")
            print(" ", msg, flush=True)
            logs.append(dict(step=step, l1=float(recon_l1), l2=float(recon_l2),
                             commit=float(commit), cb_entropy=float(ent)))

    torch.save({"state_dict": tok.state_dict(),
                "config": dict(codebook=CODEBOOK, code_dim=CODE_DIM, latent=LATENT,
                               num_ch=NUMCH, mult=MULT, down=DOWN, num_tokens=NUM_TOKENS,
                               token_res=token_res, grid=grid, input_dim=input_dim),
                "kind": "frameonly"},
               args.out / "frameonly_tokenizer.pt")
    (args.out / "frameonly_train_log.json").write_text(json.dumps(logs, indent=2))
    print(f"saved {args.out / 'frameonly_tokenizer.pt'} after {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
