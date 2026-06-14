"""Train the ORIGINAL IRIS frame tokenizer on the Crafter replay buffer.

Uses the published IRIS tokenizer verbatim (eloialonso/iris): VQ over single
frames, 512-entry codebook, 16 tokens/frame (4x4 latent at 64px), L1 +
LPIPS-perceptual + commitment loss. This is the "literal IRIS" baseline; it
differs from Δ-IRIS on several axes at once (frame vs delta, 512 vs 1024 codes,
16 vs 4 tokens, no action conditioning), so it answers "is real IRIS less
event-monosemantic?" rather than isolating the delta trick (see the frame-only
ablation for that).

Episode files are read directly (the dataset is in Δ-IRIS layout) to avoid
importing Δ-IRIS modules alongside the IRIS src (name clashes).

Saves: <out>/iris_tokenizer.pt (state_dict + config) and a small log.
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
from einops import rearrange


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iris-src", required=True, type=Path, help="iris repo src/ dir")
    ap.add_argument("--dataset", required=True, type=Path, help="Crafter replay buffer train/ dir")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    sys.path.insert(0, str(args.iris_src.resolve()))
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)

    from models.tokenizer.nets import Encoder, Decoder, EncoderDecoderConfig
    from models.tokenizer.tokenizer import Tokenizer

    cfg = EncoderDecoderConfig(resolution=64, in_channels=3, z_channels=512, ch=64,
                               ch_mult=[1, 1, 1, 1, 1], num_res_blocks=2,
                               attn_resolutions=[8, 16], out_ch=3, dropout=0.0)
    tok = Tokenizer(vocab_size=512, embed_dim=512,
                    encoder=Encoder(cfg), decoder=Decoder(cfg), with_lpips=True).to(device).train()
    n_params = sum(p.numel() for p in tok.parameters() if p.requires_grad)
    print(f"IRIS tokenizer: {n_params} trainable params; vocab 512", flush=True)

    opt = torch.optim.Adam(tok.parameters(), lr=args.lr)

    # gather episode files once (Δ-IRIS hierarchical layout); exclude info.pt
    files = [p for p in args.dataset.rglob("*.pt") if p.name != "info.pt"]
    print(f"dataset: {len(files)} episode files", flush=True)
    rng = np.random.default_rng(args.seed)

    # Preload a representative frame pool onto the GPU (uint8) to remove the
    # per-step disk bottleneck (LPIPS + disk made the streaming version ~0.2 it/s).
    N_POOL = 300_000
    print(f"preloading ~{N_POOL} frames into a GPU pool...", flush=True)
    pool = []; perm = rng.permutation(len(files)); pi = 0
    while sum(p.shape[0] for p in pool) < N_POOL and pi < len(files):
        try:
            obs = torch.load(files[perm[pi]], map_location="cpu")["observations"].numpy()
            take = min(obs.shape[0], 60)
            sel = rng.choice(obs.shape[0], take, replace=False)
            pool.append(obs[sel])
        except Exception:
            pass
        pi += 1
    pool = np.concatenate(pool, axis=0)[:N_POOL]
    pool_t = torch.from_numpy(pool).to(device)        # (N,3,64,64) uint8
    n_frames = pool_t.shape[0]
    print(f"  pool: {n_frames} frames from {pi} episodes "
          f"({pool_t.element_size()*pool_t.nelement()/1e9:.1f} GB)", flush=True)

    def sample_batch(bs):
        idx = torch.randint(0, n_frames, (bs,), device=device)
        return pool_t[idx].float().div(255).unsqueeze(1)   # (bs,1,3,64,64)

    t0 = time.time(); logs = []
    for step in range(1, args.steps + 1):
        x = sample_batch(args.batch)
        losses = tok.compute_loss({"observations": x})
        loss = losses.loss_total
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == 1:
            il = losses.intermediate_losses
            msg = (f"step {step}/{args.steps}  total {loss.item():.4f}  " +
                   "  ".join(f"{k} {v:.4f}" for k, v in il.items()) +
                   f"  {step/max(time.time()-t0,1):.1f} it/s")
            print(" ", msg, flush=True)
            logs.append(dict(step=step, total=float(loss),
                             **{k: float(v) for k, v in il.items()}))

    torch.save({"state_dict": tok.state_dict(),
                "config": dict(vocab_size=512, embed_dim=512,
                               resolution=64, ch=64, ch_mult=[1, 1, 1, 1, 1],
                               z_channels=512, num_res_blocks=2, attn_resolutions=[8, 16]),
                "kind": "iris"},
               args.out / "iris_tokenizer.pt")
    (args.out / "iris_train_log.json").write_text(json.dumps(logs, indent=2))
    print(f"saved {args.out / 'iris_tokenizer.pt'} after {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
