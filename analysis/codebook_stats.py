"""Codebook usage + per-code top-K transition retrieval for a trained Δ-IRIS tokenizer.

Iterates every transition in the training replay buffer of a finished Δ-IRIS run,
re-encodes it with the trained tokenizer, and records:

  * usage counts per (slot, code) — codebook utilisation, dead/hot codes
  * codebook embedding matrix (1024, 64) — for clustering downstream
  * top-K transitions per (slot, code) by cosine similarity with the chosen code
    — saved as (episode_id, step_idx, similarity) references, no frames here

Resulting files (under --out):
  - codebook_stats.npz   : counts, codewords_freqs, codebook embeddings
  - top_samples.npz      : per (slot, code) top-K (sim, ep_id, step_idx)
  - meta.json            : run paths, dataset shape, args echo

A second pass (render_gallery.py) reads top_samples.npz and the on-disk dataset
to produce an HTML grid — that part is CPU-only.

Usage (single H200):
  python codebook_stats.py \
    --run /mmfs1/scratch/hpc/11/xiar3/expwm-runs/delta-iris-full-21531393/hydra \
    --out /mmfs1/scratch/hpc/11/xiar3/expwm-runs/analysis-21531393 \
    --max-episodes -1
"""
from __future__ import annotations

import argparse
import heapq
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Make the trained run's saved src/ importable so `instantiate(cfg.tokenizer)`
# can resolve `models.tokenizer.TokenizerConfig` etc.


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path,
                    help="Δ-IRIS hydra.run.dir of a finished run (must contain "
                         "checkpoints/, .hydra/config.yaml, src/)")
    ap.add_argument("--out", required=True, type=Path,
                    help="output directory for stats artefacts")
    ap.add_argument("--max-episodes", type=int, default=-1,
                    help="-1 = all; otherwise process the first N episodes")
    ap.add_argument("--batch-size", type=int, default=256,
                    help="transitions per forward pass (default 256)")
    ap.add_argument("--top-k", type=int, default=32,
                    help="keep top-K best-matching transitions per (slot, code)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    run = args.run.resolve()
    assert (run / "checkpoints" / "last.pt").exists(), f"no last.pt under {run}"
    assert (run / ".hydra" / "config.yaml").exists(), f"no .hydra/config.yaml under {run}"
    assert (run / "src").exists(), f"no src/ under {run} (was it the resume run dir?)"

    sys.path.insert(0, str(run / "src"))
    args.out.mkdir(parents=True, exist_ok=True)

    # Δ-IRIS's trainer registers an `eval` resolver for ${eval:'...'} interpolations.
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    cfg = OmegaConf.load(run / ".hydra" / "config.yaml")
    # Crafter has 17 actions; tokenizer.num_actions is set lazily by the trainer at runtime,
    # so we plug it in before resolving any interpolations that reference it.
    if cfg.params.tokenizer.num_actions is None:
        cfg.params.tokenizer.num_actions = 17
    OmegaConf.resolve(cfg)

    device = torch.device(args.device)
    # Δ-IRIS instantiates the TokenizerConfig first, then wraps it in Tokenizer.
    from models.tokenizer import Tokenizer  # imported from saved run's src/
    tokenizer = Tokenizer(instantiate(cfg.params.tokenizer)).to(device).eval()

    # Pull tokenizer weights out of the agent state-dict.
    ckpt = torch.load(run / "checkpoints" / "last.pt", map_location=device,
                      weights_only=False)
    tok_state = {k[len("tokenizer."):]: v for k, v in ckpt.items()
                 if k.startswith("tokenizer.")}
    missing, unexpected = tokenizer.load_state_dict(tok_state, strict=False)
    print(f"loaded tokenizer state. missing={len(missing)} unexpected={len(unexpected)}")

    K_slots = cfg.params.tokenizer.num_tokens          # 4
    C = cfg.params.tokenizer.codebook_size              # 1024
    embed_dim = cfg.params.tokenizer.codebook_dim       # 64
    print(f"tokenizer: {K_slots} slots × codebook of {C}, embed_dim={embed_dim}")

    dataset_dir = run / "checkpoints" / "dataset" / "train"
    from data.dataset import EpisodeDataset  # uses the same hierarchical layout as training
    dataset = EpisodeDataset(dataset_dir, name="train")
    num_episodes = int(dataset.num_episodes)
    total_steps = int(dataset.num_steps)
    print(f"dataset: {num_episodes} episodes, {total_steps} steps")

    n_eps = num_episodes if args.max_episodes < 0 else min(num_episodes, args.max_episodes)
    print(f"will scan {n_eps} episodes (~{n_eps/num_episodes*100:.1f}% of dataset)")

    counts = np.zeros((K_slots, C), dtype=np.int64)
    # top-K heap per (slot, code). Each entry: (sim_float, ep_id, step_idx).
    # heapq is a min-heap, so we keep the K largest by popping the smallest.
    heaps: list[list[list]] = [[[] for _ in range(C)] for _ in range(K_slots)]

    # Cache the pre_quant_proj + codebook into a single tensor for speed.
    codebook = tokenizer.quantizer.codebook  # (C, codebook_dim) on device

    transitions_done = 0
    t0 = time.time()
    last_print = t0

    for ep_id in range(n_eps):
        try:
            ep = dataset.load_episode(ep_id)             # Episode dataclass
        except FileNotFoundError:
            continue
        obs = ep.observations                            # (T, 3, 64, 64) uint8
        act = ep.actions                                 # (T,) int64
        T = obs.shape[0]
        if T < 2:
            continue
        x1 = obs[:-1].to(device, non_blocking=True).float().div_(255)
        x2 = obs[1:].to(device, non_blocking=True).float().div_(255)
        a = act[:-1].to(device, non_blocking=True)
        n_trans = T - 1

        for s in range(0, n_trans, args.batch_size):
            e = min(s + args.batch_size, n_trans)
            b_x1 = x1[s:e].unsqueeze(0)              # (1, B, 3, 64, 64)
            b_x2 = x2[s:e].unsqueeze(0)
            b_a = a[s:e].unsqueeze(0)                # (1, B)

            with torch.no_grad():
                z = tokenizer.encode(b_x1, b_a, b_x2)  # (1, B, C', H, W)
                z = rearrange(
                    z, "b t c (h k) (w l) -> b t (h w) (k l c)",
                    h=tokenizer.tokens_grid_res, w=tokenizer.tokens_grid_res
                )
                z = tokenizer.quantizer.pre_quant_proj(z)
                z = F.normalize(z, dim=-1)               # (1, B, K_slots, embed_dim)
                # cosine similarity to every code
                cos = torch.einsum("btke,ce->btkc", z, codebook)  # (1, B, K, C)
                sims, tokens = cos.max(dim=-1)                     # (1, B, K)

            sims_np = sims.squeeze(0).cpu().numpy()        # (B, K)
            tokens_np = tokens.squeeze(0).cpu().numpy()    # (B, K)
            B = tokens_np.shape[0]

            # increment counts
            for k_slot in range(K_slots):
                np.add.at(counts[k_slot], tokens_np[:, k_slot], 1)

            # update heaps. We pre-filter: for each (slot, code) only the
            # batch-max similarity is worth a heap-push.
            for k_slot in range(K_slots):
                ts = tokens_np[:, k_slot]
                ss = sims_np[:, k_slot]
                # group by code via np.unique → for each (k_slot, code) keep
                # only its best entry in this batch
                uniq_codes, inv = np.unique(ts, return_inverse=True)
                # per-group argmax of similarity
                best_idx = np.full(uniq_codes.shape, -1, dtype=np.int64)
                best_sim = np.full(uniq_codes.shape, -np.inf, dtype=np.float32)
                for i, g in enumerate(inv):
                    if ss[i] > best_sim[g]:
                        best_sim[g] = ss[i]; best_idx[g] = i
                for g, code in enumerate(uniq_codes):
                    sim = float(best_sim[g])
                    step = int(s + int(best_idx[g]))
                    h = heaps[k_slot][int(code)]
                    if len(h) < args.top_k:
                        heapq.heappush(h, [sim, ep_id, step])
                    elif sim > h[0][0]:
                        heapq.heapreplace(h, [sim, ep_id, step])

            transitions_done += B

        now = time.time()
        if now - last_print > 10 or ep_id == n_eps - 1:
            rate = transitions_done / max(now - t0, 1e-3)
            eta = (total_steps - transitions_done) / max(rate, 1.0) if args.max_episodes < 0 else 0
            n_active = int((counts.sum(axis=0) > 0).sum())
            print(f"  ep {ep_id+1}/{n_eps}  trans {transitions_done}/{total_steps}  "
                  f"{rate:.0f} t/s  active codes {n_active}/{C}  "
                  f"eta {eta/60:.1f} min", flush=True)
            last_print = now

    print(f"\ntotal: {transitions_done} transitions in {(time.time()-t0)/60:.1f} min")

    # --- save artefacts ----------------------------------------------------
    print("\nsaving stats...")
    cb_freqs = tokenizer.quantizer.codewords_freqs.detach().cpu().numpy()
    cb_embed = codebook.detach().cpu().numpy()
    np.savez(args.out / "codebook_stats.npz",
             counts=counts,
             codewords_freqs=cb_freqs,
             codebook_embed=cb_embed)
    print(f"  {args.out / 'codebook_stats.npz'}")

    # Encode heaps as fixed-shape arrays for portability.
    sims_arr = np.full((K_slots, C, args.top_k), np.nan, dtype=np.float32)
    eps_arr = np.full((K_slots, C, args.top_k), -1, dtype=np.int32)
    steps_arr = np.full((K_slots, C, args.top_k), -1, dtype=np.int32)
    for k_slot in range(K_slots):
        for c in range(C):
            h = sorted(heaps[k_slot][c], key=lambda r: -r[0])  # high sim first
            for i, (sim, ep_id, step) in enumerate(h):
                sims_arr[k_slot, c, i] = sim
                eps_arr[k_slot, c, i] = ep_id
                steps_arr[k_slot, c, i] = step
    np.savez(args.out / "top_samples.npz",
             sims=sims_arr, eps=eps_arr, steps=steps_arr)
    print(f"  {args.out / 'top_samples.npz'}")

    meta = dict(
        run=str(run),
        out=str(args.out),
        max_episodes=args.max_episodes,
        episodes_scanned=n_eps,
        transitions_scanned=int(transitions_done),
        batch_size=args.batch_size,
        top_k=args.top_k,
        K_slots=K_slots,
        C=C,
        embed_dim=embed_dim,
        n_active_codes_total=int((counts.sum(axis=0) > 0).sum()),
        n_active_codes_per_slot=[int((counts[k] > 0).sum()) for k in range(K_slots)],
        seconds_elapsed=time.time() - t0,
    )
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  {args.out / 'meta.json'}")

    print("\ndone.")


if __name__ == "__main__":
    main()
