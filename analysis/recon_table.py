"""Matched-protocol reconstruction-quality table for all THREE tokenizers.

The paper references a reconstruction-quality comparison between the Delta-IRIS
delta tokenizer and its two ablations but never tabulates it. This script
produces that table.

For each tokenizer we report two things:

  1. Train-log metrics: the recon / codebook numbers recorded during that
     tokenizer's own training (read straight out of the baseline train logs and,
     for Delta-IRIS, recomputed from a fresh batch since the trained run keeps no
     standalone tokenizer log here).

  2. Common-batch metrics (the matched-protocol part): every tokenizer is run on
     the SAME held-out batch of transitions drawn from $ANA/rollouts_probe_obs.npz
     (frames) + $ANA/rollouts_probe.npz (actions), and we recompute, on that one
     batch:
        - MSE and L1 (raw decoder output, and clamped to [0,1])
        - PSNR (from the clamped MSE)
        - LPIPS, if the IRIS VGG-LPIPS weights are available (optional/robust)
        - codebook usage (#distinct codes used) and perplexity over the batch
        - tokens/frame and codebook size

All three tokenizers reconstruct the SAME target frame x_{t+1}:
  - deltairis : reconstruct x2 from the transition (x1, a, x2)   [delta codes]
  - frameonly : reconstruct x2 from x2 alone                     [4 codes / 1024]
  - iris      : reconstruct x2 from x2 alone                     [16 codes / 512]
so MSE/L1/LPIPS are directly comparable (same target, same pixels), which is the
"matched protocol" the table needs.

Tokenizer loaders / encode-decode call conventions are copied verbatim from
analysis/tokenizer_mono_compare.py (and train_*_tokenizer.py) so this stays
self-contained and consistent with the rest of the repo.

Output: $ANA/recon_table/recon_table.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


# ============================================================================
# Module isolation. BOTH the Delta-IRIS run tree (run/src) and the IRIS repo
# (iris_src) ship a top-level `models` package AND a `models.tokenizer.tokenizer`
# module with DIFFERENT classes (delta: Tokenizer(config); iris:
# Tokenizer(vocab_size, embed_dim, encoder, decoder)). tokenizer_mono_compare.py
# never hits this because it runs one --kind per process. Here we load all three
# in ONE process, so once `models` is imported from one tree it is cached in
# sys.modules and the OTHER tree's import silently returns the wrong package.
# We defend against that by purging the cached local packages (and dropping the
# stale src dir from sys.path) before switching trees. Purging sys.modules after
# the needed classes/objects are already imported/constructed does NOT break
# those live objects (verified) -- it only forces the next `import` to re-resolve
# against the correct source tree.
# ============================================================================

# Top-level module names that collide between the delta run tree and iris_src.
_LOCAL_PKG_PREFIXES = ("models", "data", "dataset", "utils")


def purge_local_modules(drop_path=None):
    """Remove cached local packages so a subsequent import re-resolves against
    whatever is first on sys.path. Optionally drop a src dir from sys.path."""
    for name in list(sys.modules):
        head = name.split(".", 1)[0]
        if head in _LOCAL_PKG_PREFIXES:
            del sys.modules[name]
    if drop_path is not None:
        p = str(drop_path)
        while p in sys.path:
            sys.path.remove(p)


# ============================================================================
# Tokenizer loaders -- copied from analysis/tokenizer_mono_compare.py so this
# script is self-contained (matches repo style: each script has its own loaders).
# ============================================================================

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

        @torch.no_grad()
        def encode_decode(self, x):  # x (b,1,3,64,64) -> recon (b,3,64,64)
            z = self.encoder(x)
            z = rearrange(z, 'b t c (h k) (w l) -> b t (h w) (k l c)', h=grid, w=grid)
            qo = self.quantizer(z)
            q = rearrange(qo.q, 'b t (h w) (k l e) -> b t e (h k) (w l)',
                          h=grid, k=tr, l=tr)
            r = self.decoder(q)            # (b,1,3,64,64)
            return r.squeeze(1)            # (b,3,64,64)

    tok = FrameOnlyTokenizer().to(device).eval()
    tok.load_state_dict(blob["state_dict"], strict=True)
    return tok, dict(grid=grid, token_res=tr)


def build_iris(iris_src, ckpt_path, device, with_lpips=False):
    sys.path.insert(0, str(iris_src))
    from models.tokenizer.nets import Encoder, Decoder, EncoderDecoderConfig
    from models.tokenizer.tokenizer import Tokenizer
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = EncoderDecoderConfig(resolution=64, in_channels=3, z_channels=512, ch=64,
                               ch_mult=[1, 1, 1, 1, 1], num_res_blocks=2,
                               attn_resolutions=[8, 16], out_ch=3, dropout=0.0)
    tok = Tokenizer(vocab_size=512, embed_dim=512,
                    encoder=Encoder(cfg), decoder=Decoder(cfg), with_lpips=with_lpips).to(device).eval()
    sd = {k: v for k, v in blob["state_dict"].items() if not k.startswith("lpips.")}
    tok.load_state_dict(sd, strict=False)
    return tok


# ============================================================================
# Optional LPIPS (IRIS VGG-LPIPS). Robust: returns None if weights unavailable.
# ============================================================================

def try_build_lpips(iris_src, device):
    """Return an LPIPS callable f(a,b)->per-sample tensor, with both args in
    [-1,1] (channels-first), or (None, reason). Never raises."""
    try:
        sys.path.insert(0, str(iris_src))
        from models.tokenizer.lpips import LPIPS
        net = LPIPS().eval().to(device)          # downloads / loads cached VGG
        for p in net.parameters():
            p.requires_grad_(False)
        return net, None
    except Exception as e:  # network/weights missing, etc.
        return None, f"{type(e).__name__}: {e}"


# ============================================================================
# Metric helpers
# ============================================================================

def perplexity_and_usage(token_hist, codebook_size):
    """token_hist: 1-D int array of length codebook_size (counts over the batch,
    summed over all token slots). Returns (perplexity, n_codes_used,
    usage_fraction, entropy_bits)."""
    tot = token_hist.sum()
    if tot <= 0:
        return 0.0, 0, 0.0, 0.0
    p = token_hist.astype(np.float64) / float(tot)
    nz = p[p > 0]
    entropy_nats = float(-(nz * np.log(nz)).sum())
    entropy_bits = entropy_nats / np.log(2.0)
    perplexity = float(np.exp(entropy_nats))
    n_used = int((token_hist > 0).sum())
    return perplexity, n_used, n_used / float(codebook_size), entropy_bits


def img_metrics(recon, target):
    """recon, target: (B,3,64,64) float on device, target in [0,1]. Returns dict
    of raw + clamped MSE/L1, PSNR (from clamped MSE)."""
    diff = recon - target
    mse_raw = float(diff.pow(2).mean().item())
    l1_raw = float(diff.abs().mean().item())
    rc = recon.clamp(0.0, 1.0)
    diff_c = rc - target
    mse_clamp = float(diff_c.pow(2).mean().item())
    l1_clamp = float(diff_c.abs().mean().item())
    psnr = float(10.0 * np.log10(1.0 / max(mse_clamp, 1e-12)))
    return dict(mse=mse_raw, l1=l1_raw,
                mse_clamped=mse_clamp, l1_clamped=l1_clamp, psnr_clamped=psnr)


def train_log_summary(path):
    """Read a baseline train log (list of step dicts) and return the LAST entry
    plus a min-over-steps for the recon-like keys, or None if missing."""
    p = Path(path)
    if not p.exists():
        return dict(available=False, reason=f"not found: {p}")
    try:
        log = json.loads(p.read_text())
    except Exception as e:
        return dict(available=False, reason=f"parse error: {e}")
    if not isinstance(log, list) or not log:
        return dict(available=False, reason="empty or non-list log")
    last = log[-1]
    keys = [k for k in last.keys() if k != "step"]
    mins = {}
    for k in keys:
        vals = [e[k] for e in log if isinstance(e.get(k), (int, float))]
        if vals:
            mins[k] = float(min(vals))
    return dict(available=True, n_entries=len(log),
                final_step=int(last.get("step", -1)),
                final=last, min_over_steps=mins)


# ============================================================================
# Common held-out batch construction (shared across all three tokenizers)
# ============================================================================

def build_common_batch(obs_npz_path, rollouts_path, n_samples, seed, device):
    """Deterministically sample n_samples transitions (x1, a, x2) from the probe
    rollouts, drawn uniformly over all valid transition rows across episodes.
    Returns x1,x2 (N,3,64,64) float in [0,1] on device, a (N,) long on device,
    and meta about the draw."""
    obs_npz = np.load(obs_npz_path)
    all_obs, obs_starts = obs_npz["obs"], obs_npz["episode_starts"]   # (M,3,64,64) uint8 ; (n_ep+1,)
    r = np.load(rollouts_path, allow_pickle=True)
    actions = r["actions"].astype(np.int64)
    ep_lens = r["episode_lengths"].astype(np.int64)
    n_ep = ep_lens.shape[0]
    ep_start_row = np.concatenate(([0], np.cumsum(ep_lens)))  # row index into flat transition arrays

    assert obs_starts.shape[0] == n_ep + 1, \
        f"episode_starts {obs_starts.shape} vs n_ep {n_ep}"
    # Build a global list of (frame_idx_of_x1, action_row) for every transition.
    # Within episode ep: x1 = obs[obs_starts[ep] + t], x2 = next frame; a = actions[ep_start_row[ep]+t]
    x1_idx = np.empty(int(ep_lens.sum()), dtype=np.int64)
    act_row = np.empty(int(ep_lens.sum()), dtype=np.int64)
    w = 0
    for ep in range(n_ep):
        T = int(ep_lens[ep])
        # sanity: obs slice has T+1 frames
        assert obs_starts[ep + 1] - obs_starts[ep] == T + 1, \
            f"ep {ep}: obs frames {obs_starts[ep+1]-obs_starts[ep]} != T+1 {T+1}"
        base_obs = obs_starts[ep]
        base_row = ep_start_row[ep]
        x1_idx[w:w + T] = base_obs + np.arange(T)
        act_row[w:w + T] = base_row + np.arange(T)
        w += T
    assert w == x1_idx.shape[0]

    n_total = x1_idx.shape[0]
    n_take = min(n_samples, n_total)
    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(n_total, size=n_take, replace=False))

    f1 = x1_idx[sel]
    arows = act_row[sel]
    x1_np = all_obs[f1]            # (N,3,64,64) uint8
    x2_np = all_obs[f1 + 1]        # next frame (same episode by construction)
    a_np = actions[arows]          # (N,)

    x1 = torch.from_numpy(x1_np).to(device).float().div(255)
    x2 = torch.from_numpy(x2_np).to(device).float().div(255)
    a = torch.from_numpy(a_np).to(device).long()
    assert x1.shape == x2.shape and x1.shape[1:] == (3, 64, 64)
    assert a.shape[0] == x1.shape[0]
    meta = dict(n_total_transitions=int(n_total), n_sampled=int(n_take), seed=int(seed))
    return x1, x2, a, meta


# ============================================================================
# Per-tokenizer common-batch evaluation
# ============================================================================

@torch.no_grad()
def eval_deltairis(tok, x1, x2, a, batch, device, lpips_net):
    """Delta-IRIS: reconstruct x2 from (x1,a,x2); 4 tokens, codebook 1024."""
    C, K = 1024, 4
    hist = np.zeros(C, dtype=np.int64)
    agg = _MetricAccumulator(lpips_net is not None)
    N = x1.shape[0]
    for s in range(0, N, batch):
        e = min(s + batch, N)
        xb1 = x1[s:e].unsqueeze(1)   # (b,1,3,64,64)
        xb2 = x2[s:e].unsqueeze(1)
        ab = a[s:e].unsqueeze(1)     # (b,1)
        # tokens (for codebook usage)
        z = tok.encode(xb1, ab, xb2)
        z = rearrange(z, 'b t c (h k) (w l) -> b t (h w) (k l c)',
                      h=tok.tokens_grid_res, w=tok.tokens_grid_res)
        qo = tok.quantizer(z)
        toks = qo.tokens.reshape(-1).cpu().numpy()
        np.add.at(hist, toks, 1)
        # reconstruction of x2
        recon = tok.encode_decode(xb1, ab, xb2)   # (b,1,3,64,64), clamped to [0,1]
        recon = recon.squeeze(1)                   # (b,3,64,64)
        agg.add(recon, xb2.squeeze(1), lpips_net)
    perp, n_used, frac, ent = perplexity_and_usage(hist, C)
    out = agg.finalize()
    out.update(codebook_size=C, tokens_per_frame=K,
               codebook_perplexity=perp, codes_used=n_used,
               codebook_usage_fraction=frac, codebook_entropy_bits=ent)
    return out


@torch.no_grad()
def eval_frameonly(tok, helper, x1, x2, a, batch, device, lpips_net):
    """Frame-only: reconstruct x2 from x2; 4 tokens, codebook 1024."""
    C, K = 1024, 4
    hist = np.zeros(C, dtype=np.int64)
    agg = _MetricAccumulator(lpips_net is not None)
    N = x2.shape[0]
    for s in range(0, N, batch):
        e = min(s + batch, N)
        xb2 = x2[s:e].unsqueeze(1)   # (b,1,3,64,64)
        toks = tok.encode_tokens(xb2).reshape(-1)   # (b*4,)
        np.add.at(hist, toks, 1)
        recon = tok.encode_decode(xb2)               # (b,3,64,64) raw decoder output
        agg.add(recon, xb2.squeeze(1), lpips_net)
    perp, n_used, frac, ent = perplexity_and_usage(hist, C)
    out = agg.finalize()
    out.update(codebook_size=C, tokens_per_frame=K,
               codebook_perplexity=perp, codes_used=n_used,
               codebook_usage_fraction=frac, codebook_entropy_bits=ent)
    return out


@torch.no_grad()
def eval_iris(tok, x1, x2, a, batch, device, lpips_net):
    """IRIS: reconstruct x2 from x2; 16 tokens, codebook 512."""
    C, K = 512, 16
    hist = np.zeros(C, dtype=np.int64)
    agg = _MetricAccumulator(lpips_net is not None)
    N = x2.shape[0]
    for s in range(0, N, batch):
        e = min(s + batch, N)
        xb2 = x2[s:e]                # (b,3,64,64) in [0,1]
        enc = tok.encode(xb2, should_preprocess=True)
        toks = enc.tokens.reshape(-1).cpu().numpy()   # (b*16,)
        np.add.at(hist, toks, 1)
        # encode_decode: preprocess [0,1]->[-1,1] in, postprocess back to [0,1]
        recon = tok.encode_decode(xb2, should_preprocess=True, should_postprocess=True)  # (b,3,64,64)
        agg.add(recon, xb2, lpips_net)
    perp, n_used, frac, ent = perplexity_and_usage(hist, C)
    out = agg.finalize()
    out.update(codebook_size=C, tokens_per_frame=K,
               codebook_perplexity=perp, codes_used=n_used,
               codebook_usage_fraction=frac, codebook_entropy_bits=ent)
    return out


class _MetricAccumulator:
    """Accumulate sum-of-squared-error etc. over batches so the reported MSE is
    an exact per-pixel mean over the WHOLE common batch (not a mean-of-means)."""
    def __init__(self, want_lpips):
        self.n_px = 0
        self.n_img = 0
        self.sse_raw = 0.0
        self.sae_raw = 0.0
        self.sse_clamp = 0.0
        self.sae_clamp = 0.0
        self.want_lpips = want_lpips
        self.lpips_sum = 0.0
        self.lpips_n = 0

    def add(self, recon, target, lpips_net):
        # recon, target: (b,3,64,64) on device, target in [0,1]
        b = recon.shape[0]
        npx = recon.numel()
        diff = recon - target
        self.sse_raw += float(diff.pow(2).sum().item())
        self.sae_raw += float(diff.abs().sum().item())
        rc = recon.clamp(0.0, 1.0)
        diff_c = rc - target
        self.sse_clamp += float(diff_c.pow(2).sum().item())
        self.sae_clamp += float(diff_c.abs().sum().item())
        self.n_px += npx
        self.n_img += b
        if self.want_lpips and lpips_net is not None:
            # LPIPS expects inputs in [-1,1], channels-first.
            a_in = rc.mul(2).sub(1)
            b_in = target.mul(2).sub(1)
            d = lpips_net(a_in, b_in)             # (b,1,1,1) typically
            self.lpips_sum += float(d.sum().item())
            self.lpips_n += b

    def finalize(self):
        npx = max(self.n_px, 1)
        mse = self.sse_raw / npx
        l1 = self.sae_raw / npx
        mse_c = self.sse_clamp / npx
        l1_c = self.sae_clamp / npx
        psnr = float(10.0 * np.log10(1.0 / max(mse_c, 1e-12)))
        out = dict(n_frames=int(self.n_img),
                   mse=mse, l1=l1,
                   mse_clamped=mse_c, l1_clamped=l1_c, psnr_clamped=psnr)
        if self.want_lpips and self.lpips_n > 0:
            out["lpips"] = self.lpips_sum / self.lpips_n
        else:
            out["lpips"] = None
        return out


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltairis-src", required=True, type=Path,
                    help="finished run hydra dir (has src/, .hydra/config.yaml, checkpoints/last.pt)")
    ap.add_argument("--iris-src", required=True, type=Path, help="iris repo src/ dir")
    ap.add_argument("--frameonly-ckpt", required=True, type=Path)
    ap.add_argument("--iris-ckpt", required=True, type=Path)
    ap.add_argument("--obs", required=True, type=Path, help="rollouts_probe_obs.npz")
    ap.add_argument("--rollouts", required=True, type=Path, help="rollouts_probe.npz (for actions)")
    ap.add_argument("--frameonly-train-log", required=True, type=Path)
    ap.add_argument("--iris-train-log", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-samples", type=int, default=8192,
                    help="size of the common held-out batch of transitions")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-lpips", action="store_true", help="skip LPIPS even if available")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    run = args.deltairis_src.resolve()
    iris_src = args.iris_src.resolve()
    t_start = time.time()

    print("=== building common held-out batch of transitions ===", flush=True)
    x1, x2, a, batch_meta = build_common_batch(args.obs, args.rollouts,
                                               args.n_samples, args.seed, device)
    print(f"  sampled {batch_meta['n_sampled']} / {batch_meta['n_total_transitions']} "
          f"transitions (seed {args.seed})", flush=True)
    print(f"  x1 {tuple(x1.shape)}  x2 {tuple(x2.shape)}  a {tuple(a.shape)}", flush=True)

    # ----- optional LPIPS (built once, shared across tokenizers) -------------
    # LPIPS lives in the IRIS tree (models.tokenizer.lpips). We build it FIRST,
    # while sys.modules is still clean, so its `from models...` import resolves
    # against iris_src -- THEN we purge the cached `models`/`utils`/... packages
    # and drop iris_src from sys.path so the Delta-IRIS load below re-resolves
    # `models` against run/src. The already-constructed lpips_net survives the
    # purge unharmed (it only references already-imported classes).
    if args.no_lpips:
        lpips_net, lpips_reason = None, "disabled via --no-lpips"
    else:
        lpips_net, lpips_reason = try_build_lpips(iris_src, device)
    if lpips_net is not None:
        print("  LPIPS: available (IRIS VGG-LPIPS)", flush=True)
    else:
        print(f"  LPIPS: NOT available -> reporting null ({lpips_reason})", flush=True)
    # Reset module/import state regardless of LPIPS outcome so the delta tree
    # loads cleanly (try_build_lpips may have partially imported iris `models`).
    purge_local_modules(drop_path=iris_src)

    # ----- train-log summaries ----------------------------------------------
    print("=== reading baseline train logs ===", flush=True)
    fo_log = train_log_summary(args.frameonly_train_log)
    iris_log = train_log_summary(args.iris_train_log)
    print(f"  frameonly log: {fo_log.get('available')}  iris log: {iris_log.get('available')}", flush=True)

    results = {}

    # ---------------- Delta-IRIS --------------------------------------------
    # delta + frameonly BOTH live in run/src and share the same `models` package,
    # so they load back-to-back without a purge in between.
    print("=== deltairis: load + eval ===", flush=True)
    tok_d = load_deltairis(run, device)
    cb = eval_deltairis(tok_d, x1, x2, a, args.batch, device, lpips_net)
    results["deltairis"] = dict(
        description="Delta-IRIS delta tokenizer: reconstructs x_{t+1} from the "
                    "transition (x_t, a_t, x_{t+1}); 4 codes, codebook 1024.",
        train_log=dict(available=False,
                       reason="trained run keeps no standalone tokenizer log here; "
                              "see common_batch metrics for matched recon"),
        common_batch=cb)
    print(f"  [deltairis] mse={cb['mse']:.5f} mse_clamped={cb['mse_clamped']:.5f} "
          f"psnr={cb['psnr_clamped']:.2f} perp={cb['codebook_perplexity']:.1f} "
          f"codes_used={cb['codes_used']}/{cb['codebook_size']} lpips={cb['lpips']}", flush=True)
    del tok_d
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---------------- Frame-only --------------------------------------------
    print("=== frameonly: load + eval ===", flush=True)
    tok_f, fo_helper = build_frameonly(run, args.frameonly_ckpt, device)
    cb = eval_frameonly(tok_f, fo_helper, x1, x2, a, args.batch, device, lpips_net)
    results["frameonly"] = dict(
        description="Frame-only ablation: reconstructs x_{t+1} from x_{t+1} alone; "
                    "4 codes, codebook 1024 (same backbone as Delta-IRIS, no delta/action).",
        train_log=fo_log,
        common_batch=cb)
    print(f"  [frameonly] mse={cb['mse']:.5f} mse_clamped={cb['mse_clamped']:.5f} "
          f"psnr={cb['psnr_clamped']:.2f} perp={cb['codebook_perplexity']:.1f} "
          f"codes_used={cb['codes_used']}/{cb['codebook_size']} lpips={cb['lpips']}", flush=True)
    del tok_f
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---------------- IRIS ---------------------------------------------------
    # Switch trees: drop run/src `models`/`utils`/... from the import cache and
    # from sys.path so `from models.tokenizer.tokenizer import Tokenizer` inside
    # build_iris re-resolves against iris_src (NOT the cached delta package).
    purge_local_modules(drop_path=run / "src")
    print("=== iris: load + eval ===", flush=True)
    tok_i = build_iris(iris_src, args.iris_ckpt, device, with_lpips=False)
    cb = eval_iris(tok_i, x1, x2, a, args.batch, device, lpips_net)
    results["iris"] = dict(
        description="Original IRIS frame tokenizer: reconstructs x_{t+1} from "
                    "x_{t+1} alone; 16 codes, codebook 512.",
        train_log=iris_log,
        common_batch=cb)
    print(f"  [iris] mse={cb['mse']:.5f} mse_clamped={cb['mse_clamped']:.5f} "
          f"psnr={cb['psnr_clamped']:.2f} perp={cb['codebook_perplexity']:.1f} "
          f"codes_used={cb['codes_used']}/{cb['codebook_size']} lpips={cb['lpips']}", flush=True)
    del tok_i
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---------------- assemble + save ---------------------------------------
    out_obj = dict(
        meta=dict(
            run=str(run),
            iris_src=str(iris_src),
            frameonly_ckpt=str(args.frameonly_ckpt),
            iris_ckpt=str(args.iris_ckpt),
            obs=str(args.obs),
            rollouts=str(args.rollouts),
            common_batch=batch_meta,
            lpips_available=bool(lpips_net is not None),
            lpips_reason=lpips_reason,
            note=("All three tokenizers reconstruct the SAME target frame x_{t+1} "
                  "on the SAME held-out transition batch (matched protocol). "
                  "Common-batch MSE/L1 reported both raw and clamped-to-[0,1]; "
                  "PSNR computed from the clamped MSE. Codebook perplexity is "
                  "exp(entropy) of the token histogram over all slots on the batch."),
            elapsed_min=round((time.time() - t_start) / 60.0, 2),
        ),
        tokenizers=results,
    )

    # compact comparison table (the headline numbers the paper needs)
    table = []
    for k in ("deltairis", "frameonly", "iris"):
        cb = results[k]["common_batch"]
        table.append(dict(
            tokenizer=k,
            codebook_size=cb["codebook_size"],
            tokens_per_frame=cb["tokens_per_frame"],
            common_mse=round(cb["mse"], 6),
            common_mse_clamped=round(cb["mse_clamped"], 6),
            common_l1_clamped=round(cb["l1_clamped"], 6),
            common_psnr=round(cb["psnr_clamped"], 3),
            common_lpips=(round(cb["lpips"], 5) if cb["lpips"] is not None else None),
            codebook_perplexity=round(cb["codebook_perplexity"], 2),
            codes_used=cb["codes_used"],
            codebook_usage_fraction=round(cb["codebook_usage_fraction"], 4),
        ))
    out_obj["table"] = table

    out_path = args.out / "recon_table.json"
    out_path.write_text(json.dumps(out_obj, indent=2))
    print("\n=== recon_table (common held-out batch) ===", flush=True)
    hdr = f"{'tokenizer':<11}{'cb':>6}{'tok/f':>6}{'mse':>10}{'mse_clp':>10}{'psnr':>8}{'lpips':>9}{'perp':>9}{'used':>10}"
    print(hdr, flush=True)
    for row in table:
        lp = f"{row['common_lpips']:.4f}" if row["common_lpips"] is not None else "n/a"
        print(f"{row['tokenizer']:<11}{row['codebook_size']:>6}{row['tokens_per_frame']:>6}"
              f"{row['common_mse']:>10.5f}{row['common_mse_clamped']:>10.5f}"
              f"{row['common_psnr']:>8.2f}{lp:>9}"
              f"{row['codebook_perplexity']:>9.1f}"
              f"{str(row['codes_used'])+'/'+str(row['codebook_size']):>10}", flush=True)
    print(f"\nsaved {out_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
