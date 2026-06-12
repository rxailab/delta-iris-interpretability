# Δ-IRIS Interpretability

Explainability experiments on the quantized (delta-token) representations of
[Δ-IRIS](https://github.com/vmicheli/delta-iris) ("Efficient World Models with
Context-Aware Tokenization", ICML 2024), using a full 1000-epoch Crafter
training run (single seed, single H200, final mean return ≈ 16.2 / 22
achievements per episode).

The pipeline goes from *descriptive* → *informational* → *interventional*
evidence, each tier cross-validating the previous one:

| stage | script | finding |
|---|---|---|
| Codebook usage | `analysis/codebook_stats.py` | all 1024 codes alive, Zipf-concentrated (~50 effective codes/slot) |
| Visual gallery | `analysis/render_gallery.py` | per-(slot, code) top-K transition tiles; codes look like *events* |
| Code ↔ achievement MI | `analysis/rollout_with_info.py` + `analysis/code_achievement_mi.py` | codes are near-monosemantic achievement detectors (lift up to 707×, e.g. (s3,c75) ⇒ collect_iron at P=0.88) |
| Layer-wise probes | `analysis/probe_layers.py` + `analysis/render_probes.py` | events decodable from raw codes (AUROC ≥ .97); cumulative state decodable from the **frame CNN alone** (Crafter renders inventory in the HUD — see below); only *anticipation* (reward-in-next-5) improves with transformer depth |
| CAVs + traces | `analysis/cav_traces.py` | 35 concept directions with held-out AUROC ≈ 1.0; CAV geometry mirrors the Crafter tech-tree (cos(cum[collect_stone], cum[place_stone]) = 0.95) |
| Sparse autoencoder | `analysis/train_sae.py` | TopK SAE (2048 dict, k=16, FVU 0.043) splits the polysemantic code (s3,c36) into separate place_table / make_wood_pickaxe / make_wood_sword features |
| One-step ablation | `analysis/ablate_features.py` | ablating the matched SAE direction at the unlock step costs up to 4.9 nats, selectively (random direction ≈ 0); fully recovered if ablated 3 steps earlier |
| Imagination ablation | `analysis/imagine_ablation.py` | removing one direction erases collect_wood from the model's imagined future (P 1.00 → 0.00 first-step), CAV + random directions inert → read ≠ write directions |
| Causal tracing | `analysis/causal_trace.py` | activation patching: all restoration mass sits at the final step's action+latent tokens — unlock-code prediction is quasi-Markovian |

**The HUD confound** (`probes_hud` results): Crafter draws the agent's
inventory at the bottom of every frame, so "memory-like" probes
(`ach_cum[*]`) are satisfiable from the current frame alone — `frame_emb`
matches or beats every transformer block on 15/18 concepts. The only probe
that genuinely needs the transformer is anticipation. Three independent
experiments (frame probes, ablation-recovery at gap 3, causal tracing)
converge on this.

## Layout

```
analysis/        all experiment scripts (self-contained CLIs, see module docstrings)
runs/            sbatch wrappers for every stage + dashboard.py (terminal + web monitor)
results/         small JSON/CSV summaries of every experiment (large artefacts stay on scratch)
```

Large artefacts (160 GB replay buffer, checkpoints, sprite galleries, HTML
reports, .npz activation dumps) live outside git under
`/mmfs1/scratch/.../expwm-runs/` — paths are hard-wired in `runs/*.sbatch`;
adjust for your cluster.

## Reproducing

1. Clone upstream code (this repo deliberately does not vendor it):
   `git clone https://github.com/vmicheli/delta-iris` (and optionally `eloialonso/iris`).
2. Build the env (Python 3.10): `pip install pip==23.0 setuptools==65.5.0 'wheel<0.40'`,
   then `pip install -r delta-iris/requirements.txt` (torch 2.1.2 cu121 wheels), then
   `pip install scipy scikit-learn`.
   Δ-IRIS uses bf16 autocast — needs Ampere or newer (it will not run on V100).
3. Train: `runs/full_delta_iris.sbatch` (≈ 4–9 days on one H200 for 1000 epochs;
   checkpoint + replay buffer land on scratch). Resume with `runs/resume_delta_iris.sbatch`.
4. Run the analysis stages in the table order — each `runs/*.sbatch` wraps one
   `analysis/*.py` script; each script documents its inputs/outputs in its docstring.
5. Monitor anything with `runs/dashboard.py` (terminal UI, or `--serve` for a
   browser dashboard over an SSH tunnel).

## Caveats worth knowing

- Single seed, single environment (Crafter). Atari configs untested.
- `make_iron_*` / `collect_diamond` / `eat_plant` unlock too rarely (≤ 2/500
  episodes) for any statistics.
- Combat detector codes are weak (best P(ach|code) ≈ 7 %), so
  imagination-ablation outcomes for defeat_* are unreliable.
- Slurm stdout is latin-1 on this cluster: keep prints ASCII or export
  `PYTHONIOENCODING=utf-8` (the sbatch files do).
