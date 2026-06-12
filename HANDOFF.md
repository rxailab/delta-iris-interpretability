# HANDOFF — Δ-IRIS Interpretability Project

State of the project as of 2026-06-12. Written for whoever picks this up next
(future me, a collaborator, or an assistant with no memory of the sessions
that produced it). Read this top-to-bottom before touching anything.

---

## 1. What this project is

Goal: **extract explainability from the quantized (delta-token) representations
of Δ-IRIS** ([vmicheli/delta-iris](https://github.com/vmicheli/delta-iris),
ICML 2024) — a world-model RL agent whose tokenizer encodes *frame-to-frame
deltas conditioned on action* as 4 discrete codes per timestep (2×2 spatial
slots, codebook 1024, ~40 bits/transition).

One full agent was trained (Crafter, 1000 epochs, seed 0), then a 9-stage
analysis pipeline was built on top of the frozen checkpoint. Everything is
scripted and re-runnable; all conclusions below have a JSON/CSV summary in
`results/` and a browsable HTML artefact on scratch.

## 2. Where everything lives

| thing | path |
|---|---|
| this repo (scripts, sbatch, summaries) | `/mmfs1/storage/users/xiar3/exp/ExpWM` → [rxailab/delta-iris-interpretability](https://github.com/rxailab/delta-iris-interpretability) (private) |
| upstream clones (not in git) | `ExpWM/iris`, `ExpWM/delta-iris` |
| conda envs (not in git) | `ExpWM/envs/iris-env`, `ExpWM/envs/delta-iris-env` (py3.10, torch 2.1.2+cu121; delta-iris-env also has scipy, scikit-learn) |
| **trained run** (checkpoint `last.pt`, 160 GB replay buffer, hydra config) | `/mmfs1/scratch/hpc/11/xiar3/expwm-runs/delta-iris-full-21531393/hydra` |
| **all analysis artefacts** (HTML reports, galleries, npz dumps) | `/mmfs1/scratch/hpc/11/xiar3/expwm-runs/analysis-21531393/` |
| Slurm logs of every job | `ExpWM/runs/*-<jobid>.out` (gitignored) |
| project memory (assistant) | `~/.claude/projects/-mmfs1-storage-users-xiar3-exp-ExpWM/memory/` |

Two long-running tmux sessions serve dashboards over `ssh -L`:
`dash` (training monitor, :8765) and `gallery` (`python3 -m http.server 8766`
rooted at `analysis-21531393/` — serves every HTML artefact below).

## 3. The trained agent

- Job chain: 21531393 (epochs 1–326, died: storage quota) → 21577143 (327–918,
  died: 5d12h walltime) → 21690008 (919–1000, completed). Total ≈ 8.7 days on
  one H200.
- Final eval: mean return ≈ 16.2 (500-episode re-roll), i.e. ~16/22
  achievements per episode. Paper reports 17.8 ± 1.4 multi-seed — we are one
  seed, in plausible range.
- Never-unlocked achievements: `collect_diamond`, `eat_plant`; ~2/500 for
  `make_iron_pickaxe`/`make_iron_sword`. **No statistics exist for these.**

## 4. Findings (each → artefact)

Stage order matters: each tier motivated the next.

1. **Codebook stats** (`results/meta.json`, gallery `analysis-21531393/gallery/`):
   all 1024 codes used somewhere; per-slot entropy ~6/10 bits → ~50 effective
   codes/slot; the dominant code per slot is a "nothing changed in this
   quadrant" code (slot 2's takes 47% of mass).
2. **MI table** (`results/mi/mi_table.json`, `mi/achievements.html`): codes are
   near-monosemantic event detectors. E.g. (s3,c75) ⇒ collect_iron with
   P(ach|code)=0.88, lift 707×. Some codes are polysemantic superclasses —
   (s3,c36) fires for place_table AND make_wood_pickaxe AND make_wood_sword.
3. **Layer probes + HUD control** (`results/probes*/`): `ach_just[*]` decodable
   from raw codes alone (AUROC ≥ .97). `ach_cum[*]` looks transformer-dependent
   until you probe `frame_emb` (CNN over current frame, no transformer): it
   matches/beats every block on 15/18 concepts because **Crafter renders the
   inventory in the HUD**. Only `reward_in_next_5` (anticipation) genuinely
   improves with depth (0.79 → 0.81, and frame_emb can't do it).
   ⚠ Naming wart: rep "wm_input" in probe outputs is actually the *output of
   transformer block 0* (hooks capture block exits). True pre-transformer reps
   are `frame_emb` and `wm_latents_emb` (only in `probes_hud/`).
4. **CAVs** (`results/cavs/cav_metrics.json`, `cavs/cavs.npz` on scratch, traces
   `cavs/traces.html`): 35 concept directions per WM rep with held-out AUROC ≈ 1.
   CAV cosine geometry mirrors the tech tree (cos(cum[collect_stone],
   cum[place_stone]) = +0.95; just[X] vs cum[X] ≈ −0.2 i.e. event ⊥ state).
5. **SAE** (`results/sae/features.json`, `sae/sae.pt` + `features.html` on
   scratch): TopK SAE, 2048 dict, k=16, trained on `wm_block_1` summary
   activations (mean of the 4 latent positions), val FVU 0.043, zero dead
   features. Six features triply-corroborate (achievement lift + CAV cos +
   MI-table code). **Key result: the SAE splits polysemantic code (s3,c36)
   into separate place_table (f677) / make_wood_pickaxe (f1451) /
   make_wood_sword (f372) features.**
6. **One-step ablation** (`results/ablation_gap0|gap3/causal_effects.json`):
   project the best SAE-feature direction out of block-1 output at the unlock
   step → up to +4.9 nats damage to predicting the real codes (wake_up), with
   double controls (ordinary moments ≈ 0, random direction ≈ 0). At gap=3 the
   effect vanishes entirely → model recovers via the frame path.
7. **Imagination ablation** (`results/imagination/imagine_effects.json`,
   `imagination/imagination.html`): sustained ablation during 12-step dreams.
   Headline: **collect_wood P(detector code in dream) 1.00 → 0.00/0.10; CAV
   direction and random direction both inert.** Moderate effects:
   make_stone_sword .75→.45, collect_coal .95→.70, collect_iron 1.0→.85.
   **CAV ablation does nothing anywhere → read directions ≠ write directions.**
   Known oddities: defeat_* outcomes unreliable (weak detector codes, P(a|c)≈7%);
   wake_up shows a paradoxical increase (.30→.50) under ablation.
8. **Causal tracing** (`results/causal_trace/trace_results.json`,
   `causal_trace/causal_trace.html`): activation patching, 3 blocks × 21 steps ×
   {frame, act, latents}. All restoration mass at the FINAL step (history cells
   ≈ 0.00); action token is the biggest single carrier (.46–.58 at block 2),
   latents next, frame least. → unlock-code prediction is quasi-Markovian.

**The through-line for a paper:** codes = event vocabulary (descriptive),
frame CNN = state via HUD shortcut (methodological control), transformer =
anticipation only (informational), SAE features = the causal write-variables
(interventional), probes/CAVs = readable but causally inert. Three independent
experiments (HUD probes, gap-3 recovery, tracing) converge on the
quasi-Markov/HUD claim — that triangulation is the paper's methodological spine.

## 5. How to re-run things

Every stage = one sbatch in `runs/` wrapping one script in `analysis/`
(self-documenting docstrings, plain CLIs). Standard pattern:

```bash
cd /mmfs1/storage/users/xiar3/exp/ExpWM/runs
sbatch <stage>.sbatch                      # full run
sbatch --export=ALL,N_TRIALS=10 <stage>.sbatch   # smaller smoke (var names differ per file)
```

Dependencies between stages (data flow):
`rollout_with_info.py` (+`--save-obs`) produces `rollouts_probe.npz` +
`rollouts_probe_obs.npz` → consumed by probes, CAVs, SAE, both ablations,
tracing. The MI table needs `rollouts.npz` (500-ep version without obs).
The gallery needs `codebook_stats.npz` + the on-disk replay buffer.

GPU: everything fits on one H200 (`--nodelist=gpu11,gpu12`); nothing needs
more than ~30 GB VRAM; most stages run in minutes. **Nothing runs on V100**
(Δ-IRIS hard-codes bf16 autocast).

## 6. Cluster gotchas (each cost a failed job once)

- pip-installed torch + system CUDA on `LD_LIBRARY_PATH` → error 803. Don't.
- `gym==0.21.0` needs `pip==23.0`, `setuptools==65.5.0`, `wheel<0.40`.
- Slurm stdout is **latin-1**: a single `→`/`Δ`/`±` in a print kills a job at
  the last line. All sbatch files export `PYTHONIOENCODING=utf-8`; keep it.
- `sbatch --export=ALL,VAR=a,b` splits on the comma. Never pass lists that way.
- sata1-storage quota = 200G hard 220G (killed the first training segment).
  Big outputs → `/mmfs1/scratch/hpc/11/xiar3/` (10T quota).
- Hydra 1.1 + py3.10 masks tracebacks ("print_exception() got etype"); the
  real error is higher up in the .out; read with `sed 's/\r/\n/g'`.
- Loading the saved config requires `OmegaConf.register_new_resolver("eval",
  eval)` and filling `num_actions=17` *before* `OmegaConf.resolve`.
- Δ-IRIS's trainer copies `src/` from the launch cwd — always launch from the
  repo root with `hydra.run.dir` redirected (see `full_delta_iris.sbatch`).

## 7. Open threads (ranked, with effort estimates)

1. **Δ-IRIS vs original-IRIS codebook comparison** (~1 day GPU + ½ day
   analysis). Train the non-delta IRIS tokenizer on the same replay buffer,
   rerun stages 1–2, quantify how much delta-encoding drives code
   monosemanticity. This is the cleanest "why Δ matters" section.
2. **Dream videos for the collect_wood ablation** (~1–2 h). Decode imagined
   frames side-by-side (baseline vs ablated) from `imagine_ablation.py` —
   the poster/talk artefact. The frames are already computed, just not saved.
3. **SAE sweeps** (~½ day). Other blocks (0, 2), widths (8k), k values;
   check whether the collect_wood write-feature is stable across SAEs.
4. **Iron-chain statistics** (~1 day). 5000-episode re-roll to get make_iron_*
   unlocks, then rerun MI + ablations for the rare tail.
5. **Per-quadrant slot semantics** (~½ day). Test the slot↔screen-quadrant
   hypothesis by correlating code activations with *where* in the frame the
   change happened (we never verified the spatial claim).
6. Atari transfer of the whole pipeline (~1 week incl. training).

## 8. Caveats to carry into any write-up

Single seed; single environment; n=10–30 for some per-achievement statistics;
detector-code outcome metric is weak for combat events; "wm_input" naming wart
(see §4.3); imagination horizon limited to 12 steps by the 21-block KV cache
(burn-in 8 + horizon 12 + 1; longer needs cache surgery or shorter burn-in).
