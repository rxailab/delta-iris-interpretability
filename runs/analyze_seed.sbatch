#!/bin/bash
#SBATCH --job-name=ana-seed
#SBATCH --partition=gpu-medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/ana-seed-%j.out
#SBATCH --error=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/ana-seed-%j.err

# Replication analysis: run the headline pipeline on an independently trained seed.
# Submit with:  sbatch --export=ALL,SEED=1 runs/analyze_seed.sbatch
# Stages are fault-tolerant: a failure is logged and the pipeline continues, so one
# bad stage does not waste the rest. Dependency order:
#   codebook_stats -> probe-rollouts(+obs) -> full-rollouts+MI -> probes -> CAVs
#   -> SAE -> projection-ablation(gap0/3) -> imagine-ablation -> behavioural cascade
set -uo pipefail
source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate /mmfs1/storage/users/xiar3/exp/ExpWM/envs/delta-iris-env
export PYTHONIOENCODING=utf-8 LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

SEED="${SEED:?set SEED via --export=ALL,SEED=N}"
RUN=/mmfs1/scratch/hpc/11/xiar3/expwm-runs/delta-iris-seed${SEED}/hydra
ANA=/mmfs1/scratch/hpc/11/xiar3/expwm-runs/analysis-seed${SEED}
A=/mmfs1/storage/users/xiar3/exp/ExpWM/analysis
mkdir -p "$ANA"
echo "=== seed=$SEED RUN=$RUN ANA=$ANA on $SLURMD_NODENAME job $SLURM_JOB_ID ==="
nvidia-smi --query-gpu=name --format=csv

stage(){ echo; echo "######## [$1] start $(date +%T) ########"; SECONDS=0; shift; "$@"; echo "######## exit $? after ${SECONDS}s ########"; }

stage codebook    python $A/codebook_stats.py --run "$RUN" --out "$ANA" \
  --max-episodes "${MAX_EPISODES:-3000}" --batch-size 256 --top-k 32 --device cuda:0 || echo "STAGE codebook FAILED"

stage rollout_probe python $A/rollout_with_info.py --run "$RUN" \
  --out "$ANA/rollouts_probe.npz" --save-obs "$ANA/rollouts_probe_obs.npz" \
  --n-episodes 150 --temperature 1.0 --device cuda:0 || echo "STAGE rollout_probe FAILED"

stage rollout_full python $A/rollout_with_info.py --run "$RUN" \
  --out "$ANA/rollouts.npz" --n-episodes 500 --temperature 1.0 --device cuda:0 || echo "STAGE rollout_full FAILED"

stage mi          python $A/code_achievement_mi.py --rollouts "$ANA/rollouts.npz" \
  --out "$ANA/mi" --gallery "$ANA/gallery" --top-k 12 --min-support 5 || echo "STAGE mi FAILED"

stage probes      python $A/probe_layers.py --run "$RUN" \
  --rollouts "$ANA/rollouts_probe.npz" --obs "$ANA/rollouts_probe_obs.npz" \
  --out "$ANA/probes" --batch-windows 8 --device cuda:0 || echo "STAGE probes FAILED"
python $A/render_probes.py --probes "$ANA/probes" || true

stage cavs        python $A/cav_traces.py --run "$RUN" \
  --rollouts "$ANA/rollouts_probe.npz" --obs "$ANA/rollouts_probe_obs.npz" \
  --out "$ANA/cavs" --trace-rep wm_block_1 --min-auroc 0.90 --n-showcase 6 --device cuda:0 || echo "STAGE cavs FAILED"

stage sae         python $A/train_sae.py --run "$RUN" \
  --rollouts "$ANA/rollouts_probe.npz" --obs "$ANA/rollouts_probe_obs.npz" \
  --cavs "$ANA/cavs/cavs.npz" --codebook-stats "$ANA/codebook_stats.npz" \
  --out "$ANA/sae" --layer wm_block_1 --n-features 2048 --k-active 16 \
  --epochs 80 --batch-size 4096 --lr 5e-4 --device cuda:0 || echo "STAGE sae FAILED"

for G in 0 3; do
stage ablation_gap${G} python $A/ablate_features.py --run "$RUN" \
  --rollouts "$ANA/rollouts_probe.npz" --obs "$ANA/rollouts_probe_obs.npz" \
  --sae "$ANA/sae/sae.pt" --features "$ANA/sae/features.json" \
  --out "$ANA/ablation_gap${G}" --n-trials-per-set 80 --window 21 --gap "$G" \
  --target-block 1 --strength 1.0 --device cuda:0 || echo "STAGE ablation_gap${G} FAILED"
done

stage imagine     python $A/imagine_ablation.py --run "$RUN" \
  --rollouts "$ANA/rollouts_probe.npz" --obs "$ANA/rollouts_probe_obs.npz" \
  --sae "$ANA/sae/sae.pt" --features "$ANA/sae/features.json" \
  --cavs "$ANA/cavs/cavs.npz" --mi-table "$ANA/mi/mi_table.json" \
  --out "$ANA/imagination" --burn-in 8 --horizon 12 --n-moments 20 \
  --target-block 1 --device cuda:0 || echo "STAGE imagine FAILED"

stage cascade     python $A/imagine_cascade.py --run "$RUN" \
  --rollouts "$ANA/rollouts_probe.npz" --obs "$ANA/rollouts_probe_obs.npz" \
  --sae "$ANA/sae/sae.pt" --features "$ANA/sae/features.json" \
  --cavs "$ANA/cavs/cavs.npz" --mi-table "$ANA/mi/mi_table.json" \
  --out "$ANA/cascade_collect_wood" --focus collect_wood \
  --burn-in 8 --horizon 12 --n-moments 40 --target-block 1 --seed 42 --device cuda:0 || echo "STAGE cascade FAILED"

echo; echo "=== ALL STAGES DONE for seed $SEED at $(date +%T) ==="
