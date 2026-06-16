#!/bin/bash
#SBATCH --job-name=subspace-ablation
#SBATCH --partition=gpu-medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=gpu11,gpu12
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/subspace-ablation-%j.out
#SBATCH --error=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/subspace-ablation-%j.err

set -euo pipefail
source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate /mmfs1/storage/users/xiar3/exp/ExpWM/envs/delta-iris-env
export PYTHONIOENCODING=utf-8 LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

RUN=/mmfs1/scratch/hpc/11/xiar3/expwm-runs/delta-iris-full-21531393/hydra
ANA=/mmfs1/scratch/hpc/11/xiar3/expwm-runs/analysis-21531393

SECONDS=0
python /mmfs1/storage/users/xiar3/exp/ExpWM/analysis/subspace_ablation.py \
  --run "$RUN" \
  --rollouts "$ANA/rollouts_probe.npz" \
  --obs "$ANA/rollouts_probe_obs.npz" \
  --sae "$ANA/sae/sae.pt" \
  --features "$ANA/sae/features.json" \
  --cavs "$ANA/cavs/cavs.npz" \
  --out "$ANA/subspace_ablation_gap${GAP:-0}" \
  --n-trials-per-set "${N_TRIALS:-40}" \
  --window 21 \
  --gap "${GAP:-0}" \
  --ranks "${RANKS:-1,2,4,8,12,16}" \
  --max-concepts "${MAX_CONCEPTS:-8}" \
  --concepts "${CONCEPTS:-}" \
  --target-block 1 \
  --strength 1.0 \
  --seed 42 \
  --device cuda:0 2>&1
echo "=== exit $? after ${SECONDS}s ==="
