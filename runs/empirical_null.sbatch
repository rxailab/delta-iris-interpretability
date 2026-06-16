#!/bin/bash
#SBATCH --job-name=empirical-null
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=gpu11,gpu12
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/empirical-null-%j.out
#SBATCH --error=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/empirical-null-%j.err

set -euo pipefail
source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate /mmfs1/storage/users/xiar3/exp/ExpWM/envs/delta-iris-env
export PYTHONIOENCODING=utf-8 LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

RUN=/mmfs1/scratch/hpc/11/xiar3/expwm-runs/delta-iris-full-21531393/hydra
ANA=/mmfs1/scratch/hpc/11/xiar3/expwm-runs/analysis-21531393

SECONDS=0
python /mmfs1/storage/users/xiar3/exp/ExpWM/analysis/empirical_null.py \
  --run "$RUN" \
  --rollouts "$ANA/rollouts_probe.npz" \
  --obs "$ANA/rollouts_probe_obs.npz" \
  --sae "$ANA/sae/sae.pt" \
  --features "$ANA/sae/features.json" \
  --out "$ANA/empirical_null_gap${GAP:-0}" \
  --n-moments "${N_MOMENTS:-40}" \
  --n-random "${N_RANDOM:-50}" \
  --max-concepts "${MAX_CONCEPTS:-8}" \
  --window 21 \
  --gap "${GAP:-0}" \
  --target-block 1 \
  --strength 1.0 \
  --device cuda:0 2>&1
echo "=== exit $? after ${SECONDS}s ==="
