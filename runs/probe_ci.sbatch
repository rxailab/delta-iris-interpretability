#!/bin/bash
#SBATCH --job-name=probe-ci
#SBATCH --partition=gpu-medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=gpu11,gpu12
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/probe-ci-%j.out
#SBATCH --error=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/probe-ci-%j.err

set -euo pipefail
source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate /mmfs1/storage/users/xiar3/exp/ExpWM/envs/delta-iris-env
export PYTHONIOENCODING=utf-8
export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

RUN=/mmfs1/scratch/hpc/11/xiar3/expwm-runs/delta-iris-full-21531393/hydra
ANA=/mmfs1/scratch/hpc/11/xiar3/expwm-runs/analysis-21531393

echo "=== job $SLURM_JOB_ID on $SLURMD_NODENAME ==="
nvidia-smi --query-gpu=name --format=csv

echo
echo "=== probe CI sweep (bootstrap B=1000 + 5-fold CV + HUD margins) ==="
SECONDS=0
python /mmfs1/storage/users/xiar3/exp/ExpWM/analysis/probe_ci.py \
  --run "$RUN" \
  --rollouts "$ANA/rollouts_probe.npz" \
  --obs "$ANA/rollouts_probe_obs.npz" \
  --out "$ANA/probe_ci" \
  --n-boot 1000 \
  --n-folds 5 \
  --device cuda:0 2>&1
echo "probe-ci time ${SECONDS}s"
echo "=== done ==="
