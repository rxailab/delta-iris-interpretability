#!/bin/bash
#SBATCH --job-name=codebook-stats
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=gpu11,gpu12
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/codebook-stats-%j.out
#SBATCH --error=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/codebook-stats-%j.err

set -euo pipefail
source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate /mmfs1/storage/users/xiar3/exp/ExpWM/envs/delta-iris-env

RUN=/mmfs1/scratch/hpc/11/xiar3/expwm-runs/delta-iris-full-21531393/hydra
OUT=/mmfs1/scratch/hpc/11/xiar3/expwm-runs/analysis-21531393
MAX_EPISODES="${MAX_EPISODES:--1}"   # override with --export=ALL,MAX_EPISODES=50

echo "=== job $SLURM_JOB_ID on $SLURMD_NODENAME ==="
nvidia-smi --query-gpu=name,driver_version --format=csv
echo "RUN=$RUN"
echo "OUT=$OUT"
echo "MAX_EPISODES=$MAX_EPISODES"

SECONDS=0
python /mmfs1/storage/users/xiar3/exp/ExpWM/analysis/codebook_stats.py \
  --run "$RUN" \
  --out "$OUT" \
  --max-episodes "$MAX_EPISODES" \
  --batch-size 256 \
  --top-k 32 \
  --device cuda:0 2>&1
echo "=== codebook_stats exit: $? after ${SECONDS}s ==="
