#!/bin/bash
#SBATCH --job-name=tok-frameonly
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=gpu11,gpu12
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/tok-frameonly-%j.out
#SBATCH --error=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/tok-frameonly-%j.err

set -euo pipefail
source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate /mmfs1/storage/users/xiar3/exp/ExpWM/envs/delta-iris-env
export PYTHONIOENCODING=utf-8 LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

RUN=/mmfs1/scratch/hpc/11/xiar3/expwm-runs/delta-iris-full-21531393/hydra
OUT=/mmfs1/scratch/hpc/11/xiar3/expwm-runs/baselines

SECONDS=0
python /mmfs1/storage/users/xiar3/exp/ExpWM/analysis/train_baseline_tokenizer.py \
  --deltairis-src "$RUN" --out "$OUT" \
  --steps "${STEPS:-60000}" --batch 64 --lr 1e-4 --device cuda:0 2>&1
echo "=== exit $? after ${SECONDS}s ==="
