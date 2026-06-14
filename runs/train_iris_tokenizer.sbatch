#!/bin/bash
#SBATCH --job-name=tok-iris
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=gpu11,gpu12
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/tok-iris-%j.out
#SBATCH --error=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/tok-iris-%j.err

set -euo pipefail
source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate /mmfs1/storage/users/xiar3/exp/ExpWM/envs/iris-env
export PYTHONIOENCODING=utf-8 LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

IRIS_SRC=/mmfs1/storage/users/xiar3/exp/ExpWM/iris/src
DATASET=/mmfs1/scratch/hpc/11/xiar3/expwm-runs/delta-iris-full-21531393/hydra/checkpoints/dataset/train
OUT=/mmfs1/scratch/hpc/11/xiar3/expwm-runs/baselines

SECONDS=0
python /mmfs1/storage/users/xiar3/exp/ExpWM/analysis/train_iris_tokenizer.py \
  --iris-src "$IRIS_SRC" --dataset "$DATASET" --out "$OUT" \
  --steps "${STEPS:-60000}" --batch 64 --lr 1e-4 --device cuda:0 2>&1
echo "=== exit $? after ${SECONDS}s ==="
