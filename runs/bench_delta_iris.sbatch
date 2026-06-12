#!/bin/bash
#SBATCH --job-name=delta-iris-bench
#SBATCH --partition=gpu-short
#SBATCH --nodelist=gpu11,gpu12
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/delta-iris-bench-%j.out
#SBATCH --error=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/delta-iris-bench-%j.err

set -euo pipefail

source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate /mmfs1/storage/users/xiar3/exp/ExpWM/envs/delta-iris-env

echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv

RUN_OUT=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/delta-iris-bench-${SLURM_JOB_ID}
mkdir -p "$RUN_OUT"
cd /mmfs1/storage/users/xiar3/exp/ExpWM/delta-iris

# Run 2 epochs with FULL default config except:
#  - epochs=2 (epoch 1 is expensive, epoch 2 is "typical")
#  - evaluation off (eval kicks in every 50 epochs so wouldn't trigger anyway,
#    but we make sure)
#  - wandb disabled
#  - checkpointing off (skip dataset-to-disk write)
echo "=== launching Δ-IRIS bench (Crafter, 2 epochs, default config) ==="
SECONDS=0
python src/main.py \
  hydra.run.dir="$RUN_OUT/hydra" \
  params.common.device=cuda:0 \
  params.common.epochs=2 \
  params.common.do_checkpoint=False \
  wandb.mode=disabled \
  params.evaluation.should=False 2>&1
echo "=== bench exit: $? after ${SECONDS}s ==="
