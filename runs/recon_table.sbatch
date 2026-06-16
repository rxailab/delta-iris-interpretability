#!/bin/bash
#SBATCH --job-name=recon-table
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=gpu11,gpu12
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/recon-table-%j.out
#SBATCH --error=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/recon-table-%j.err

set -euo pipefail
source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate /mmfs1/storage/users/xiar3/exp/ExpWM/envs/delta-iris-env
export PYTHONIOENCODING=utf-8 LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

RUN=/mmfs1/scratch/hpc/11/xiar3/expwm-runs/delta-iris-full-21531393/hydra
ANA=/mmfs1/scratch/hpc/11/xiar3/expwm-runs/analysis-21531393
BASE=/mmfs1/scratch/hpc/11/xiar3/expwm-runs/baselines
IRIS_SRC=/mmfs1/storage/users/xiar3/exp/ExpWM/iris/src
SCRIPT=/mmfs1/storage/users/xiar3/exp/ExpWM/analysis/recon_table.py
OUT="$ANA/recon_table"
OBS="$ANA/rollouts_probe_obs.npz"
ROLL="$ANA/rollouts_probe.npz"

SECONDS=0
python "$SCRIPT" \
  --deltairis-src "$RUN" \
  --iris-src "$IRIS_SRC" \
  --frameonly-ckpt "$BASE/frameonly_tokenizer.pt" \
  --iris-ckpt "$BASE/iris_tokenizer.pt" \
  --obs "$OBS" \
  --rollouts "$ROLL" \
  --frameonly-train-log "$BASE/frameonly_train_log.json" \
  --iris-train-log "$BASE/iris_train_log.json" \
  --out "$OUT" \
  --n-samples 8192 --batch 256 --seed 0 --device cuda:0 2>&1
echo "=== exit $? after ${SECONDS}s ==="
