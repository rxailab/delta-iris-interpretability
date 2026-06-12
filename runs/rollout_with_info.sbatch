#!/bin/bash
#SBATCH --job-name=rollout-info
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=gpu11,gpu12
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/rollout-info-%j.out
#SBATCH --error=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/rollout-info-%j.err

set -euo pipefail
source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate /mmfs1/storage/users/xiar3/exp/ExpWM/envs/delta-iris-env

RUN=/mmfs1/scratch/hpc/11/xiar3/expwm-runs/delta-iris-full-21531393/hydra
OUT_DIR=/mmfs1/scratch/hpc/11/xiar3/expwm-runs/analysis-21531393
N_EPS="${N_EPS:-500}"
TEMP="${TEMP:-1.0}"

echo "=== job $SLURM_JOB_ID on $SLURMD_NODENAME ==="
nvidia-smi --query-gpu=name --format=csv
echo "RUN=$RUN  OUT=$OUT_DIR  N_EPS=$N_EPS  TEMP=$TEMP"

SECONDS=0
python /mmfs1/storage/users/xiar3/exp/ExpWM/analysis/rollout_with_info.py \
  --run "$RUN" \
  --out "$OUT_DIR/rollouts.npz" \
  --n-episodes "$N_EPS" \
  --temperature "$TEMP" \
  --device cuda:0 2>&1
echo "=== rollout exit: $? after ${SECONDS}s ==="

echo
echo "=== running MI analysis ==="
python /mmfs1/storage/users/xiar3/exp/ExpWM/analysis/code_achievement_mi.py \
  --rollouts "$OUT_DIR/rollouts.npz" \
  --out "$OUT_DIR/mi" \
  --gallery "$OUT_DIR/gallery" \
  --top-k 12 --min-support 5 2>&1
echo "=== MI exit: $? ==="
