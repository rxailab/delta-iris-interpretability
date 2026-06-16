#!/bin/bash
#SBATCH --job-name=di-seed
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=gpu11,gpu12
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=5-12:00:00
#SBATCH --output=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/di-seed-%j.out
#SBATCH --error=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/di-seed-%j.err

# Multi-seed Delta-IRIS training for seed-robustness validation.
# Seed is passed via the SEED env var (sbatch --export). Each run lives in a
# FIXED per-seed scratch dir so chained jobs (afterany) resume it; a run that
# already reached 1000 epochs exits immediately, so over-provisioning the
# chain is safe.
set -uo pipefail
: "${SEED:?export SEED=N before sbatch}"

RUN_OUT=/mmfs1/scratch/hpc/11/xiar3/expwm-runs/delta-iris-seed${SEED}
HY="$RUN_OUT/hydra"
REPO=/mmfs1/storage/users/xiar3/exp/ExpWM/delta-iris
mkdir -p "$RUN_OUT"

source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate /mmfs1/storage/users/xiar3/exp/ExpWM/envs/delta-iris-env

echo "=== job $SLURM_JOB_ID  seed=$SEED  node=$SLURMD_NODENAME  $(date) ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
python -c "import torch; assert torch.cuda.is_available(); print('cuda OK', torch.__version__)"

cur_epoch() { python -c "import torch; print(int(torch.load('$HY/checkpoints/epoch.pt')))" 2>/dev/null || echo 0; }

EP=$(cur_epoch)
echo "current checkpoint epoch = $EP"
if [ "$EP" -ge 1000 ]; then
  echo "seed $SEED already complete (epoch $EP) — nothing to do."
  exit 0
fi

SECONDS=0
if [ -f "$HY/checkpoints/epoch.pt" ]; then
  echo "=== resuming seed $SEED from epoch $EP ==="
  cd "$HY"
  python "$REPO/src/main.py" \
    hydra.run.dir=. hydra.output_subdir=null \
    params.common.resume=true \
    params.common.device=cuda:0 \
    wandb.mode=disabled 2>&1
else
  echo "=== fresh start seed $SEED (1000 epochs) ==="
  cd "$REPO"
  python src/main.py \
    hydra.run.dir="$HY" \
    params.common.device=cuda:0 \
    params.common.seed=${SEED} \
    wandb.mode=disabled 2>&1
fi
RC=$?
echo "=== seed $SEED exit $RC after ${SECONDS}s; epoch now $(cur_epoch) ==="
