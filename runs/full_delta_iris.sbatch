#!/bin/bash
#SBATCH --job-name=delta-iris-full
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=gpu11,gpu12
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=5-12:00:00
#SBATCH --output=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/delta-iris-full-%j.out
#SBATCH --error=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/delta-iris-full-%j.err
#SBATCH --signal=B:USR1@300

set -euo pipefail

source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate /mmfs1/storage/users/xiar3/exp/ExpWM/envs/delta-iris-env

echo "=== job $SLURM_JOB_ID on $SLURMD_NODENAME ==="
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); t=torch.tensor([1.0]).cuda(); print('alloc OK', t)"

# Outputs live on the 10TB scratch fileset (the 200G storage fileset can't
# hold Δ-IRIS's growing replay buffer for a 1000-epoch run).
RUN_OUT=/mmfs1/scratch/hpc/11/xiar3/expwm-runs/delta-iris-full-${SLURM_JOB_ID}
mkdir -p "$RUN_OUT"
cd /mmfs1/storage/users/xiar3/exp/ExpWM/delta-iris

echo "=== launching Δ-IRIS full Crafter run (1000 epochs, seed=0) ==="
SECONDS=0
python src/main.py \
  hydra.run.dir="$RUN_OUT/hydra" \
  params.common.device=cuda:0 \
  params.common.seed=0 \
  wandb.mode=disabled 2>&1
echo "=== full run exit: $? after ${SECONDS}s ==="
