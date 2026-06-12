#!/bin/bash
#SBATCH --job-name=delta-iris-smoke
#SBATCH --partition=gpu-short
#SBATCH --nodelist=gpu11,gpu12
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/delta-iris-smoke-%j.out
#SBATCH --error=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/delta-iris-smoke-%j.err

set -euo pipefail

source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate /mmfs1/storage/users/xiar3/exp/ExpWM/envs/delta-iris-env

echo "=== nvidia-smi ==="
nvidia-smi || true
echo "=== python ==="
which python && python --version
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devs', torch.cuda.device_count()); t=torch.tensor([1.0]).cuda(); print('alloc OK', t)"

# Trainer copies <orig_cwd>/src and <orig_cwd>/scripts, so we must launch
# from the repo root. Redirect Hydra's output dir into runs/.
RUN_OUT=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/delta-iris-smoke-${SLURM_JOB_ID}
mkdir -p "$RUN_OUT"
cd /mmfs1/storage/users/xiar3/exp/ExpWM/delta-iris

echo "=== launching Δ-IRIS smoke train (Crafter) ==="
python src/main.py \
  hydra.run.dir="$RUN_OUT/hydra" \
  params.common.device=cuda:0 \
  params.common.epochs=1 \
  params.common.do_checkpoint=False \
  wandb.mode=disabled \
  params.collection.train.num_envs=1 \
  params.collection.train.config.num_steps=200 \
  params.collection.train.config.num_steps_first_epoch=200 \
  params.training.tokenizer.steps_first_epoch=3 \
  params.training.tokenizer.steps_per_epoch=3 \
  params.training.tokenizer.batch_num_samples=4 \
  params.training.world_model.steps_first_epoch=3 \
  params.training.world_model.steps_per_epoch=3 \
  params.training.world_model.batch_num_samples=4 \
  params.training.actor_critic.steps_per_epoch=3 \
  params.training.actor_critic.batch_num_samples=4 \
  params.evaluation.should=False 2>&1
echo "=== smoke train exit: $? ==="
