#!/bin/bash
#SBATCH --job-name=iris-smoke
#SBATCH --partition=gpu-short
#SBATCH --nodelist=gpu11,gpu12
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/iris-smoke-%j.out
#SBATCH --error=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/iris-smoke-%j.err

set -euo pipefail

source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate /mmfs1/storage/users/xiar3/exp/ExpWM/envs/iris-env

echo "=== nvidia-smi ==="
nvidia-smi || true
echo "=== python ==="
which python && python --version
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devs', torch.cuda.device_count()); t=torch.tensor([1.0]).cuda(); print('alloc OK', t)"

# Trainer copies <orig_cwd>/src and <orig_cwd>/scripts, so we must launch
# from the repo root. Redirect Hydra's output dir into runs/.
RUN_OUT=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/iris-smoke-${SLURM_JOB_ID}
mkdir -p "$RUN_OUT"
cd /mmfs1/storage/users/xiar3/exp/ExpWM/iris

echo "=== launching IRIS smoke train ==="
python src/main.py \
  hydra.run.dir="$RUN_OUT/hydra" \
  env.train.id=BreakoutNoFrameskip-v4 \
  common.device=cuda:0 \
  common.epochs=1 \
  wandb.mode=disabled \
  collection.train.num_envs=1 \
  collection.train.config.num_steps=200 \
  training.tokenizer.start_after_epochs=0 \
  training.tokenizer.steps_per_epoch=3 \
  training.tokenizer.batch_num_samples=4 \
  training.world_model.start_after_epochs=0 \
  training.world_model.steps_per_epoch=3 \
  training.world_model.batch_num_samples=4 \
  training.actor_critic.start_after_epochs=0 \
  training.actor_critic.steps_per_epoch=3 \
  training.actor_critic.batch_num_samples=4 \
  evaluation.should=False \
  common.do_checkpoint=False 2>&1
echo "=== smoke train exit: $? ==="
