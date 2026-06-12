#!/bin/bash
#SBATCH --job-name=cav-traces
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=gpu11,gpu12
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/cav-traces-%j.out
#SBATCH --error=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/cav-traces-%j.err

set -euo pipefail
source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate /mmfs1/storage/users/xiar3/exp/ExpWM/envs/delta-iris-env
export PYTHONIOENCODING=utf-8 LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

RUN=/mmfs1/scratch/hpc/11/xiar3/expwm-runs/delta-iris-full-21531393/hydra
ANA=/mmfs1/scratch/hpc/11/xiar3/expwm-runs/analysis-21531393

echo "=== job $SLURM_JOB_ID on $SLURMD_NODENAME ==="
SECONDS=0
python /mmfs1/storage/users/xiar3/exp/ExpWM/analysis/cav_traces.py \
  --run "$RUN" \
  --rollouts "$ANA/rollouts_probe.npz" \
  --obs "$ANA/rollouts_probe_obs.npz" \
  --out "$ANA/cavs" \
  --trace-rep wm_block_1 \
  --min-auroc 0.90 \
  --n-showcase 6 \
  --device cuda:0 2>&1
echo "=== exit $? after ${SECONDS}s ==="
