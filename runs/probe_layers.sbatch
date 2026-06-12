#!/bin/bash
#SBATCH --job-name=probe-layers
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=gpu11,gpu12
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/probe-layers-%j.out
#SBATCH --error=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/probe-layers-%j.err

set -euo pipefail
source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate /mmfs1/storage/users/xiar3/exp/ExpWM/envs/delta-iris-env
export PYTHONIOENCODING=utf-8
export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

RUN=/mmfs1/scratch/hpc/11/xiar3/expwm-runs/delta-iris-full-21531393/hydra
ANA=/mmfs1/scratch/hpc/11/xiar3/expwm-runs/analysis-21531393
N_EPS="${N_EPS:-150}"

echo "=== job $SLURM_JOB_ID on $SLURMD_NODENAME ==="
nvidia-smi --query-gpu=name --format=csv

# Step 1: re-roll with obs (smaller N, only for probing)
echo
echo "=== rollout with obs (N_EPS=$N_EPS) ==="
SECONDS=0
python /mmfs1/storage/users/xiar3/exp/ExpWM/analysis/rollout_with_info.py \
  --run "$RUN" \
  --out "$ANA/rollouts_probe.npz" \
  --save-obs "$ANA/rollouts_probe_obs.npz" \
  --n-episodes "$N_EPS" \
  --temperature 1.0 --device cuda:0 2>&1
echo "rollout time ${SECONDS}s"

# Step 2: extract activations + train probes
echo
echo "=== probe sweep ==="
SECONDS=0
python /mmfs1/storage/users/xiar3/exp/ExpWM/analysis/probe_layers.py \
  --run "$RUN" \
  --rollouts "$ANA/rollouts_probe.npz" \
  --obs "$ANA/rollouts_probe_obs.npz" \
  --out "$ANA/probes" \
  --batch-windows 8 --device cuda:0 2>&1
echo "probe time ${SECONDS}s"

# Step 3: render HTML
echo
echo "=== render report ==="
python /mmfs1/storage/users/xiar3/exp/ExpWM/analysis/render_probes.py \
  --probes "$ANA/probes" 2>&1
echo "=== done ==="
