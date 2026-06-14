#!/bin/bash
#SBATCH --job-name=tok-mono
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=gpu11,gpu12
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/tok-mono-%j.out
#SBATCH --error=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/tok-mono-%j.err

set -euo pipefail
source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate /mmfs1/storage/users/xiar3/exp/ExpWM/envs/delta-iris-env
export PYTHONIOENCODING=utf-8 LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

RUN=/mmfs1/scratch/hpc/11/xiar3/expwm-runs/delta-iris-full-21531393/hydra
ANA=/mmfs1/scratch/hpc/11/xiar3/expwm-runs/analysis-21531393
BASE=/mmfs1/scratch/hpc/11/xiar3/expwm-runs/baselines
IRIS_SRC=/mmfs1/storage/users/xiar3/exp/ExpWM/iris/src
OUT="$ANA/mono_compare"
MONO=/mmfs1/storage/users/xiar3/exp/ExpWM/analysis/tokenizer_mono_compare.py
ROLL="$ANA/rollouts_probe.npz"; OBS="$ANA/rollouts_probe_obs.npz"

echo "=== deltairis ==="
python "$MONO" --kind deltairis --deltairis-src "$RUN" --rollouts "$ROLL" --obs "$OBS" --out "$OUT" --device cuda:0 2>&1
echo "=== frameonly ==="
python "$MONO" --kind frameonly --deltairis-src "$RUN" --ckpt "$BASE/frameonly_tokenizer.pt" --rollouts "$ROLL" --obs "$OBS" --out "$OUT" --device cuda:0 2>&1
echo "=== iris ==="
python "$MONO" --kind iris --deltairis-src "$RUN" --iris-src "$IRIS_SRC" --ckpt "$BASE/iris_tokenizer.pt" --rollouts "$ROLL" --obs "$OBS" --out "$OUT" --device cuda:0 2>&1
echo "=== all done ==="
