#!/bin/bash
#SBATCH --job-name=steer-imagination
#SBATCH --partition=gpu-medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=gpu11,gpu12
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/steer-imagination-%j.out
#SBATCH --error=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/steer-imagination-%j.err

set -euo pipefail
source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate /mmfs1/storage/users/xiar3/exp/ExpWM/envs/delta-iris-env
export PYTHONIOENCODING=utf-8 LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

RUN=/mmfs1/scratch/hpc/11/xiar3/expwm-runs/delta-iris-full-21531393/hydra
ANA=/mmfs1/scratch/hpc/11/xiar3/expwm-runs/analysis-21531393

SECONDS=0
python /mmfs1/storage/users/xiar3/exp/ExpWM/analysis/steer_imagination.py \
  --run "$RUN" \
  --rollouts "$ANA/rollouts_probe.npz" \
  --obs "$ANA/rollouts_probe_obs.npz" \
  --sae "$ANA/sae/sae.pt" \
  --features "$ANA/sae/features.json" \
  --cavs "$ANA/cavs/cavs.npz" \
  --mi-table "$ANA/mi/mi_table.json" \
  --suppress-effects "$ANA/imagination/imagine_effects.json" \
  --out "$ANA/steer_imagination" \
  --burn-in 8 --horizon 12 \
  --n-moments "${N_MOMENTS:-20}" \
  --alphas "${ALPHAS:-2,4,8}" \
  --max-concepts "${MAX_CONCEPTS:-12}" \
  --concepts "${CONCEPTS:-}" \
  --target-block 1 \
  --device cuda:0 2>&1
echo "=== exit $? after ${SECONDS}s ==="
