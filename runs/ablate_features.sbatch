#!/bin/bash
#SBATCH --job-name=ablate-features
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=gpu11,gpu12
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/ablate-features-%j.out
#SBATCH --error=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/ablate-features-%j.err

set -euo pipefail
source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate /mmfs1/storage/users/xiar3/exp/ExpWM/envs/delta-iris-env
export PYTHONIOENCODING=utf-8 LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

RUN=/mmfs1/scratch/hpc/11/xiar3/expwm-runs/delta-iris-full-21531393/hydra
ANA=/mmfs1/scratch/hpc/11/xiar3/expwm-runs/analysis-21531393

N_TRIALS="${N_TRIALS:-80}"

# Run two conditions back-to-back so the user can see both immediate causal
# effect (gap=0) and downstream propagation (gap=3) on one report.
for G in 0 3; do
  echo "=== running gap=$G ==="
  SECONDS=0
  python /mmfs1/storage/users/xiar3/exp/ExpWM/analysis/ablate_features.py \
    --run "$RUN" \
    --rollouts "$ANA/rollouts_probe.npz" \
    --obs "$ANA/rollouts_probe_obs.npz" \
    --sae "$ANA/sae/sae.pt" \
    --features "$ANA/sae/features.json" \
    --out "$ANA/ablation_gap${G}" \
    --n-trials-per-set "$N_TRIALS" \
    --window 21 \
    --gap "$G" \
    --target-block 1 \
    --strength 1.0 \
    --device cuda:0 2>&1
  echo "gap=$G done after ${SECONDS}s"
done
echo "=== exit $? ==="
