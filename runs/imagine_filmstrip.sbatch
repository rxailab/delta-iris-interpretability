#!/bin/bash
#SBATCH --job-name=imagine-film
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=gpu11,gpu12
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/imagine-film-%j.out
#SBATCH --error=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/imagine-film-%j.err

set -euo pipefail
source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate /mmfs1/storage/users/xiar3/exp/ExpWM/envs/delta-iris-env
export PYTHONIOENCODING=utf-8 LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

RUN=/mmfs1/scratch/hpc/11/xiar3/expwm-runs/delta-iris-full-21531393/hydra
ANA=/mmfs1/scratch/hpc/11/xiar3/expwm-runs/analysis-21531393
OUTDIR="$ANA/filmstrips"

CONCEPTS="${CONCEPTS:-collect_wood collect_coal collect_iron place_stone}"
HORIZON="${HORIZON:-16}"
NM="${NM:-6}"

for C in $CONCEPTS; do
  echo "=== concept $C ==="
  SECONDS=0
  python /mmfs1/storage/users/xiar3/exp/ExpWM/analysis/imagine_filmstrip.py \
    --run "$RUN" \
    --rollouts "$ANA/rollouts_probe.npz" \
    --obs "$ANA/rollouts_probe_obs.npz" \
    --sae "$ANA/sae/sae.pt" \
    --features "$ANA/sae/features.json" \
    --mi-table "$ANA/mi/mi_table.json" \
    --out "$OUTDIR" \
    --concept "$C" \
    --n-moments "$NM" \
    --burn-in 4 --horizon "$HORIZON" \
    --target-block 1 --seed 0 --device cuda:0 2>&1
  echo "  $C done in ${SECONDS}s"
done
echo "=== all done ==="
