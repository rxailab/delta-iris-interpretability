#!/bin/bash
#SBATCH --job-name=cascade-nohud
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=gpu11,gpu12
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/cascade-nohud-%j.out
#SBATCH --error=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/cascade-nohud-%j.err

set -uo pipefail
source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate /mmfs1/storage/users/xiar3/exp/ExpWM/envs/delta-iris-env
export PYTHONIOENCODING=utf-8 LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

# No-HUD agent: most direct test of the central claim --- with the HUD inventory
# removed from the frame, does lesioning the wood SAE feature now CASCADE to the
# downstream wooden-tool tier (i.e. does write begin to drive)?  Identical protocol
# to the with-HUD cascade (burn-in 8, horizon 12, n-moments 40) for direct comparison.
RUN=/mmfs1/scratch/hpc/11/xiar3/expwm-runs/delta-iris-nohud-22054506/hydra
ANA=/mmfs1/scratch/hpc/11/xiar3/expwm-runs/analysis-nohud-22054506
FOCUS="${FOCUS:-collect_wood}"
HORIZON="${HORIZON:-12}"
NMOM="${NMOM:-40}"
SUFFIX="${SUFFIX:-}"

python /mmfs1/storage/users/xiar3/exp/ExpWM/analysis/imagine_cascade.py \
  --run "$RUN" \
  --rollouts "$ANA/rollouts_probe.npz" \
  --obs "$ANA/rollouts_probe_obs.npz" \
  --sae "$ANA/sae/sae.pt" \
  --features "$ANA/sae/features.json" \
  --cavs "$ANA/cavs/cavs.npz" \
  --mi-table "$ANA/mi/mi_table.json" \
  --out "$ANA/cascade_${FOCUS}${SUFFIX}" \
  --focus "$FOCUS" \
  --burn-in 8 --horizon "$HORIZON" --n-moments "$NMOM" \
  --target-block 1 --seed 42 --device cuda:0 2>&1
echo "=== exit $? ==="
