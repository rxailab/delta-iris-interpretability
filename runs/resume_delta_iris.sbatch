#!/bin/bash
#SBATCH --job-name=delta-iris-resume
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=gpu11,gpu12
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=5-12:00:00
#SBATCH --output=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/delta-iris-resume-%j.out
#SBATCH --error=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/delta-iris-resume-%j.err

# Resume an earlier Δ-IRIS run from its on-disk checkpoint.
# Usage:
#   sbatch --export=ALL,RESUME_DIR=/mmfs1/scratch/.../delta-iris-full-XXXX/hydra runs/resume_delta_iris.sbatch

set -euo pipefail

if [[ -z "${RESUME_DIR:-}" ]]; then
  echo "ERROR: set RESUME_DIR=/path/to/old-run/hydra (i.e. the dir that contains checkpoints/)" >&2
  exit 2
fi
if [[ ! -d "$RESUME_DIR/checkpoints" ]]; then
  echo "ERROR: $RESUME_DIR/checkpoints not found — wrong RESUME_DIR?" >&2
  exit 2
fi

source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate /mmfs1/storage/users/xiar3/exp/ExpWM/envs/delta-iris-env

echo "=== resuming Δ-IRIS run on $SLURMD_NODENAME (job $SLURM_JOB_ID) ==="
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
python -c "import torch; t=torch.tensor([1.0]).cuda(); print('cuda OK')"

echo "RESUME_DIR=$RESUME_DIR"
df -h "$RESUME_DIR" 2>&1 | head -3
python - <<PY
import torch
e = torch.load("$RESUME_DIR/checkpoints/epoch.pt")
print(f"resuming from epoch {e} (next training epoch = {e+1})")
PY

cd "$RESUME_DIR"

SECONDS=0
python /mmfs1/storage/users/xiar3/exp/ExpWM/delta-iris/src/main.py \
  hydra.run.dir=. \
  hydra.output_subdir=null \
  params.common.resume=true \
  params.common.device=cuda:0 \
  wandb.mode=disabled 2>&1
echo "=== resume exit: $? after ${SECONDS}s ==="
