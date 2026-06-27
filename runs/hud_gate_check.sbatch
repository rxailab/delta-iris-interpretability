#!/bin/bash
#SBATCH --job-name=hud-gate
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/hud-gate-%j.out
#SBATCH --error=/mmfs1/storage/users/xiar3/exp/ExpWM/runs/hud-gate-%j.err

set -uo pipefail
source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate /mmfs1/storage/users/xiar3/exp/ExpWM/envs/delta-iris-env
export PYTHONIOENCODING=utf-8 LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
export PYTHONPATH=/mmfs1/storage/users/xiar3/exp/ExpWM/delta-iris/src:${PYTHONPATH:-}

A=/mmfs1/storage/users/xiar3/exp/ExpWM/analysis
S=/mmfs1/scratch/hpc/11/xiar3/expwm-runs
declare -A RUN=( [orig]=delta-iris-full-21531393 [seed1]=delta-iris-seed1 [seed2]=delta-iris-seed2 )
declare -A ANA=( [orig]=analysis-21531393 [seed1]=analysis-seed1 [seed2]=analysis-seed2 )
declare -A GATE=( [orig]=187 [seed1]=255 [seed2]=996 )

for k in orig seed1 seed2; do
  echo; echo "################ $k  gate=f${GATE[$k]} ################"
  python $A/hud_gate_check.py \
    --run "$S/${RUN[$k]}/hydra" \
    --rollouts "$S/${ANA[$k]}/rollouts_probe.npz" \
    --obs "$S/${ANA[$k]}/rollouts_probe_obs.npz" \
    --sae "$S/${ANA[$k]}/sae/sae.pt" \
    --features "$S/${ANA[$k]}/sae/features.json" \
    --out "$S/${ANA[$k]}/cascade_gate" \
    --feat-id ${GATE[$k]} --focus collect_wood \
    --burn-in 8 --horizon 12 --n-moments 40 \
    --target-block 1 --seed 42 --device cuda:0 2>&1 || echo "$k FAILED"
done
echo "=== ALL DONE ==="
