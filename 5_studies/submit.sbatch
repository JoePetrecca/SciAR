#!/bin/bash --login
#SBATCH --job-name=sciar
#SBATCH --partition=general-long-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=a100:1
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-14%2
#SBATCH --output=5_studies/logs/%x-%A_%a.out
#SBATCH --error=5_studies/logs/%x-%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
# Activate your CUDA-enabled environment before submission, or set SCIAR_PYTHON.
PYTHON="${SCIAR_PYTHON:-python}"
DATA="${SCIAR_DATA:-5_studies/data/fixed_100k}"
REPORT="${SCIAR_PREFLIGHT:-5_studies/verification/gpu.json}"
"$PYTHON" 5_studies/verify.py --check-report --report "$REPORT"
mkdir -p 5_studies/logs
nvidia-smi
# Sample only the GPU(s) allocated by SLURM, not other users' devices on this node.
nvidia-smi --id="${CUDA_VISIBLE_DEVICES:?SLURM did not expose a GPU}" \
  --query-gpu=timestamp,uuid,name,utilization.gpu,memory.used,memory.total \
  --format=csv --loop=30 > "5_studies/logs/gpu-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
ARGS=(--task-id "${SLURM_ARRAY_TASK_ID:?}" --data "$DATA")
if [[ "${SCIAR_RESUME:-0}" == 1 ]]; then ARGS+=(--resume); fi
if [[ -n "${SCIAR_MICROBATCH:-}" ]]; then ARGS+=(--microbatch-size "$SCIAR_MICROBATCH"); fi
if [[ "${SCIAR_GRADIENT_CHECKPOINTING:-0}" == 1 ]]; then ARGS+=(--gradient-checkpointing); fi
srun --cpus-per-task="${SLURM_CPUS_PER_TASK}" "$PYTHON" -u 5_studies/train.py "${ARGS[@]}"
