#!/bin/bash
#SBATCH --job-name=retrieval_qqp_2k
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --gpus-per-node=a40
#SBATCH --output=slurm/slurm_outputs/retrieval_qqp_2k.out
#SBATCH --error=slurm/slurm_errors/retrieval_qqp_2k.err
#SBATCH --account="overcap"
#SBATCH --partition="overcap"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --qos="short"

set -euo pipefail

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
conda activate rag-cobweb
cd ~/flash/cobweb-language-embedding
export PYTHONPATH=$(pwd)

echo "Starting retrieval QQP 2k at $(date)"

DATASET=${DATASET:-qqp}
MODEL_NAME=${MODEL_NAME:-all-roberta-large-v1}
SUBSET_SIZE=${SUBSET_SIZE:-2000}
TARGET_SIZE=${TARGET_SIZE:-500}
TOP_K=${TOP_K:-3}
TARGET_DIM=${TARGET_DIM:-256}
RUN_LOG_DIR=${RUN_LOG_DIR:-outputs/runtime}
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RUN_LOG="${RUN_LOG_DIR}/retrieval_${DATASET}_2k_${TIMESTAMP}.log"

mkdir -p slurm/slurm_outputs slurm/slurm_errors "${RUN_LOG_DIR}"

srun python benchmarks/retrieval/benchmark.py \
  --dataset "${DATASET}" \
  --model_name "${MODEL_NAME}" \
  --subset_size "${SUBSET_SIZE}" \
  --target_size "${TARGET_SIZE}" \
  --top_k "${TOP_K}" \
  --method all \
  --target_dim "${TARGET_DIM}" \
  > "${RUN_LOG}"

echo "Retrieval QQP 2k completed at $(date)"
echo "Run log: ${RUN_LOG}"