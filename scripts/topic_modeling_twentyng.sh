#!/bin/bash
#SBATCH --job-name=topic_modeling_twentyng
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --gpus-per-node=a40
#SBATCH --output=slurm/slurm_outputs/topic_modeling_twentyng.out
#SBATCH --error=slurm/slurm_errors/topic_modeling_twentyng.err
#SBATCH --account="overcap"
#SBATCH --partition="overcap"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --qos="short"

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
conda activate rag-cobweb
cd ~/flash/cobweb-language-embedding
export PYTHONPATH=$(pwd)

echo "Starting 20 Newsgroups topic modeling at $(date)"

DATASET=${DATASET:-20newsgroups}
TOP_N_WORDS=${TOP_N_WORDS:-15}
NUM_CLUSTERS=${NUM_CLUSTERS:-20}
MODEL_NAME=${MODEL_NAME:-all-roberta-large-v1}
DEVICE=${DEVICE:-cuda}
RUNTIME_DIR=${RUNTIME_DIR:-outputs/runtime}
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RUNTIME_LOG="${RUNTIME_DIR}/topic_modeling_twentyng_${TIMESTAMP}.json"

mkdir -p slurm/slurm_outputs slurm/slurm_errors "${RUNTIME_DIR}"

srun python benchmarks/topic_modeling/benchmark.py "${DATASET}" \
  --top-n-words "${TOP_N_WORDS}" \
  --num-clusters "${NUM_CLUSTERS}" \
  --model-name "${MODEL_NAME}" \
  --device "${DEVICE}" \
  --runtime-log "${RUNTIME_LOG}" \
  --test-hierarchical

echo "20 Newsgroups topic modeling completed at $(date)"
echo "Runtime log: ${RUNTIME_LOG}"