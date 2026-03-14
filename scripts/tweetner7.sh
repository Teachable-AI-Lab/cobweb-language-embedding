#!/bin/bash
#SBATCH --job-name=tweetner7_incremental
#SBATCH --time=24:00:00
#SBATCH --mem=24G
#SBATCH --gpus-per-node=a40
#SBATCH --output=slurm/slurm_outputs/tweetner7_incremental.out
#SBATCH --error=slurm/slurm_errors/tweetner7_incremental.err
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

CONFIG_PATH=${1:?"Usage: $0 <config_path>"}

echo "Starting TweetNER7 Incremental Topic Modeling at $(date)"

srun python benchmarks/topic_modeling/incremental_benchmark.py --config "$CONFIG_PATH"

echo "TweetNER7 Incremental Topic Modeling completed at $(date)"
