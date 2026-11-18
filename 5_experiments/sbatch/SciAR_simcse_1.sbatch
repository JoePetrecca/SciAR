#!/bin/bash --login
#SBATCH --job-name=SciAR_simcse_1
#SBATCH --output=/mnt/ufs18/rs/deepthought/SciAR/SciAR/5_experiments/logs/SciAR_simcse_1_%j.out
#SBATCH --error=/mnt/ufs18/rs/deepthought/SciAR/SciAR/5_experiments/logs/SciAR_simcse_1_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --mail-user=petrecc1@msu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

cd /mnt/ufs18/rs/deepthought/SciAR/SciAR
source ~/.bashrc
mamba activate SciARvenv

python3 5_experiments/train_lora.py \
    "princeton-nlp/sup-simcse-bert-base-uncased" \
    "SciAR_simcse_1" \
    "1"
