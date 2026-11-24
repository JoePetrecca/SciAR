#!/bin/bash --login
#SBATCH --job-name=SciAR_specter2_1
#SBATCH --output=/mnt/ufs18/rs/deepthought/SciAR/SciAR/7_InfoNCE/logs/SciAR_specter2_1_%j.out
#SBATCH --error=/mnt/ufs18/rs/deepthought/SciAR/SciAR/7_InfoNCE/logs/SciAR_specter2_1_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --mail-user=petrecc1@msu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/.bashrc

module purge
unset PYTHONPATH
export PYTHONNOUSERSITE=1

mamba activate SciARcuda

set -euo pipefail

cd /mnt/ufs18/rs/deepthought/SciAR/SciAR

# Log GPU information
nvidia-smi

 python3 7_InfoNCE/train_infonce.py \
    "allenai/specter2_base" \
    "SciAR_specter2_1" \
    "1"
