from pathlib import Path
import subprocess

MODELS = {
    "bert": "google-bert/bert-base-uncased",
    "scibert": "allenai/scibert_scivocab_uncased",
    "specter": "allenai/specter",
    "scincl": "malteos/scincl",
    "specter2": "allenai/specter2_base",
    "roberta": "FacebookAI/roberta-base",
    "simcse": "princeton-nlp/sup-simcse-bert-base-uncased",
}

RUNS_PER_MODEL = 5
PROJECT_DIR = Path("/mnt/ufs18/rs/deepthought/SciAR/SciAR")
SBATCH_DIR = PROJECT_DIR / "7_InfoNCE" / "sbatch"
LOG_DIR = PROJECT_DIR / "7_InfoNCE" / "logs"

SBATCH_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

for short_name, model_name in MODELS.items():
    for run_number in range(RUNS_PER_MODEL):
        run_name = f"SciAR_{short_name}_{run_number}"
        job_path = SBATCH_DIR / f"{run_name}.sbatch"

        job_path.write_text(f"""\
#!/bin/bash --login
#SBATCH --job-name={run_name}
#SBATCH --output={LOG_DIR}/{run_name}_%j.out
#SBATCH --error={LOG_DIR}/{run_name}_%j.err
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

cd {PROJECT_DIR}

# Log GPU information
nvidia-smi

 python3 7_InfoNCE/train_infonce.py \\
    "{model_name}" \\
    "{run_name}" \\
    "{run_number}"
""")

        print(f"Submitting {job_path}")
        subprocess.run(["sbatch", str(job_path)], check=True)
