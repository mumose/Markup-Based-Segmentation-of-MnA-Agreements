#!/bin/bash

#SBATCH --job-name=3-50pct_training_curves-run2
#SBATCH --open-mode=append
#SBATCH --output=/scratch/str8775/conda_env/maud/logs/training_curves/50pct/slurm/%x_%j.out
#SBATCH --error=/scratch/str8775/conda_env/maud/logs/training_curves/50pct/slurm/%x_%j.err
#SBATCH --export=ALL
#SBATCH --time=7:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

export ENV=$1

singularity exec --nv --bind /scratch --overlay /scratch/str8775/conda_env/maud/maud_env${ENV}.ext3:ro /scratch/str8775/conda_env/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif /bin/bash -c "
source /ext3/env.sh
conda activate maud
cd /scratch/str8775/conda_env/maud/Markup-Based-Segmentation-of-MnA-Agreements
python3 train.py --config=./configs/config_training_curves50pct.yaml
"
