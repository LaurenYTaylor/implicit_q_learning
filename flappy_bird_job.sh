#!/bin/bash -l
#SBATCH -p a100cpu
#SBATCH -N 1
#SBATCH -n 15
#SBATCH --time=25:00:00
#SBATCH --mem-per-cpu=2000MB

# Notification configuration
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lauren.taylor@adelaide.edu.au

module load Anaconda3/2023.03 
conda activate jsrlgs-env
python train_flappy_bird.py
conda deactivate
