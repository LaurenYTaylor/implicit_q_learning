#!/bin/bash -l
#SBATCH -p a100cpu
#SBATCH -N 1
#SBATCH -n 3
#SBATCH --time=00:20:00
#SBATCH --mem-per-cpu=2000MB

# Notification configuration
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lauren.taylor@adelaide.edu.au

module load Anaconda3/2023.03 
conda activate jsrlgs-env
python parallel_training.py --test
conda deactivate
