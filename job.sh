#!/bin/bash -l
#SBATCH -p icelake
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=00:15:00
#SBATCH --mem=2000MB

# Notification configuration
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lauren.taylor@adelaide.edu.au

module load Anaconda3/2023.03 
conda activate jsrlgs-env
python parallel_training.py --test
conda deactivate
