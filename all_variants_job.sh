#!/bin/bash -l
#SBATCH -p skylake
#SBATCH -N 1
#SBATCH -n 20
#SBATCH --time=35:00:00
#SBATCH --mem-per-cpu=2000MB

# Notification configuration
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lauren.taylor@adelaide.edu.au

module load Anaconda3/2023.03 
conda activate jsrlgs-env
python parallel_training_sep_algs.py --algo jsrlgs --warm_start --at_thresh --env FlappyBird-v0 --dataset_name flappy_ppo
conda deactivate
