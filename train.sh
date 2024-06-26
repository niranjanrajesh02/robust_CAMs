#!/bin/bash
#SBATCH -p gpu_v100_2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 20G
#SBATCH -t 1-23:59 # time (D-HH:MM)
#SBATCH --job-name="robCAM_train"
#SBATCH -o out.log
#SBATCH -e err.log
#SBATCH --gres=gpu:1
srun python /home/venkat/niranjan/robust_CAMs/train.py