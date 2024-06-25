#!/bin/bash
#SBATCH -p gpu_v100_2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 20G
#SBATCH -t 0-23:59 # time (D-HH:MM)
#SBATCH --job-name="robCAM_train"
#SBATCH -o out.log
#SBATCH -e err.log
#SBATCH --gres=gpu:1
#SBATCH --mail-user=niranjanrajesh02@gmail.com
#SBATCH --mail-type=ALL
srun python /home/venkat/niranjan/robust_CAMs/E0.py