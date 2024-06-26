#!/bin/bash
#SBATCH -p gpu_v100_2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 20G
#SBATCH -t 2-23:59 # time (D-HH:MM)
#SBATCH --job-name="robCAM_adv_train"
#SBATCH -o aout.log
#SBATCH -e aerr.log
#SBATCH --gres=gpu:1
srun python /home/venkat/niranjan/robust_CAMs/train.py --adv_train