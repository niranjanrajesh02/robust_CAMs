#!/bin/bash
#SBATCH -p gpu_a100_8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 20G
#SBATCH -t 0-23:59 # time (D-HH:MM)
#SBATCH --job-name="robCAM2_acts"
#SBATCH -o cam2_out.log
#SBATCH -e cam2_err.log
#SBATCH --gres=gpu:1


srun python /home/venkat/niranjan/robust_CAMs/class_acts.py --dataset imagenet --arch resnet --model_type adv_trained --data_split train --task acts