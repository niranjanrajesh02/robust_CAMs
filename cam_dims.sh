#!/bin/bash
#SBATCH -p gpu_a100_8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 20G
#SBATCH -t 0-23:59 # time (D-HH:MM)
#SBATCH --job-name="robCAM_acts"
#SBATCH -o cam_out.log
#SBATCH -e cam_err.log
#SBATCH --gres=gpu:1


srun python /home/venkat/niranjan/robust_CAMs/class_acts.py --dataset imagenet --arch resnet --model_type standard --data_split train --task acts
srun python /home/venkat/niranjan/robust_CAMs/class_acts.py --dataset imagenet --arch resnet --model_type standard --data_split train --task dims

