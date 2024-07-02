#!/bin/bash
#SBATCH -p bigcompute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 100G
#SBATCH -t 0-23:59 # time (D-HH:MM)
#SBATCH --job-name="robCAM_acts"
#SBATCH -o cam_out.log
#SBATCH -e cam_err.log
#SBATCH --cpus-per-task=100


srun python /home/venkat/niranjan/robust_CAMs/class_acts.py --dataset imagenet --arch resnet --model_type standard --task dims
srun python /home/venkat/niranjan/robust_CAMs/class_acts.py --dataset imagenet --arch resnet --model_type adv_trained --task dims