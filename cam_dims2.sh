#!/bin/bash
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 20G
#SBATCH -t 0-23:59 # time (D-HH:MM)
#SBATCH --job-name="robCAM2_acts"
#SBATCH -o cam2_out.log
#SBATCH -e cam2_err.log
#SBATCH --cpus-per-task=50


srun python /home/venkat/niranjan/robust_CAMs/class_acts.py --dataset imagenet --model_arch resnet50 --model_type standard --data_split train --task wc_vars

