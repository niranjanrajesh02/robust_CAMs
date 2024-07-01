#!/bin/bash
#SBATCH -p gpu_v100_1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 20G
#SBATCH -t 2-23:59 # time (D-HH:MM)
#SBATCH --job-name="robCAM2_eval"
#SBATCH -o ev2_out.log
#SBATCH -e ev2_err.log
#SBATCH --gres=gpu:1

srun python /home/venkat/niranjan/robust_CAMs/evaluate_art.py --model_type vone_resnet --dataset imagenet --eps 0
srun python /home/venkat/niranjan/robust_CAMs/evaluate_art.py --model_type vone_resnet --dataset imagenet --eps 2
srun python /home/venkat/niranjan/robust_CAMs/evaluate_art.py --model_type vone_resnet --dataset imagenet --eps 3



