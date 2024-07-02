#!/bin/bash
#SBATCH -p gpu_a100_2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 20G
#SBATCH -t 0-23:59 # time (D-HH:MM)
#SBATCH --job-name="robCAM_eval"
#SBATCH -o ev_out.log
#SBATCH -e ev_err.log
#SBATCH --gres=gpu:1

srun python /home/venkat/niranjan/robust_CAMs/evaluate_art.py --model_type adv_trained --dataset imagenet --eps 3
srun python /home/venkat/niranjan/robust_CAMs/evaluate_art.py --model_type adv_trained --dataset imagenet --eps 2
srun python /home/venkat/niranjan/robust_CAMs/evaluate_art.py --model_type adv_trained --dataset imagenet --eps 0




