#!/bin/bash
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 20G
#SBATCH -t 3-23:59 # time (D-HH:MM)
#SBATCH --job-name="robCAM_eval2"
#SBATCH -o ev2_out.log
#SBATCH -e ev2_err.log
#SBATCH --cpus-per-task=50

srun python /home/venkat/niranjan/robust_CAMs/evaluate_art.py --model_type adv_trained --dataset imagenet --eps 0
srun python /home/venkat/niranjan/robust_CAMs/evaluate_art.py --model_type adv_trained --dataset imagenet --eps 2
srun python /home/venkat/niranjan/robust_CAMs/evaluate_art.py --model_type adv_trained --dataset imagenet --eps 3

