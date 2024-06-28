#!/bin/bash
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 20G
#SBATCH -t 2-23:59 # time (D-HH:MM)
#SBATCH --job-name="robCAM_eval"
#SBATCH -o ev_out.log
#SBATCH -e ev_err.log
#SBATCH --cpus-per-task=100

srun python /home/venkat/niranjan/robust_CAMs/evaluate.py --model_type standard --dataset imagenet --eps 0
srun python /home/venkat/niranjan/robust_CAMs/evaluate.py --model_type standard --dataset imagenet --eps 3
