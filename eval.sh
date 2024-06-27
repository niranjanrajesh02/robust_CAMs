#!/bin/bash
#SBATCH -p gpu_v100_2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 20G
#SBATCH -t 0-10:59 # time (D-HH:MM)
#SBATCH --job-name="robCAM_eval"
#SBATCH -o ev_out.log
#SBATCH -e ev_err.log
#SBATCH --gres=gpu:1
srun python /home/venkat/niranjan/robust_CAMs/evaluate.py --model_type standard --eps 0
srun python /home/venkat/niranjan/robust_CAMs/evaluate.py --model_type standard --eps 0.1
srun python /home/venkat/niranjan/robust_CAMs/evaluate.py --model_type adv_trained --eps 
srun python /home/venkat/niranjan/robust_CAMs/evaluate.py --model_type adv_trained --eps 0.1
srun python /home/venkat/niranjan/robust_CAMs/evaluate.py --model_type robust --eps 
srun python /home/venkat/niranjan/robust_CAMs/evaluate.py --model_type robust --eps 0.1
