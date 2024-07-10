#!/bin/bash
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 20G
#SBATCH -t 2-23:59 # time (D-HH:MM)
#SBATCH --job-name="robCAM3_eval"
#SBATCH -o ev3_out.log
#SBATCH -e ev3_err.log
#SBATCH --cpus-per-task=50


srun python /home/venkat/niranjan/robust_CAMs/evaluate.py --model_arch densenet161 --model_type standard 
srun python /home/venkat/niranjan/robust_CAMs/evaluate.py --model_arch densenet161 --model_type standard --adv_evaluate True --l_constraint l2
srun python /home/venkat/niranjan/robust_CAMs/evaluate.py --model_arch densenet161 --model_type standard --adv_evaluate True --l_constraint linf 




