#!/bin/bash
#SBATCH -p gpu_v100_2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 20G
#SBATCH -t 1-23:59 # time (D-HH:MM)
#SBATCH --job-name="robCAM_acts"
#SBATCH -o cam_out.log
#SBATCH -e cam_err.log
#SBATCH --gres=gpu:1
srun python /home/venkat/niranjan/robust_CAMs/class_acts.py --model_type standard --data_split test
srun python /home/venkat/niranjan/robust_CAMs/class_acts.py --model_type adv_trained --data_split test
srun python /home/venkat/niranjan/robust_CAMs/class_acts.py --model_type robust --data_split test
