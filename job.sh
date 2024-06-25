#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 512M
#SBATCH -t 0-23:59 # time (D-HH:MM)
#SBATCH --job-name="robCAM_train"
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --gres=gpu:1
#SBATCH --mail-user=niranjanrajesh02@gmail.com
#SBATCH --mail-type=ALL
spack load cuda@11.8.0%gcc@11.2.0
srun ./E0.py
