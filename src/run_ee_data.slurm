#!/bin/bash

#SBATCH --job-name=Language_model
#SBATCH --partition=a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-type=end
#SBATCH --mail-user=Febraury25th@sjtu.edu.cn
#SBATCH --output=./slurm_log/%j.out
#SBATCH --error=./slurm_log/%j.err
#SBATCH --gres=gpu:1
python ee_data.py