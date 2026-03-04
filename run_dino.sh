#!/bin/bash
#SBATCH --job-name=dino
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

source ~/.bashrc
conda activate nucleole
cd ~/SingleCellEmb

srun --exclusive --gpus=1 python main.py --todo Todo_List/td1 --train --wandb &
srun --exclusive --gpus=1 python main.py --todo Todo_List/td2 --train --wandb &

wait