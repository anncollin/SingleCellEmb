#!/bin/bash
#SBATCH --job-name=td2
#SBATCH --gres=gpu:TeslaA10:1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

source ~/.bashrc
conda activate nucleole
cd ~/SingleCellEmb

python main.py --todo Todo_List/td2 --train --wandb