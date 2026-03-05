#!/bin/bash
#SBATCH --job-name=dino_td2
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

source ~/.bashrc
conda activate nucleole
cd ~/SingleCellEmb

python main.py --todo Todo_List/td3 --train --wandb