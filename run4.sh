#!/bin/bash
#SBATCH --job-name=td4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:TeslaA100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

source ~/.bashrc
conda activate nucleole
cd ~/SingleCellEmb

python main.py --todo Todo_List/td4 --wandb