#!/bin/bash
#SBATCH --job-name=3CNN        # Job name
#SBATCH --output=output_%j.log            # Output file (%j = job ID)
#SBATCH --error=error_%j.log              # Error file (%j = job ID)
#SBATCH --time=24:00:00                   # Time limit (hh:mm:ss)
#SBATCH --mem=128G                         # Memory per node
#SBATCH --gres=gpu:4                   # Request 2 GPU




# Activate virtual environment (if needed)
source /home/amyn/AIProject/bin/activate # Change "myenv" to your virtual environment name

# Run your Python script
python 3CNN_fusion_3class_attention.py
