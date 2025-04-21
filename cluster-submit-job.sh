#!/bin/bash

# CLUSTER
#SBATCH --job-name=YOLO_FACE_4_KYC
#SBATCH --time=24:00:00                
#SBATCH --account=XXXXXXXXXXXXXXXXXXXXX
#SBATCH --partition=gpu                 
#SBATCH --qos=default-gpu
#SBATCH --gres=gpu:1                  
#SBATCH --output=sbatchlogs/%j.out 

# ENVIRONMENT
source ~/.bashrc
conda activate deepleaning-training-cluster

# RUN-JOB
# papermill finetune.ipynb finetune-output.ipynb
bash cluster-train-yolo.sh