#!/bin/bash

#SBATCH --job-name=fromage
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=4
#SBATCH --partition=mid
#SBATCH --gres=gpu:tesla_t4:1    
#SBATCH --time=1-0:0:0   
#SBATCH --output=logs/fromage-%j.out
#SBATCH --mem=60G
# Load Anaconda
echo "======================="
echo "Loading Anaconda Module..."
#module load anaconda/2.7
module load cuda/11.7.1
module load cudnn/8.2.0/cuda-11.X

export TOKENIZERS_PARALLELISM=true
python train.py --config-name "train-tied.yaml"

# sbatch fromage_train.sh
