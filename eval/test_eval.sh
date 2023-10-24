#!/bin/bash

#SBATCH --job-name=fromage
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=4
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --time=1-0:0:0   
#SBATCH --output=eval_logs/test_eval-%j.out
#SBATCH --mem=48G
# Load Anaconda
echo "======================="
echo "Loading Anaconda Module..."
#module load anaconda/2.7
module load cuda/11.8.0
module load cudnn/8.2.0/cuda-11.X

export TOKENIZERS_PARALLELISM=true

python -u VQAEval.py 1

# sbatch fromage_train.sh
