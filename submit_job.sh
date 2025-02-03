#! /bin/bash

#SBATCH --job-name=rag_eval
#SBATCH --gres=gpu:1
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --mem=32GB

bash run_test.sh