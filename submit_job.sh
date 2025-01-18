#! /bin/bash

#SBATCH --job-name="RAG_Eval"
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --mem=32GB

sbatch run_test.sh