#! /bin/bash

#SBATCH --job-name=rag_eval
#SBATCH --gres=gpu:1
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --mem=32GB

module load anaconda3
conda activate env # run conda env create -f environment.yml first if this causes an error!

THRESHOLD="0.7" # threshold score for test to pass
NUM_GENERATE="10" # number of concepts or outcomes to generate

COURSES=("cs2")

TESTS=("concepts" "outcomes")
RETRIEVERS=("transformer" "vectordb")

SENTENCE_TRANSFORMERS=("msmarco-distilbert-base-tas-b" "msmarco-MiniLM-L-6-v3" "msmarco-MiniLM-L-12-v3" "msmarco-distilbert-base-v4")
LANGUAGE_MODELS=("gpt-4o" "gpt-4o-mini")

for course in "${COURSES[@]}"
do
    for test in "${TESTS[@]}"
    do
        for retriever in "${RETRIEVERS}"
        do
            for llm in "${LANGUAGE_MODELS[@]}"
            do
                for transformer in "${SENTENCE_TRANSFORMERS[@]}"
                do
                    srun python3 evaluate_textbook_pipeline.py $llm $transformer $retriever $course $test $NUM_GENERATE $THRESHOLD
                done
            done
        done
    done
done