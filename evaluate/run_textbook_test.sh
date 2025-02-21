#! /bin/bash

module load anaconda3
conda activate env # run conda env create -f environment.yml first if this causes an error!

THRESHOLD="0.7" # threshold score for test to pass
NUM_GENERATE="10" # number of concepts, outcomes, or key terms to generate

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
                    srun python3 evaluate.py $llm $transformer $retriever $course $test $NUM_GENERATE $THRESHOLD
                    echo "Submitted job python3 evaluate.py $llm $transformer $retriever $course $test $NUM_GENERATE $THRESHOLD"
                done
            done
        done
    done
done
