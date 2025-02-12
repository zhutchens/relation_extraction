#! /bin/bash

TEMPERATURE="0.1"
THRESHOLD="0.7" # threshold score for test to pass
NUM_GENERATE="10" # number of concepts, outcomes, or key terms to generate

# TEXTBOOKS = ("dsa_2214" "dsa_6114" "cs_3190")
TEXTBOOKS=("dsa_2214")

# TESTS=("concepts" "outcomes" "key_terms")
TESTS=("concepts" "outcomes")

# testing top three most performant sentence transformers
SENTENCE_TRANSFORMERS=("msmarco-distilbert-base-tas-b" "msmarco-MiniLM-L-6-v3" "msmarco-MiniLM-L-12-v3" "msmarco-distilbert-base-v4")
LANGUAGE_MODELS=("gpt-4o" "gpt-4o-mini")

for textbook in "${TEXTBOOKS[@]}"
do
    for test in "${TESTS[@]}"
    do
        for llm in "${LANGUAGE_MODELS[@]}"
        do
            for transformer in "${SENTENCE_TRANSFORMERS[@]}"
            do
                python evaluate.py $llm $TEMPERATURE $transformer $textbook $test $NUM_GENERATE $THRESHOLD
            done
        done
    done
done
