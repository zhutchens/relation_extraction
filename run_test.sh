#! /bin/bash

TEMPERATURE="0.1"
OPENAI_MODEL="gpt-4o"
THRESHOLD="0.7" # threshold score for test to pass
NUM_GENERATE="5" # number of concepts, outcomes, or key terms to generate

# TEXTBOOKS = ("dsa_2214" "dsa_6114" "cs_3190")
TEXTBOOKS=("dsa_2214")

# TESTS=("concepts" "outcomes" "key_terms")
TESTS=("concepts")

# testing top three most performant sentence transformers
SENTENCE_TRANSFORMERS = ("all-mpnet-base-v2" "multi-qa-mpnet-base-dot-v1" "all-distilroberta-v1")

for textbook in TEXTBOOKS
do 
    for transformer in SENTENCE_TRANSFORMERS
    do
        for test in TESTS
        do
            # 'usage: <openai_model> <openai_model_temperature> <sentence_transformer_model> <which_textbook> <which_chapters> <what_to_test> <num_to_generate> <threshold>'
            python3 test_script.py $OPENAI_MODEL $TEMPERATURE $transformer $textbook "6,7,8,9" $test $NUM_GENERATE $THRESHOLD
        done
    done
done
