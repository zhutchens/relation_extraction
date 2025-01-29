#! /bin/bash

TEMPERATURE="0.1"
THRESHOLD="0.7" # threshold score for test to pass
NUM_GENERATE="10" # number of concepts, outcomes, or key terms to generate

# TEXTBOOKS = ("dsa_2214" "dsa_6114" "cs_3190")
TEXTBOOKS=("dsa_2214")

# TESTS=("concepts" "outcomes" "key_terms")
TESTS=("concepts" "outcomes")

# testing top three most performant sentence transformers
SENTENCE_TRANSFORMER="msmarco-distilbert-base-tas-b"

for textbook in TEXTBOOKS
do
    for test in TESTS
    do
        python3 evaluate.py $TEMPERATURE $SENTENCE_TRANSFORMER $textbook $test $NUM_GENERATE
    done
done

# for textbook in TEXTBOOKS
# do 
#     for transformer in SENTENCE_TRANSFORMERS
#     do
#         for test in TESTS
#         do
#             # usage: <llm_temperature> <sentence_transformer_model> <which_textbook> <what_to_test> <num_to_generate> <threshold>
#             python3 evaluate.py 
#         done
#     done
# done
