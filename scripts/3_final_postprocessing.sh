#!/bin/bash

set -e
EVAL_TYPE=multi-request
EVAL_MODEL=gpt-4o-mini-2024-07-18

DATASET_DOMAIN="conv"

for SUMMARIZER_MODEL in  gemini-1.5-flash ; do  #  accounts/fireworks/models/llama-v3p1-70b-instruct accounts/fireworks/models/qwen2-72b-instruct gpt-4o-2024-05-13 gpt-3.5-turbo-0125
    for GROUP_SIZE in 2 3 4 5 10; do 
        for prompt_type in subtopic subtopic_trustworthy; do 
            echo "============================================================================================="
            echo "       Creating canonic result files for combinations-$GROUP_SIZE | $SUMMARIZER_MODEL        "
            echo "============================================================================================="
            BASENAME=$EVAL_MODEL/results_some_shared/$prompt_type/SummHay__combinations-${GROUP_SIZE}/$SUMMARIZER_MODEL
            python src/run_final_postprocessing.py \
                --input_dir ./outputs_$DATASET_DOMAIN/run_evals_postprocessing-$EVAL_TYPE/$BASENAME \
                --output_dir ./outputs_$DATASET_DOMAIN/run_final_postprocessing-$EVAL_TYPE/$BASENAME
        done
    done
done