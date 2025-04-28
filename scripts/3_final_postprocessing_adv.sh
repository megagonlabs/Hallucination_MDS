#!/bin/bash

set -e

for DATASET_DOMAIN in conv news; do
    for SUMMARIZER_MODEL in accounts/fireworks/models/qwen2-72b-instruct gpt-4o-2024-05-13 gpt-3.5-turbo-0125 accounts/fireworks/models/llama-v3p1-70b-instruct gemini-1.5-flash; do 
        for GROUP_SIZE in 2 3 4 5 10; do
            echo "============================================================================================="
            echo "       Creating canonic result files for combinations-$GROUP_SIZE | $SUMMARIZER_MODEL        "
            echo "============================================================================================="
            python src/run_final_postprocessing.py \
                --adversarial \
                --input_dir outputs_$DATASET_DOMAIN/generate_summaries/results_adversarial_subtopic/subtopic/SummHay__combinations-$GROUP_SIZE/$SUMMARIZER_MODEL \
                --output_dir ./outputs_$DATASET_DOMAIN/run_final_postprocessing-adv/results_adversarial_subtopic/subtopic/SummHay__combinations-${GROUP_SIZE}/$SUMMARIZER_MODEL
        done
    done
done