#!/bin/bash
set -e

# how many samples to generate from each topic (each domain has 5 topics)
# Since in total, we have 5 topics, we'll have 50 samples per topic, totalling 250 examples
TARGET_SAMPLES=50

# domain to run the experiment for. Can either be "news" or "conv"
DATASET_DOMAIN="conv"

ORIGINAL_SAMPLES=300


# Add other models that you'd like to try in this list
# --> Note that you need to create configs/api_keys for each of the apis 
# that you'd like to use.
for model in gpt-3.5-turbo-0125 gpt-4o-2024-05-13; do
    CONFIG_FILEPATH=./configs/api_keys/openai.txt
    echo "Config filepath path is set to $CONFIG_FILEPATH"

    for group_size in 2 3 4 5 10; do
        echo "============================================================="
        echo "    Generating summaries for combinations of $group_size     "
        echo "============================================================="
        for topic in 1 2 3 4 5; do
            python src/generate_summaries.py \
                --input_filepath ./data/SummHay_$DATASET_DOMAIN/preprocessed/adversarial_subtopic/combinations-$group_size/topic_$DATASET_DOMAIN${topic}__$ORIGINAL_SAMPLES.json \
                --output_filepath ./outputs_$DATASET_DOMAIN/generate_summaries/results_adversarial_subtopic/subtopic/SummHay__combinations-$group_size/$model/topic_$DATASET_DOMAIN${topic}__$ORIGINAL_SAMPLES.json \
                --config_filepath $CONFIG_FILEPATH \
                --prompt_filepath ./configs/prompts/generation_$DATASET_DOMAIN/subtopic.txt \
                --range 0,$TARGET_SAMPLES \
                --temperature 1 \
                --top_p 0.9 \
                --model $model \
                --conversation
        done
    done
done


