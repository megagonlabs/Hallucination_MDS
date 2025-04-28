#!/bin/bash
set -e
# how many samples to generate from each topic (each domain has 5 topics)
# Since in total, we have 5 topics, we'll have 100 samples per topic, totalling 500 examples
TARGET_SAMPLES=100

# domain to run the experiment for. Can either be "news" or "conv"
DATASET_DOMAIN="conv"


ORIGINAL_SAMPLES=300
# Add other models that you'd like to try in this list
# --> Note that you need to create configs/api_keys for each of the apis 
# that you'd like to use.
for model in gpt-3.5-turbo-0125 gpt-4o-2024-05-13; do # accounts/fireworks/models/qwen2-72b-instruct # 
    if [[ $model == gpt-* ]]; then
        echo "Using OpenAI API"
        CONFIG_FILEPATH=./configs/api_keys/openai.txt
    else
        CONFIG_FILEPATH=./configs/api_keys/fireworks.json

    fi
    echo "Config filepath path is set to $CONFIG_FILEPATH"

    for group_size in 2 3 4 5 10; do
        for topic in 1 2 3 4 5; do
            echo "Generating summaries using: $model  |  $topic   | group_size"

            python src/generate_summaries.py \
                --input_filepath ./data/SummHay_$DATASET_DOMAIN/preprocessed/some_shared/combinations-$group_size/topic_$DATASET_DOMAIN${topic}__$ORIGINAL_SAMPLES.json \
                --output_filepath ./outputs_$DATASET_DOMAIN/generate_summaries/results_some_shared/subtopic/SummHay__combinations-$group_size/$model/topic_$DATASET_DOMAIN${topic}__$ORIGINAL_SAMPLES.json \
                --config_filepath $CONFIG_FILEPATH \
                --prompt_filepath ./configs/prompts/generation_$DATASET_DOMAIN/subtopic.txt \
                --range 0,$TARGET_SAMPLES \
                --temperature 1 \
                --top_p 0.9 \
                --model $model \
                --conversation
                
            python src/generate_summaries.py \
                --input_filepath ./data/SummHay_$DATASET_DOMAIN/preprocessed/some_shared/combinations-$group_size/topic_$DATASET_DOMAIN${topic}__$ORIGINAL_SAMPLES.json \
                --output_filepath ./outputs_$DATASET_DOMAIN/generate_summaries/results_some_shared/subtopic_trustworthy/SummHay__combinations-$group_size/$model/topic_$DATASET_DOMAIN${topic}__$ORIGINAL_SAMPLES.json \
                --config_filepath $CONFIG_FILEPATH \
                --prompt_filepath ./configs/prompts/generation_$DATASET_DOMAIN/subtopic_trustworthy.txt \
                --range 0,$TARGET_SAMPLES \
                --temperature 1 \
                --top_p 0.9 \
                --model $model \
                --conversation

        done
    done
done