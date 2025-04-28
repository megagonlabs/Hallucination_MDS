#!/bin/bash
set -e
DATASET_DOMAIN=$1
# DATASET_DOMAIN="news"
# DATASET_DOMAIN="conv"
echo "| ============================================================================================ |"
echo "|                  Creating adversarial datasets for domain '$DATASET_DOMAIN'                  |"
echo "| ============================================================================================ |"
for j in 2 3 4 5 10; do
    python src/create_dataset_adversarial.py \
        --input_dir ./data/SummHay_$DATASET_DOMAIN/preprocessed/some_shared/combinations-$j \
        --output_dir ./data/SummHay_$DATASET_DOMAIN/preprocessed/adversarial_subtopic/combinations-$j \
        --logging_filepath ./outputs_$DATASET_DOMAIN/create_dataset_adversarial/SummHay_$DATASET_DOMAIN-combinations-$j.log \
        --seed 89172
done
