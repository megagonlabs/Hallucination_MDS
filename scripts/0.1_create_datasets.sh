#!/bin/bash
set -e

DATASET_DOMAIN=$1
# DATASET_DOMAIN="news"
# DATASET_DOMAIN="conv"
for j in 2 3 4 5 10; do
    python src/create_dataset.py \
        --input_dir ./data/SummHay_$DATASET_DOMAIN/raw \
        --output_dir ./data/SummHay_$DATASET_DOMAIN/preprocessed \
        --logging_filepath ./outputs/create_dataset__SummHay_$DATASET_DOMAIN-combinations-$j.log \
        --combinations $j \
        --sample_size 300 \
        --eta 2 \
        --insights_shared_some \
        --seed 17623
done
