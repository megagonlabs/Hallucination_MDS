#!/bin/bash
set -e
EVAL_MODEL=gpt-4o-mini-2024-07-18
for prompt_type in subtopic_trustworthy; do
    echo "============================================================================================="
    echo "============================================================================================="
    echo "============================================================================================="
    echo "============================================================================================="
    for group_size in 10; do 
        for SUMMARIZER_MODEL in gemini-1.5-flash; do #  accounts/fireworks/models/qwen2-72b-instruct; do 
            python src/run_evals_postprocessing.py \
                ++combinations=$group_size \
                +requests=open_ai_batch_kwargs \
                ++setup.batch=true \
                ++eval_model=$EVAL_MODEL \
                summarizer_model=$SUMMARIZER_MODEL subsample=\'0,100\' \
                dataset_type=\'results_some_shared\' prompt_type=$prompt_type \
                setup.config_filepath=configs/api_keys/openai.txt \
                ++setup.evaluations_filepath="outputs_conv/run_evals-multi-request/${EVAL_MODEL}/\${dataset_type}/\${prompt_type}/SummHay__combinations-\${combinations}/\${summarizer_model}" \
                ++setup.output_filepath="outputs_conv/\${hydra:job.name}-multi-request/${EVAL_MODEL}/\${dataset_type}/\${prompt_type}/SummHay__combinations-\${combinations}/\${summarizer_model}"
        done
    done
done