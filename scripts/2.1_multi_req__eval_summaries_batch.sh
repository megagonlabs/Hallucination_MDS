#!/bin/bash
set -e
EVAL_MODEL=gpt-4o-mini-2024-07-18
for prompt_type in subtopic subtopic_trustworthy; do 
    echo "============================================================================================="
    echo "============================================================================================="
    echo "============================================================================================="
    echo "============================================================================================="
    for SUMMARIZER_MODEL in gemini-1.5-flash; do # gpt-4o-2024-05-13 gpt-3.5-turbo-0125 accounts/fireworks/models/llama-v3p1-70b-instruct accounts/fireworks/models/qwen2-72b-instruct
        for group_size in 2 3 4 5 10; do 
            python src/run_evals.py \
                ++combinations=$group_size \
                +requests=multi_requests \
                ++requests._target_=\'evaluators.MultiRequestBatch\' \
                ++requests.eval_kwargs.model=$EVAL_MODEL \
                summarizer_model=$SUMMARIZER_MODEL subsample=\'0,100\' \
                dataset_type=\'results_some_shared\' prompt_type=$prompt_type \
                setup.config_filepath=configs/api_keys/openai.txt \
                ++setup.data_filepath='outputs_conv/generate_summaries/${dataset_type}/${prompt_type}/SummHay__combinations-${combinations}/${summarizer_model}' \
                ++setup.output_filepath='outputs_conv/${hydra:job.name}-multi-request/${requests.eval_kwargs.model}/${dataset_type}/${prompt_type}/SummHay__combinations-${combinations}/${summarizer_model}'
        done
    done
done