#!/bin/bash
set -e
EVAL_MODEL=gpt-4o-mini-2024-07-18
for prompt_type in subtopic subtopic_trustworthy; do 
    for dataset_type in results_some_shared results_adversarial_subtopic; do
        if [[ $dataset_type == results_adversarial_subtopic ]]; then
            SUBSAMPLE=\'25,50\'
        else
            SUBSAMPLE=\'50,100\'
        fi
        echo "------------------- USING $SUBSAMPLE -------------------"
        for group_size in 2 3 4 5 10; do
            for SUMMARIZER_MODEL in accounts/fireworks/models/llama-v3-70b-instruct accounts/fireworks/models/mixtral-8x7b-instruct-hf gpt-3.5-turbo-0125; do 
                echo "============================================================================================="
                echo "Evaluating summaries for topic_news${topic} | combinations-$group_size | $SUMMARIZER_MODEL   "
                echo "-> Using prompt_type: $prompt_type"
                echo "============================================================================================="
                python src/run_evals.py \
                    ++combinations=$group_size \
                    +requests=single_requests \
                    ++requests._target_=\'evaluators.SingleRequestBatch\' \
                    ++requests.eval_kwargs.model=$EVAL_MODEL \
                    summarizer_model=$SUMMARIZER_MODEL subsample=$SUBSAMPLE \
                    dataset_type=$dataset_type prompt_type=$prompt_type \
                    setup.config_filepath=configs/api_keys/openai.txt \
                    ++setup.output_filepath='outputs/${hydra:job.name}-single-request/${requests.eval_kwargs.model}/${dataset_type}/${prompt_type}/SummHay__combinations-${combinations}/${summarizer_model}'
            done
        done
    done
done
