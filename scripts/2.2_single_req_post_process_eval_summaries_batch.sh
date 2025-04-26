#!/bin/bash
set -e
EVAL_MODEL=gpt-4o-mini-2024-07-18

for SUMMARIZER_MODEL in accounts/fireworks/models/mixtral-8x7b-instruct-hf; do
    for prompt_type in subtopic subtopic_trustworthy; do 
        for dataset_type in results_some_shared results_adversarial_subtopic; do
            if [[ $dataset_type == results_adversarial_subtopic ]]; then
                SUBSAMPLE=\'0,25\'
            else
                SUBSAMPLE=\'0,50\'
            fi
            echo "------------------- USING $SUBSAMPLE -------------------"
            for group_size in 2 3 4 5; do
                echo "============================================================================================="
                echo "Evaluating summaries for topic_news${topic} | combinations-$group_size | $SUMMARIZER_MODEL   "
                echo "-> Using prompt_type: $prompt_type"
                echo "============================================================================================="
                python src/run_evals_postprocessing.py \
                    ++combinations=$group_size \
                    +requests=open_ai_batch_kwargs \
                    ++setup.batch=true \
                    ++eval_model=$EVAL_MODEL \
                    summarizer_model=$SUMMARIZER_MODEL subsample=$SUBSAMPLE \
                    dataset_type=$dataset_type prompt_type=$prompt_type \
                    setup.config_filepath=configs/api_keys/openai.txt \
                    ++setup.evaluations_filepath="outputs/run_evals-single-request/${EVAL_MODEL}/\${dataset_type}/\${prompt_type}/SummHay__combinations-\${combinations}/\${summarizer_model}" \
                    ++setup.output_filepath="outputs/\${hydra:job.name}-single-request/${EVAL_MODEL}/\${dataset_type}/\${prompt_type}/SummHay__combinations-\${combinations}/\${summarizer_model}"
            done
        done
    done
done
