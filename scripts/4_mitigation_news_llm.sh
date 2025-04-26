#!/bin/bash
set -e
EVAL_TYPE=multi-request
EVAL_MODEL=gpt-4o-mini-2024-07-18

FILTERS=$1
PROMPT_TYPE="subtopic"
DATASET_DOMAIN="news"

# none of the filters is summarizer specific. while it can eventually lead to bottlenecks in loading the files,
# it should be fine for now and may actually be beneficial in terms of cost and time complexity (if there are
# repeated predicted insights)
TEMP_DIR=./outputs_$DATASET_DOMAIN/run_mitigation-$EVAL_TYPE/$EVAL_MODEL/results_some_shared/$PROMPT_TYPE/SummHay/filters

for GROUP_SIZE in 2; do # 
    for SUMMARIZER_MODEL in gpt-3.5-turbo-0125; do # gpt-4o-2024-05-13 gemini-1.5-flash accounts/fireworks/models/llama-v3p1-70b-instruct accounts/fireworks/models/qwen2-72b-instruct; do # 
        BASENAME=$EVAL_MODEL/results_some_shared/$PROMPT_TYPE/SummHay__combinations-${GROUP_SIZE}/$SUMMARIZER_MODEL        
        # -------------------------------------------------
        # Filter 1: Execute output-based position mitigation
        # -------------------------------------------------        
        if [[ $FILTERS == *"position"* ]]; then
            END=6
            echo "============================> POSITION-$END <==========================="
            python src/run_mitigation.py \
            ++filepaths.input=./outputs_$DATASET_DOMAIN/run_evals_postprocessing-$EVAL_TYPE/$BASENAME \
            ++filepaths.output=./outputs_$DATASET_DOMAIN/run_mitigation-$EVAL_TYPE/$BASENAME/position-$END \
            ++filepaths.temp=$TEMP_DIR \
            +filters=position ++filters.position.end=$END

            python src/run_final_postprocessing_after_mitigation.py \
                --input_dir ./outputs_$DATASET_DOMAIN/run_mitigation-$EVAL_TYPE/$BASENAME/position-$END \
                --output_dir ./outputs_$DATASET_DOMAIN/run_mitigation_final_postprocessing-$EVAL_TYPE/$BASENAME/position-$END
        fi

        # -------------------------------------------------
        # Filter 2: Executes non-shared output mitigation (only for subtopic_trustworthy)
        # -------------------------------------------------        
        if [[ $FILTERS == *"not_shared"* ]] && [[ $PROMPT_TYPE == "subtopic_trustworthy" ]]; then
            SUBSTR_LEN=50

            echo "============================> NOT_SHARED-$PROMPT_TYPE <==========================="
            python src/run_mitigation.py \
                ++filepaths.input=./outputs_$DATASET_DOMAIN/run_evals_postprocessing-$EVAL_TYPE/$BASENAME \
                ++filepaths.output=./outputs_$DATASET_DOMAIN/run_mitigation-$EVAL_TYPE/$BASENAME/not_shared_$SUBSTR_LEN \
                ++filepaths.temp=$TEMP_DIR \
                +filters=not_shared ++filters.not_shared.substr_len=$SUBSTR_LEN ++filters.not_shared.shared_by_or_more=2

            python src/run_final_postprocessing_after_mitigation.py \
                --input_dir ./outputs_$DATASET_DOMAIN/run_mitigation-$EVAL_TYPE/$BASENAME/not_shared_$SUBSTR_LEN \
                --output_dir ./outputs_$DATASET_DOMAIN/run_mitigation_final_postprocessing-$EVAL_TYPE/$BASENAME/not_shared_$SUBSTR_LEN
        fi

        # -------------------------------------------------
        # Filter 3: Unrelated subtopic
        # -------------------------------------------------        
        if [[ $FILTERS == *"unrelated_subtopic"* ]]; then
            SCORE_THRESHOLD=0.8
            echo "============================> UNRELATED_SUBTOPIC-$SCORE_THRESHOLD <==========================="
            python src/run_mitigation.py \
            ++filepaths.input=./outputs_$DATASET_DOMAIN/run_evals_postprocessing-$EVAL_TYPE/$BASENAME \
            ++filepaths.output=./outputs_$DATASET_DOMAIN/run_mitigation-$EVAL_TYPE/$BASENAME/unrelated_subtopic-$SCORE_THRESHOLD \
            ++filepaths.temp=$TEMP_DIR \
            +filters=unrelated_subtopic ++filters.unrelated_subtopic.confident_threshold=$SCORE_THRESHOLD

            python src/run_final_postprocessing_after_mitigation.py \
                --input_dir ./outputs_$DATASET_DOMAIN/run_mitigation-$EVAL_TYPE/$BASENAME/unrelated_subtopic-$SCORE_THRESHOLD \
                --output_dir ./outputs_$DATASET_DOMAIN/run_mitigation_final_postprocessing-$EVAL_TYPE/$BASENAME/unrelated_subtopic-$SCORE_THRESHOLD
        fi

        # -------------------------------------------------
        # Filter 4: Subtopic paraphrase
        # -------------------------------------------------        
        if [[ $FILTERS == *"subtopic_paraphrase"* ]]; then
            SCORE_THRESHOLD=null
            echo "============================> SUBTOPIC_PARAPHRASE-$SCORE_THRESHOLD <==========================="
            python src/run_mitigation.py \
            ++filepaths.input=./outputs_$DATASET_DOMAIN/run_evals_postprocessing-$EVAL_TYPE/$BASENAME \
            ++filepaths.output=./outputs_$DATASET_DOMAIN/run_mitigation-$EVAL_TYPE/$BASENAME/subtopic_is_paraphrase__sts_llm-$SCORE_THRESHOLD \
            ++filepaths.temp=$TEMP_DIR \
            +filters=subtopic_is_paraphrase__sts_llm ++filters.subtopic_is_paraphrase__sts_llm.confident_threshold=$SCORE_THRESHOLD

            python src/run_final_postprocessing_after_mitigation.py \
                --input_dir ./outputs_$DATASET_DOMAIN/run_mitigation-$EVAL_TYPE/$BASENAME/subtopic_is_paraphrase__sts_llm-$SCORE_THRESHOLD \
                --output_dir ./outputs_$DATASET_DOMAIN/run_mitigation_final_postprocessing-$EVAL_TYPE/$BASENAME/subtopic_is_paraphrase__sts_llm-$SCORE_THRESHOLD
        fi

        # -------------------------------------------------
        # Filter 5: Redundant preds
        # -------------------------------------------------        
        if [[ $FILTERS == *"preds_redundant"* ]]; then
            SCORE_THRESHOLD=null
            echo "============================> PREDS_REDUNDANT-$SCORE_THRESHOLD <==========================="
            python src/run_mitigation.py \
            ++filepaths.input=./outputs_$DATASET_DOMAIN/run_evals_postprocessing-$EVAL_TYPE/$BASENAME \
            ++filepaths.output=./outputs_$DATASET_DOMAIN/run_mitigation-$EVAL_TYPE/$BASENAME/preds_are_redundant_sts_llm-$SCORE_THRESHOLD \
            ++filepaths.temp=$TEMP_DIR \
            +filters=preds_are_redundant_sts_llm ++filters.preds_are_redundant_sts_llm.threshold=$SCORE_THRESHOLD

            python src/run_final_postprocessing_after_mitigation.py \
                --input_dir ./outputs_$DATASET_DOMAIN/run_mitigation-$EVAL_TYPE/$BASENAME/preds_are_redundant_sts_llm-$SCORE_THRESHOLD \
                --output_dir ./outputs_$DATASET_DOMAIN/run_mitigation_final_postprocessing-$EVAL_TYPE/$BASENAME/preds_are_redundant_sts_llm-$SCORE_THRESHOLD
        fi
    done
done 

