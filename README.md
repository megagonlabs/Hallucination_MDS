# From Single to Multi: How LLMs Hallucinate in Multi-Document Summarization

This repository contains the official implementation for the paper "[_From Single to Multi: How LLMs Hallucinate in Multi-Document Summarization_](https://arxiv.org/abs/2410.13961)" published at NAACL 2025 - Findings.

![alt text](https://github.com/megagonlabs/Hallucination_MDS/blob/main/figs/overview_MDS.png)

## 1. Constructing the datasets

Our work uses [SummHay (Laban et al 2024)](https://github.com/salesforce/summary-of-a-haystack/tree/master) as its basis. To create the dataset, we download the original `topic_news1.json`, ..., `topic_news5.json` files from the SummHay repository into the folder `./data/SummHay_news/raw` (or `./data/SummHay_conv/raw` for the conversation domain). 

The following script downloads the original `news` datasets from Laban et al 2024. Consider changing `news` with `conv` to obtain the datasets concerning the conversation domain.

```bash
# create the raw folder with the original dataset from Laban et al (2024)
mkdir -p ./data/SummHay_news/raw
# download each file
for i in 1 2 3 4 5; do
    curl -o  topic_news$i.json https://raw.githubusercontent.com/salesforce/summary-of-a-haystack/refs/heads/master/data/topic_news$i.json
done
```

Having created the raw files, we create different combinations of these datasets using the `src/create_dataset.py` Python code. If you'd like to replicate our setup, please consider running the script [`scripts/0.1_create_datasets.sh`](./scripts/0.1_create_datasets.sh) or [`scripts/0.2_create_datasets_adv.sh`](./scripts/0.2_create_datasets_adv.sh). 

```bash
$ ./scripts/0.1_create_datasets.sh news
```

or 

```bash
$ ./scripts/0.1_create_datasets_adv.sh news
```

These scripts do not depend on OpenAI or other APIs. They will generate different 5 different variants of the SummHay dataset. **Important Note**: the code has not been optimized, therefore running combinations of 10 documents will take a considerable amount of time for the `news` domain.


The generated variants can be described as: 
- `all_shared`: the goal is to test the models abilities to summarize the insights that are shared across all documents. The model should not generate insights that are not explicitly mentioned in all `n` documents.
- `some_shared`:  the goal is to test the models abilities to summarize the insights that are shared in at least `eta` documents. The model should not generate insights that are present or mentioned in a single document.
- `adversarial_subtopic`: the goal of this adversarial example is to test the model ability to follow the instruction and detect relevant information. We purposedly curate a set of examples where despite sharing information among each other, the model is asked to summarize the insights about a subtopic that is not shared across documents. The expected result is that the model would be able to identify and refuse to summarize. This one is generated from the set of `all_shared` by selecting a subtopic that is not present in either documents.


## 2. Generating summaries

Consider editing the scripts [1.1_generate_summaries.sh](./scripts/1.1_generate_summaries.sh) and [1.2_generate_summaries_adv.sh](./scripts/1.2_generate_summaries_adv.sh) to generate summaries for the desired models.

```batch
$ ./scripts/1.1_generate_summaries.sh
```

or 

```batch
$ ./scripts/1.2_generate_summaries_adv.sh
```

## 3. Automatic evaluation

We use a LLM-as-a-judge approach to automate the **coverage** evaluation. We rely on gpt-4o-mini. If you'd like to run the same evaluation as we did in the paper, then execute in your command line: 

```batch
$ ./scripts/2.1_single_req__eval_summaries_batch.sh
```

Note that we use the Batch API from OpenAI to reduce costs. As such, we 
need to collect the responses from OpenAI once they are completed, by running the script: 

```batch
$ ./scripts/2.2_single_req_post_process_eval_summaries_batch.sh
```

**Note**: We implement and validated two different approaches depending on how many annotations are provided in the output. We differentiate the two approaches into (1) single-request and (2) multi-request. The former consists of two OpenAI requests where each response contains the coverage annotation for all the insights; whereas the latter consists of |num_reference_insights| + |num_pred_insights| per example and the output only contains the coverage annotation for a single insight. See [single-request prompt](./configs/prompts/evaluation/single_request.txt) and [multi-request prompt](./configs/prompts/evaluation/multi_request.txt) to see examples. The second approach will be much more costly -- **USE WITH CAUTION**.

## 4. Combine coverage labels

Once we have collected all coverage labels for each reference insight and each predicted insight, we need to combine them to determine the final coverage label. This is done by running the `3_final_postprocessing.sh` scripts.

```bash
$ ./scripts/3_final_postprocessing.sh
```

## 5. Running mitigation

To run the mitigation techniques, consider running the `4_mitigation_{domain}_llm.sh` or `4_mitigation_{domain}_nli.sh` scripts, depending on whether you'd like to run the LLM-based techniques or the NLI-based ones, respectively.

# Data Source Attribution

Our benchmarks build upon data derived from SummHay datasets:
- Source: [SummHay Repository](https://github.com/salesforce/summary-of-a-haystack/tree/master)  
- License: **[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)**

**Please refer to the respective source for detailed licensing terms.**


# Usage Guidelines

- Use this dataset for **research and educational purposes**.  
- Commercial use may require additional permissions depending on source licenses. 


# Citing our work

If you found our work useful, please consider citing it:

```bibtex
@article{cbelem2025-hallucination-in-mds,
    title={From Single to Multi: How LLMs Hallucinate in Multi-Document Summarization},
    author={Belem, Catarina G, Pezeshkpour, Pouya and Iso, Hayate and Maekawa, Seiji and Bhutani, Nikita and Hruschka, Estevam},
    journal={arXiv preprint arXiv:2410.13961},
    year={2025}
}
```

# Disclosure
Embedded in, or bundled with, this product are open source software (OSS) components, datasets and other third party components identified below. The license terms respectively governing the datasets and third-party components continue to govern those portions, and you agree to those license terms, which, when applicable, specifically limit any distribution. You may receive a copy of, distribute and/or modify any open source code for the OSS component under the terms of their respective licenses, which may be CC license and Apache 2.0 license. In the event of conflicts between Megagon Labs, Inc., license conditions and the Open Source Software license conditions, the Open Source Software conditions shall prevail with respect to the Open Source Software portions of the software. You agree not to, and are not permitted to, distribute actual datasets used with the OSS components listed below. You agree and are limited to distribute only links to datasets from known sources by listing them in the datasets overview table below. You are permitted to distribute derived datasets of data sets from known sources by including links to original dataset source in the datasets overview table below. You agree that any right to modify datasets originating from parties other than Megagon Labs, Inc. are governed by the respective third party’s license conditions. All OSS components and datasets are distributed WITHOUT ANY WARRANTY, without even implied warranty such as for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE, and without any liability to or claim against any Megagon Labs, Inc. entity other than as explicitly documented in this README document. You agree to cease using any part of the provided materials if you do not agree with the terms or the lack of any warranty herein. While Megagon Labs, Inc., makes commercially reasonable efforts to ensure that citations in this document are complete and accurate, errors may occur. If you see any error or omission, please help us improve this document by sending information to contact_oss@megagon.ai.
