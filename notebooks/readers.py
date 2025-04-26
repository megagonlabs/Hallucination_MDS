from collections import Counter
import re, json, pickle
import pandas as pd


def read_json(fp):
    with open(fp) as f:
        return json.load(f)


def read_pickle(fp):
    with open(fp, "rb") as f:
        return pickle.load(f)


def read_and_merge_paths(fps):
    def _get_eval_model(fp): 
        return fp.rpartition("collect_metrics/")[-1].split("/")[0]

    def _get_num_docs(fp):
        text = re.sub(r'combinations-\d{1,3}', 'combinations-', fp, flags=re.I)
        text = fp.rpartition("combinations-")[-1]
        text = text.split("/")[0]
        return int(text)
    
    def _get_summarizer_model(fp, num_docs: int): 
        text = fp.rpartition("/")[0]

        offset = len(str(num_docs)) + 1
        text = re.sub(r'combinations-\d{1,3}', 'combinations-', text, flags=re.I)
        text = text.rpartition("combinations-")[-1][1:]
        text = text.rpartition("models/")[-1]
        return text

    cols_counter = Counter()
    results = []
    for fp in sorted(fps):
        try:
            # print("Reading file:", fp)
            df = pd.read_csv(fp)
            
            n_docs = _get_num_docs(fp)
            df["num_documents"] = n_docs
    
            df["eval_model"] = _get_eval_model(fp)
            df["summarizer_model"] = _get_summarizer_model(fp, n_docs)
            cols_counter.update(df.columns)
            results.append(df)
        except:
            print("\t -> Couldn't read file", fp, "\n")

    cols = sorted([k for k, v in cols_counter.items() if v == len(results)])
    results = pd.concat([r[cols] for r in results])
    return results