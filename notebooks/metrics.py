import pandas as pd

from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Optional, Union


def combine_multiple_cov_labels(lst: List[Dict], metric_name: str) -> "Coverage":
    """Combines multiple coverage labels for the same piece of 
    information.
    
    Uses the best-case scenario to combine the coverage labels.
    Given the mapping from coverage labels to integers, we can
    code this to return the minimum integer value in the list.         
    """
    cov_mapping = {
        "FULL_COVERAGE": 0,
        "PARTIAL_COVERAGE": 1, 
        "NO_COVERAGE": 2,
    }
    if len(lst) == 0:
        raise ValueError("No coverage labels found.")

    values = lst
    if metric_name == "bidirectional":
        assert len(lst) == 1, f"{len(lst)} for {metric_name}"
        return values[0]
    elif metric_name == "ref_coverage":
        values = sorted(lst, key=lambda e: cov_mapping[e["coverage"]])
    elif metric_name == "pred_coverage":
        raise NotImplementedError("Not implemented")
    else:
        raise ValueError(f"Unknown metric name: {metric_name}")
    return values[0]


def compute_recall(
    labels: List[Dict[str, Any]],
    ref_is_queried_subtopic: bool=True,
    ref_is_shared: bool=False,
    max_rank: Union[str, bool]|None=None,
) -> pd.DataFrame:
    """Compute the recall per response id."""
    labels_by_idx = defaultdict(list)
    for label in labels:
        labels_by_idx[label["response_idx"]].append(label)
    
    results = defaultdict(list)
    for resp_uuid, ref_labels in labels_by_idx.items():        
        # Determine the number of expected relevant insights
        ref_labels = [l for l in ref_labels if l["ref_uuid"] is not None]
        
        if ref_is_queried_subtopic:
            ref_labels = [l for l in ref_labels if l["ref_is_queried_subtopic"]]
        
        if ref_is_shared:         
            ref_labels = [l for l in ref_labels if l["ref_is_shared"]]
        
        if len(ref_labels) == 0:
            print("0 after queried subtopic")
        
        num_refs = len(ref_labels)
        if max_rank is not None:
            if isinstance(max_rank, int):
                ref_labels = [l for l in ref_labels if l["pred_rank"] is not None and l["pred_rank"] < max_rank]
            elif max_rank == "num_refs":
                ref_labels = [l for l in ref_labels if l["pred_rank"] is not None and l["pred_rank"] < num_refs]
            else:
                raise ValueError("Unclear value for max_rank")
        
        num_full_cov_preds = len([l for l in ref_labels if l["coverage"] == "FULL_COVERAGE"])
        num_part_cov_preds = len([l for l in ref_labels if l["coverage"] == "PARTIAL_COVERAGE"])

        results["response_id"].extend([resp_uuid, resp_uuid])
        results["label_type"].extend(["fc", "fc+pc"])
        results["n"].extend([num_refs, num_refs])
        results["tp"].extend([num_full_cov_preds, 
                              num_full_cov_preds + num_part_cov_preds])
        results["fraction"].extend([num_full_cov_preds / num_refs, 
                                    (num_full_cov_preds+num_part_cov_preds) / num_refs])

        results["compute_recall_kwargs"].extend([{
            "max_rank": max_rank,
            "ref_is_shared": ref_is_shared,
            "ref_is_queried_subtopic": ref_is_queried_subtopic,
        }] *2)

    return pd.DataFrame(results)


def compute_error_rate(
    labels: List[Dict[str, Any]],
    ref_is_queried_subtopic: bool=True,
    ref_is_shared: bool=False,
    max_rank: Optional[Union[str, bool]]=None,
    metric_name: str="bidirectional",
    debug_num_preds: bool=True,
) -> pd.DataFrame:
    """Compute the hallucination_rate per response id."""
    def get_num_ref_labels(ls):
        ref_labels = [l for l in ls if l["ref_uuid"] is not None]
        if ref_is_queried_subtopic:
            ref_labels = [l for l in ref_labels if l["ref_is_queried_subtopic"]]
        if ref_is_shared:
            ref_labels = [l for l in ref_labels if l["ref_is_shared"]]
        return len(ref_labels)
        
    labels_by_idx = defaultdict(list)
    for label in labels:
        labels_by_idx[label["response_idx"]].append(label)

    count_checks = []
    results = defaultdict(list)
    for resp_uuid, labels in labels_by_idx.items():
        pred_labels = [l for l in labels if l["pred_uuid"] is not None]
        # ^note: pred_uuid will be none for the reference annotations that were not covered
        
        if len(pred_labels) < labels[0]["num_preds"]:
            count_checks.append(f"{resp_uuid}: {len(pred_labels)} != {labels[0]["num_preds"]}")
            # will happen if pred_uuids were exactly the same (which happens for LLAMA 3.1 once)
        elif len(pred_labels) > labels[0]["num_preds"]:
            raise ValueError(f"resp_id={resp_uuid}: {len(pred_labels)} "
                             f"!= {labels[0]["num_preds"]}, {labels[0]["num_refs_in_context"]}")

        # filter out the number of predictions based on max_rank
        # ----------------------------------------------------------
        if max_rank is not None:
            if isinstance(max_rank, int): # all preds w/ rank less than max_rank
                pred_labels = [l for l in pred_labels if l["pred_rank"] < max_rank]
            elif max_rank == "num_refs":  # limit based on ground truth value
                num_refs = get_num_ref_labels(labels)
                pred_labels = [l for l in pred_labels if l["pred_rank"] < num_refs]
            else:
                raise ValueError(f"Unclear value for max_rank: {max_rank}")

        # Number of predictions after filtering based on max_rank
        num_preds = len(pred_labels)
        
        # Given that the metric considers all insights in context,
        # we know that if there is a full or partial coverage it means
        # that it was mapped to some insight that is in context.
        tp_labels = pred_labels
        if ref_is_queried_subtopic:
            tp_labels = [l for l in tp_labels if l["ref_is_queried_subtopic"]]
        if ref_is_shared:         
            tp_labels = [l for l in tp_labels if l["ref_is_shared"]]

        num_full_cov_preds = len([l for l in tp_labels if l["coverage"] == "FULL_COVERAGE"])
        num_part_cov_preds = len([l for l in tp_labels if l["coverage"] == "PARTIAL_COVERAGE"])
        results["response_id"].extend([resp_uuid, resp_uuid])
        results["label_type"].extend(["fc", "fc+pc"])
        results["n"].extend([num_preds] *2)
        results["tp"].extend([
            num_full_cov_preds, 
            num_full_cov_preds + num_part_cov_preds
        ])
        results["fp"].extend([
            num_preds - num_full_cov_preds, 
            num_preds - (num_full_cov_preds + num_part_cov_preds)
        ])
        if num_preds == 0:
            results["fraction"].extend([None, None])
        else:
            results["fraction"].extend([
                (num_preds - num_full_cov_preds) / num_preds, 
                (num_preds - (num_full_cov_preds + num_part_cov_preds)) / num_preds
            ])

        results["compute_hallucination_kwargs"].extend([{
            "max_rank": max_rank,
            "ref_is_shared": ref_is_shared,
            "ref_is_queried_subtopic": ref_is_queried_subtopic,
        }] *2 )

    if debug_num_preds and len(count_checks)>0:
        print("pred_counts mismatch:", count_checks)
        
    return pd.DataFrame(results)


def f_beta_score(recall, precision, beta=1):
    """By default it computes f1 score, which assumes equal importance
    between recall and precision. If recall is 2x more important than
    precision, then set `beta=2`."""
    fac1 = (1 + beta*beta)
    fac2_num = precision*recall
    fac2_den = (beta*beta) * precision + recall
    if fac2_den == 0:
        return 0
        
    return fac1 * fac2_num / fac2_den


def compute_doc_provenance(data, cov_labels=("NO_COVERAGE","PARTIAL_COVERAGE", "FULL_COVERAGE")):
    # Label by response id
    labels_by_idx = defaultdict(list)
    for label in data["labels"]:
        labels_by_idx[label["response_idx"]].append(label)
        
    # For each response, check the attribution of every ref insight
    results = defaultdict(list)
    for response_id, labels in labels_by_idx.items():
        # get ref labels
        ref_labels = [l for l in labels if l["ref_uuid"] is not None]
    
        for l in ref_labels:
            ids_of_docs = set([i for i in range(len(ref_labels[0]["docs_uuids_ordered"]))])

            if l["coverage"] in cov_labels:
                for doc_id in l["ref_doc_attribution"]:
                    results["response_id"].append(response_id)
                    results["doc_id"].append(doc_id)
                    results["ref_uuid"].append(l["ref_uuid"])
                    results["count"].append(1)
                    ids_of_docs.discard(doc_id)
    
            for doc_id in ids_of_docs:
                results["response_id"].append(response_id)
                results["doc_id"].append(doc_id)
                results["ref_uuid"].append(l["ref_uuid"])
                results["count"].append(0)
    return pd.DataFrame(results)