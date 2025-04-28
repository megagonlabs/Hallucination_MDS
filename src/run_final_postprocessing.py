from collections import defaultdict
from utils import generate_uuid, bullet_processor
from utils_io import read_json, to_json

import argparse, logging, os, copy

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


LLM_COVERAGE_LABELS = {
    "bidirectional": ("metric__bidirectional", "labels"),
    # Due to some inconsistencies in the data, we need to parse the coverage metrics differently
    # Note: the coverage metrics below are actually unidirectional but since they were computed
    # as a product of the bidirectional metric, they were placed under the "metric__bidirectional" key.
    "ref_coverage": ("metric__bidirectional", "ref_coverage"),
    "pred_coverage": ("metric__bidirectional", "pred_coverage"),
}


def get_eval_data(data_eval: dict, metric_name: str) -> dict:
    """Extract the evaluation data for a given metric."""
    parsing_keys = LLM_COVERAGE_LABELS.get(metric_name, [])
    
    for key in parsing_keys:
        data_eval = data_eval[key]
    return data_eval
    

def main(input_filepath: str, output_filepath: str, metric_name: str):
    # Read the file
    data = read_json(input_filepath)
    assert "assignments" in data, "No assignments found in the input file."

    # labels: will have a list of labels for each response
    # where each label is of the format: 
    #  { "coverage": "FULL_COVERAGE", "pred_uuid": "8f9b0ead689eda82700b81ff13a2bf42", "ref_uuid": "SfUMOWV1DMIMOJEQoep1m76h", "response_idx": 9 }
    labels: dict[int, list[dict]] = defaultdict(list)
    for r in get_eval_data(data["evaluation_assignments"], metric_name):
        labels[r["response_idx"]].append(r)
        
    final_labels_response = []
    for resp_id, response in enumerate(data["assignments"]):
        final_labels = []
        docs_uuid_ordered = [response["docs_uuids"][ord] for ord in response["docs_order"]]    
        doc_uuids2doc_ranks = {
            response["docs_uuids"][ord]:
            ord for ord in response["docs_order"]
        }
        preds_uuids = defaultdict(list)
        for rank, text in enumerate(response["response__parsed"]):
            pred_uuid = generate_uuid({"text": text})
            preds_uuids[pred_uuid].append((rank, text))
            
        
        # Keep track of preds and refs that are missing in the labels
        missing_preds_uuids = set(preds_uuids)        
        missing_refs_uuids = set(response["all_insights_uuids"])
        
        # Iterate over the labesl first, and populate w/ metadata
        # (these are known to be either PARTIAL_COVERAGE or FULL_COVERAGE)
        for label in labels[resp_id]:
            preds_info = preds_uuids[label["pred_uuid"]]
            
            for pred_info in preds_info:
                pred_label = copy.deepcopy(label)
                # ^Note: (rank, text)
                pred_label["pred_rank"] = pred_info[0]
                pred_label["pred_text"] = pred_info[1]
                
                ref_info = data["insights"][pred_label["ref_uuid"]]
                pred_label["ref_text"] = ref_info["insight"]
                pred_label["ref_subtopic"] = ref_info["subtopic"]
                pred_label["ref_subtopic_uuid"] = ref_info["subtopic_id"]

                # Determine the number of documents in which the ref insight occurs
                shared_doc_uuids = response["insights_to_documents"][pred_label["ref_uuid"]]
                pred_label["ref_num_shared"] = len(shared_doc_uuids)
                pred_label["ref_is_shared"] = len(shared_doc_uuids) > 1
                pred_label["ref_is_queried_subtopic"] = ref_info["subtopic_id"] == response["subtopic_uuid"]
                # Determine the origin of the reference insights
                pred_label["ref_doc_attribution"] = [doc_uuids2doc_ranks[id] for id in shared_doc_uuids]
                final_labels.append(pred_label)
            
            # Remove the preds and refs that are already covered
            missing_preds_uuids.discard(label["pred_uuid"])
            missing_refs_uuids.discard(label["ref_uuid"])
        
        # For any remaining uncovered reference, add a label with NO_COVERAGE
        # and no pred information
        for ref_uuid in missing_refs_uuids:
            label = {"coverage": "NO_COVERAGE", "ref_uuid": ref_uuid, "response_idx": resp_id}
            label["pred_uuid"] = None
            label["pred_rank"] = None
            label["pred_text"] = None
            
            # Determine the number of documents in which the ref insight occurs
            shared_doc_uuids = response["insights_to_documents"][ref_uuid]
            ref_info = data["insights"][ref_uuid]
            label["ref_text"] = ref_info["insight"]
            label["ref_subtopic"] = ref_info["subtopic"]
            label["ref_subtopic_uuid"] = ref_info["subtopic_id"]

            label["ref_num_shared"] = len(shared_doc_uuids)
            label["ref_is_shared"] = len(shared_doc_uuids) > 1
            label["ref_is_queried_subtopic"] = ref_info["subtopic_id"] == response["subtopic_uuid"]
            # Determine the origin of the reference insights
            label["ref_doc_attribution"] = [doc_uuids2doc_ranks[id] for id in shared_doc_uuids]
            final_labels.append(label)

        # For any remaining uncovered predictions, add a label with NO_COVERAGE
        # and no ref information (ref_uuid = None).
        for pred_uuid in missing_preds_uuids:
            label = {"coverage": "NO_COVERAGE", "pred_uuid": pred_uuid, "response_idx": resp_id, "ref_uuid": None}
            preds_info = preds_uuids[label["pred_uuid"]]
            
            for pred_info in preds_info:
                pred_label = copy.deepcopy(label)

                pred_label["pred_rank"] = pred_info[0]
                pred_label["pred_text"] = pred_info[1]
                
                pred_label["ref_text"] = None 
                pred_label["ref_subtopic"] = None # FIXME: Determine whether the topic is one of the available in the context
                pred_label["ref_subtopic_uuid"] = None

                pred_label["ref_num_shared"] = None
                pred_label["ref_is_shared"] = None
                pred_label["ref_is_queried_subtopic"] = None # FIXME: determine whether the topic is one
                pred_label["ref_doc_attribution"] = None # FIXME: determine doc attribution
                final_labels.append(pred_label)

        for label in final_labels:
            label["num_preds"] = len(response["response__parsed"])
            label["num_refs_in_context"] = len(response["all_insights_uuids"])
            label["docs_uuids_ordered"] = docs_uuid_ordered
            label["queried_subtopic"] = data["subtopics"][response["subtopic_uuid"]]["subtopic"]
            label["queried_subtopic_uuid"] = response["subtopic_uuid"]
            
        final_labels_response.extend(final_labels)

    final_data = {
        "documents": data["documents"],
        "insights": data["insights"],
        "subtopics": data["subtopics"],
        "topic": data["topic"],
        "topic_id": data["topic_id"],
        "labels": final_labels_response,
        "metric_name": metric_name,
        "input_filepath": input_filepath
    }
    to_json(output_filepath, final_data)
    
    
def is_adv_correct(bullet: str, adv_type: str) -> bool:
    if adv_type == "non-existing-subtopic":
        bullet_str = bullet.lower()
        for expr in ("no insights found", "insight not found", "insights not found", "no insights", "no related insights"):
            if expr in bullet_str:
                return True
        return False
    else:
        raise NotImplementedError(f"Adversarial type {adv_type} not implemented.")
 
def assert_valid_adv_response(response):
    assert response["is_adversarial"], "Response should be adversarial."
    
    if response["adversarial_type"] == "non-existing-subtopic":
        # Unlike the non-adversarial case, there are no labels for the adversarial data
        # (as a last check we will make sure that there are no insights related to the
        # subtopic in the context of the example)
        assert response["subtopic_uuid"] not in response["all_subtopics_uuids"],\
            "Adversarial data should not have insights related to the subtopic."
    else:
        raise NotImplementedError(f"Adversarial type {response['adversarial_type']} not implemented.")
 
 
def main_adversarial(input_filepath: str, output_filepath: str):
    # Read the file
    data = read_json(input_filepath)
    assert "assignments" in data, "No assignments found in the input file."
            
    # since we may want the information about the predictions
    # to determine where they're coming from in the input
    # we will keep track of the predictions and their ranks
    final_labels_response = []
    for response_id, response in enumerate(data["assignments"]):
        final_labels = []
        assert_valid_adv_response(response)

        docs_uuid_ordered = [response["docs_uuids"][ord] for ord in response["docs_order"]]
        bullets = [b.strip() for b in bullet_processor(response["response"])]
        for bullet_pos, bullet in enumerate(bullets):
            pred_uuid = generate_uuid({"text": bullet})
            
            label = {
                "correct": is_adv_correct(bullet, response["adversarial_type"]),
                "pred_uuid": pred_uuid,
                "response_idx": response_id,
                "ref_uuid": None,
            }
            label["pred_rank"] = bullet_pos
            label["pred_text"] = bullet
            
            label["ref_text"] = None 
            label["ref_subtopic"] = None
            label["ref_subtopic_uuid"] = None

            label["ref_num_shared"] = None
            label["ref_is_shared"] = None
            label["ref_is_queried_subtopic"] = None
            label["ref_doc_attribution"] = None
            final_labels.append(label)

            for label in final_labels:
                label["num_preds"] = len(bullets)
                label["num_refs_in_context"] = 0
                label["docs_uuids_ordered"] = docs_uuid_ordered
                label["queried_subtopic"] = data["subtopics"][response["subtopic_uuid"]]["subtopic"]
                label["queried_subtopic_uuid"] = response["subtopic_uuid"]
                label["is_adversarial"] = True
                label["adversarial_type"] = response["adversarial_type"]
        final_labels_response.extend(final_labels)

    final_data = {
        "documents": data["documents"],
        "insights": data["insights"],
        "subtopics": data["subtopics"],
        "topic": data["topic"],
        "topic_id": data["topic_id"],
        "labels": final_labels_response,
        "metric_name": "adversarial",
        "input_filepath": input_filepath
    }
    to_json(output_filepath, final_data)
    

   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the directory containing the evaluated summaries.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the directory to save the parsed metric.")
    parser.add_argument("--adversarial", action="store_true", help="Whether the input data is adversarial.")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    filenames = os.listdir(args.input_dir)
    filenames = [fn for fn in filenames if fn.endswith(".json")]
    
    for fn in filenames:
        input_path = f"{args.input_dir}/{fn}"
        
        if args.adversarial:
            output_path = f"{args.output_dir}/{fn}"
            main_adversarial(input_path, output_path)
            print("Done!")
        else:
            for metric_name in LLM_COVERAGE_LABELS.keys():
                output_path = f"{args.output_dir}/{metric_name}__{fn}"
                main(input_path, output_path, metric_name)
                print("Done!")
