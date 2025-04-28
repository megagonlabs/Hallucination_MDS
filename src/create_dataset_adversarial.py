import argparse, logging, os, random
import utils_io
from utils_logging import setup_logger


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def make_assignments_adversarial(filepath, filename, args):
    os.makedirs(args.output_dir, exist_ok=True)
    output_filepath = f"{args.output_dir}/{filename}.json"

    data = utils_io.read_json(filepath)
    # ^Note: we assume this is an assignment input path
    # that has been previously generated using `create_dataset.py`
    uuid2subtopics = data["subtopics"]
    num_subtopics = len(uuid2subtopics)
    # Switch the subtopic with a subtopic that is not present in the
    # union of the n_combination documents
    final_assignments = []
    for assignment in data["assignments"]:
        all_subtopics = assignment["all_subtopics_uuids"]
        
        if len(all_subtopics) == num_subtopics:
            logger.warning(f"Skipping assignment {assignment}:"
                        "impossible to create adversarial example")
            continue
        
        # Select a subtopic that is not present in either document 
        subtopics = [uuid for uuid in uuid2subtopics.keys() if uuid not in all_subtopics]
        rand_id = random.randrange(len(subtopics))
        subtopic = subtopics[rand_id]
        
        assignment["is_adversarial"] = True
        assignment["adversarial_type"] = 'non-existing-subtopic'
        assignment["subtopic_uuid"] = subtopic
        assignment["ground_truth_insights_uuids"] = []
        final_assignments.append(assignment)
    
    data["assignments"] = final_assignments
    for a in data["assignments"]:
        assert a["subtopic_uuid"] not in a["all_subtopics_uuids"], f"Subtopic {a["subtopic_uuid"]} is present in the document."
    logger.info(f"-> {len(final_assignments)} modified assignments dumped @ {output_filepath}")
    utils_io.to_json(output_filepath, data)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create an adversarial dataset for the summarization task."
    )
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--logging_filepath", type=str, default="./create_dataset_adversarial.log")
    parser.add_argument("--seed", type=int, default=17623)
    args = parser.parse_args()

    setup_logger(args.logging_filepath, logger, __name__)
    logger.info(f"\n{"=" * 80}\n{args}\n{"=" * 80}")
    
    # Load documents
    random.seed(args.seed)
    filenames = [fn for fn in os.listdir(args.input_dir) if fn.endswith(".json") and (fn.startswith("topic_news") or fn.startswith("topic_conv"))]
    for file in sorted(filenames):
        input_filepath = f"{args.input_dir}/{file}"
        logger.info(f"*** Processing file {input_filepath}")
        make_assignments_adversarial(input_filepath, file.replace(".json", ""), args)
        