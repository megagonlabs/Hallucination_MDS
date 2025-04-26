import argparse, itertools, functools, logging, os, random
import utils, utils_io

from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple
from utils_logging import setup_logger


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def collect_uuids(data: Dict[str, Any]) -> Tuple[Dict[str, Any], ...]:
    uuid2docs = {doc["document_id"]: doc for doc in data["documents"]}

    exclude_cols = ["eval_summaries", "retriever", "summaries"]
    # Dont want to add unnecessary information
    uuid2subtopics = {} 
    for stopic in data["subtopics"]: 
        stopic = {k: v for k, v in stopic.items() if k not in exclude_cols}
        uuid2subtopics[stopic["subtopic_id"]] = stopic

    uuid2insights = {}
    for stopic in data["subtopics"]:
        for ins in stopic["insights"]:
            ins.update({k: v for k, v in stopic.items() if k not in (exclude_cols + ["insights"])})
            uuid2insights[ins["insight_id"]] = ins

    return uuid2docs, uuid2subtopics, uuid2insights


def subtopic_insight_info(
    document: dict,
    uuid2subtopics: dict,
    uuid2insights: dict,
    subtopic: bool = True,
    insight: bool = True,
):
    label_cols = [c.replace("label_", "") for c in document if c.startswith("label_")]

    info = {"subtopic": {}, "insight": {}}
    for lcol in label_cols:
        if subtopic and uuid2subtopics.get(lcol) is not None:
            if document["label_" + lcol] > 0:
                info["subtopic"][lcol] = document["label_" + lcol]

        if insight and uuid2insights.get(lcol) is not None:
            if document["label_" + lcol] > 0:
                info["insight"][lcol] = document["label_" + lcol]
    return info


def get_overlapping_docs(
    data: Dict[str, Any], uuid2subtopics: Dict[str, Any], uuid2insights: Dict[str, Any]
) -> Tuple[Dict[str, Any], ...]:
    all_docs_info = {}

    # subtopic to list of documents containing that subtopic
    subtopic_overlap = defaultdict(list)
    # insight uuid to list of documents containing that insight
    insight_overlap = defaultdict(list)
    for document in data["documents"]:
        info = subtopic_insight_info(
            document,
            subtopic=True,
            insight=True,
            uuid2subtopics=uuid2subtopics,
            uuid2insights=uuid2insights,
        )
        for subtopic_id in info["subtopic"]:
            subtopic_overlap[subtopic_id].append(document["document_id"])
        for insight_id in info["insight"]:
            insight_overlap[insight_id].append(document["document_id"])
        all_docs_info[document["document_id"]] = info

    return all_docs_info, subtopic_overlap, insight_overlap


def shared_insights_all(
    subtopic_uuid: str, doc_uuids: list, threshold: int, all_docs_info, uuid2insights
):
    """Determines whether the number of shared insights about the same
    subtopic is greater than or equal to ``threshold``.
    
    Parameters
    ----------
    subtopic_uuid : str
        The subtopic uuid to check for shared insights.

    doc_uuids : list[str]
        The list of document uuids of the combination.

    threshold : int
        The minimum number of shared insights required to be present
        in the combination to be a valid assignment.

    all_docs_info : dict[str, dict[str, dict[str, int]]]
        The metadata information about the documents' insights and
        subtopic counts. The format is as follows:
        { 
            "doc_uuid1": {
                "subtopic": {"subtopic_uuid1": count1, ...},
                "insight": {"insight_uuid1": count1, ...},
            }
        }

    uuid2insights : dict[str, dict[str, Any]]
        A dictionary mapping insight uuids to their metadata information.
    """
    if len(doc_uuids) == 0:
        return False

    # Count the number of documents containing each insight
    shared_insights = Counter()
    for uuid in doc_uuids:
        shared_insights.update(all_docs_info[uuid]["insight"].keys())

    # Filter out insights that are not about the subtopic_uuid
    shared_insights = {uuid: counts for uuid, counts in shared_insights.items() if uuid2insights[uuid]["subtopic_id"] == subtopic_uuid}
    
    # Filter out insights whose counts are not equal to the number of documents
    shared_insights = {uuid: counts for uuid, counts in shared_insights.items() if counts == len(doc_uuids)}

    if len(shared_insights) < threshold:
        return False

    # Check subtopic for each insight
    shared_subtopics = [uuid2insights[ins]["subtopic_id"] for ins in shared_insights]
    shared_subtopics = [uuid for uuid in shared_subtopics if uuid == subtopic_uuid]
    return len(shared_subtopics) >= threshold


def shared_insights_across_some(
    subtopic_uuid: str, doc_uuids: list, threshold: int, uuid2insights, all_docs_info, insights_overlap
):
    if len(doc_uuids) == 0:
        return False

    # Filter the insights to be the ones referring to the same subtopic
    shared_insights = Counter()
    for uuid in doc_uuids:
        shared_insights.update(all_docs_info[uuid]["insight"].keys())

    # Filter out insights whose counts are either equal to the number of documents
    # or less than the threshold
    shared_insights = {uuid: counts for uuid, counts in shared_insights.items() if threshold <= counts < len(doc_uuids)}
    # Filter by subtopic
    shared_subtopics = [uuid2insights[ins]["subtopic_id"] for ins in shared_insights]
    shared_subtopics = [uuid for uuid in shared_subtopics if uuid == subtopic_uuid]

    # occurs in 2 or more documents
    return len(shared_subtopics) >= 2


def no_shared_insights(
    subtopic_uuid: str, doc_uuids: list, threshold: int, uuid2insights, all_docs_info, insights_overlap
):
    if len(doc_uuids) == 0:
        return False

    # Filter the insights to be the ones referring to the same subtopic
    filter_insights = set()
    for uuid in doc_uuids:
        filter_insights.update(
            [
                ins
                for ins in all_docs_info[uuid]["insight"]
                if uuid2insights[ins]["subtopic_id"] == subtopic_uuid
            ]
        )

    if len(filter_insights) < threshold:
        return False

    # Check if there are enough insights that are shared across
    shared = []
    for uuid in filter_insights:
        docs_w_insight = set(insights_overlap[uuid]).intersection(set(doc_uuids))
        if len(docs_w_insight) > 1:
            shared.append(shared)

    # Check if there are any insights that are shared across all documents
    # (if so, return False as we want to avoid this scenario)
    if any([len(t) == len(doc_uuids) for t in shared]):
        return False
    return len(shared) >= threshold


def create_assignments(
    subtopic_overlap: Dict[str, List[str]], filter_fn: callable, n_docs: int
) -> List[Dict[str, Any]]:
    """Create assignments (or combinations) of documents that share at 
    least a subtopic.
    
    Parameters
    ----------
    subtopic_overlap : Dict[str, List[str]]
        A dictionary mapping subtopic uuids to lists of document uuids.
        
    filter_fn : callable
        A function that filters out invalid assignments.
        
    n_docs : int
        The number of documents to include in the assignments.
    """
    # 1. Generate all possible combinations for each subtopic
    possible_combinations = {
        st_uuid: itertools.combinations(st_docs, n_docs)
        for st_uuid, st_docs in subtopic_overlap.items()
    }

    # Filter out invalid combinations and store the valid ones
    assignments = []
    for st_uuid, combinations in possible_combinations.items():
        for combination in combinations:
            if filter_fn(st_uuid, combination):
                combo = {"subtopic_uuid": st_uuid, "docs_uuids": combination}
                combo.update(
                    {
                        "_uuid_": utils.generate_uuid(combo),
                        # ^Previously named "example_uuid_". This can be used
                        # to uniquely identify the combination of documents for a given
                        # subtopic.
                        "_documents_uuid_": utils.generate_uuid(combination),
                        # ^Previously named "assignment_uuid_" identifies the
                        # set of documents uniquely (regardless of the subtopic)
                        # Can be used to find if there is a particular combination
                        # of docs that appears and is used to query about different
                        # subtopics.
                    }
                )
                assignments.append(combo)
                
    assert len(assignments) > 0, "No assignments found!"
    return assignments


def populate_(
    assignments: list,
    dataset_type: str,
    uuid2docs,
    uuid2subtopics,
    uuid2insights,
) -> Dict[str, Any]:
    """Populate the document assignments with references.

    The document assignment are lists of dictionaries formatted as follows:
    {
        "subtopic_uuid": str,
        "docs_uuids": list[str],
        "assignment_uuid_": str,
        "example_uuid_": str,
    }, where:
     * subtopic_uuid: The subtopic uuid that the documents share.
     * docs_uuids: The list of document uuids that share the subtopic.
     * assignment_uuid_: The unique identifier for the assignment.
     * example_uuid_: The unique identifier for the example.

    Each assignment is a tuple of document uuids and a subtopic uuid. To
    this end, we will store the following metadata:
     * num_documents: Number of documents in the assignment.
     * dataset_type: The type of dataset that we are creating.
     * subtopic_uuid: The subtopic uuid that the documents share. Note that
        the documents may share other insights about other subtopics but this
        represents the one that is to be evaluated in the task.
     * docs_uuids: Document uuids of the combination.
     * ground_truth_insights_uuids: Shared insights of the same subtopic.
     * all_insights_uuids: All insights in the combination.
     * insights_to_documents: Mapping of insights to documents uuids.
     * subtopics_to_documents: Mapping of subtopics to documents uuids.
     * document_uuid_i_insights: The insights for each document.
     * document_shared_insights: The insights that are shared across
        the documents in the assignment.
     * document_uuid_i_subtopic: The subtopics for each document.
     * document_shared_subtopics: The subtopics that are shared across the
        documents in the assignment.
    """
    metadata = []
    for assignment in assignments:
        assignment.update({"dataset_type": dataset_type})
        assignment["num_documents"] = len(assignment["docs_uuids"])

        assignment["insights_to_documents"] = defaultdict(list)
        assignment["subtopics_to_documents"] = defaultdict(list)

        all_insights = Counter()
        all_subtopics = Counter()
        for i, doc_uuid in enumerate(assignment["docs_uuids"]):
            assignment.update({f"doc_uuid_{i}": doc_uuid})

            # Collect insights
            doc = uuid2docs[doc_uuid]
            assignment.update({f"insights__doc_uuid_{i}": doc["insights_included"]})
            # Collect subtopics
            subtopics = set([
                    uuid2insights[ins_uuid]["subtopic_id"]
                    for ins_uuid in doc["insights_included"]
            ])
            assignment.update({f"subtopics__doc_uuid_{i}": list(subtopics)})

            # Update counter
            all_insights.update(doc["insights_included"])
            all_subtopics.update(subtopics)
            
            for ins in doc["insights_included"]:
                assignment["insights_to_documents"][ins].append(doc_uuid)
            for sub in subtopics:
                assignment["subtopics_to_documents"][sub].append(doc_uuid)

        assignment["all_insights_uuids"] = list(all_insights.keys())
        assignment["all_subtopics_uuids"] = list(all_subtopics.keys())

        n = assignment["num_documents"]
        assignment["all_shared_insights_uuids"] = [
            uuid for uuid, counts in all_insights.items() if counts == n
        ]
        assignment["all_shared_subtopics_uuids"] = [
            uuid for uuid, counts in all_subtopics.items() if counts == n
        ]

        assignment["some_shared_insights_uuids"] = [
            uuid for uuid, counts in all_insights.items() if 1 < counts < n
        ]
        assignment["some_shared_subtopics_uuids"] = list(set([
            uuid2insights[a]['subtopic_id'] for a in assignment["some_shared_insights_uuids"]
        ]))
        
        ground_truth = [
            uuid for uuid in assignment[f"{dataset_type}_insights_uuids"]
            if uuid2insights[uuid]["subtopic_id"] == assignment["subtopic_uuid"]
        ]
        assignment["ground_truth_insights_uuids"] = ground_truth
        
        metadata.append(assignment)
    return metadata



def main(input_filepath: str, filename: str, args, insights_shared_all=False, insights_shared_some=False, insights_adversarial=False): 
    data = utils_io.read_json(input_filepath)
    uuid2docs, uuid2subtopics, uuid2insights = collect_uuids(data)

    all_docs_info, subtopic_overlap, insights_overlap = get_overlapping_docs(
        data, uuid2subtopics=uuid2subtopics, uuid2insights=uuid2insights
    )

    # =========================================================================
    #   Generate insights that are shared across all documents in combination
    # =========================================================================
    if insights_shared_all and args.combinations > 1:
        logger.info(f"-> Generating insights shared across all documents...")
        output_fp = f"{args.output_dir}/all_shared/combinations-{args.combinations}"
        os.makedirs(output_fp, exist_ok=True)
        output_fp = (
            f"{output_fp}/{filename}__{args.sample_size}.json"
        )

        filter_fn = functools.partial(
            shared_insights_all,
            threshold=args.eta,
            uuid2insights=uuid2insights,
            all_docs_info=all_docs_info,
        )
        assignments = create_assignments(
            subtopic_overlap, filter_fn, args.combinations
        )

        assignments = utils.sample_list(assignments, n=args.sample_size)
        preprocessed_data = populate_(
            assignments,
            "all_shared",
            uuid2docs,
            uuid2subtopics,
            uuid2insights,
        )
        
        for a in preprocessed_data:
            assert len(a["ground_truth_insights_uuids"]) > 0, "Unexpected: No insights found!"
            for uuid in a["ground_truth_insights_uuids"]:
                assert uuid2insights[uuid]["subtopic_id"] == a["subtopic_uuid"], "Unexpected: Insights do not match subtopic!"
                docs_per_insight = set(insights_overlap[uuid]).intersection(a["docs_uuids"])
                assert len(docs_per_insight) == len(a["docs_uuids"]), "Unexpected: Insights do not match documents!"
        
        preprocessed_data = {
            "assignments": preprocessed_data,
            "documents": uuid2docs,
            "insights": uuid2insights,            
            "subtopics": uuid2subtopics,
            "topic": data["topic"],
            "topic_id": data["topic_id"],  
            "topic_metadata": data["topic_metadata"],        
        }
        logger.info(f"-> Assignments dumped @ {output_fp}")
        utils_io.to_json(output_fp, preprocessed_data)
        
        if args.combinations == 2:
            output_fp = output_fp.replace("all_shared", "some_shared")
            logger.info(f"-> Assignments dumped @ {output_fp}")
            utils_io.to_json(output_fp, preprocessed_data)

    # =============================================================================
    # Generate insights that are shared across only some documents in combination
    # =============================================================================
    if insights_shared_some and args.combinations > 2:
        logger.info(f"-> Generating insights shared across some documents only...")
        output_fp = f"{args.output_dir}/some_shared/combinations-{args.combinations}"
        os.makedirs(output_fp, exist_ok=True)
        output_fp = (
            f"{output_fp}/{filename}__{args.sample_size}.json"
        )

        filter_fn = functools.partial(
            shared_insights_across_some,
            threshold=args.eta,
            uuid2insights=uuid2insights,
            all_docs_info=all_docs_info,
            insights_overlap=insights_overlap,
        )
        assignments = create_assignments(subtopic_overlap, filter_fn, args.combinations)
        assignments = utils.sample_list(assignments, n=args.sample_size)
        preprocessed_data = populate_(
            assignments,
            "some_shared",
            uuid2docs,
            uuid2subtopics,
            uuid2insights,
        )
        
        for a in preprocessed_data:
            assert len(a["ground_truth_insights_uuids"]) > 0, "Unexpected: No insights found!"
            for uuid in a["ground_truth_insights_uuids"]:
                assert uuid2insights[uuid]["subtopic_id"] == a["subtopic_uuid"], "Unexpected: Insights do not match subtopic!"
                docs_per_insight = set(insights_overlap[uuid]).intersection(a["docs_uuids"])
                assert args.eta <= len(docs_per_insight), "Unexpected: Insight does not satisfy threshold!"
                assert len(docs_per_insight) < len(a["docs_uuids"]), "Unexpected: Insight is present in all documents!"
        
        
        preprocessed_data = {
            "assignments": preprocessed_data,
            "documents": uuid2docs,
            "insights": uuid2insights,            
            "subtopics": uuid2subtopics,
            "topic": data["topic"],
            "topic_id": data["topic_id"],
            "topic_metadata": data["topic_metadata"],
        }
        logger.info(f"-> Assignments dumped @ {output_fp}")
        utils_io.to_json(output_fp, preprocessed_data)
        
    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a dataset for the summarization task."
    )
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--logging_filepath", type=str, default="./create_datasets.log")
    parser.add_argument("--combinations", type=int, default=2)
    parser.add_argument(
        "--sample_size", type=int, default=100, help="Number of examples to create."
    )
    parser.add_argument(
        "--eta", type=int, default=2, help="Number of minimum shared insights."
    )
    parser.add_argument("--insights_shared_all", action="store_true")
    parser.add_argument("--insights_shared_some", action="store_true")
    parser.add_argument("--insights_adversarial", action="store_true")
    parser.add_argument("--seed", type=int, default=17623)
    args = parser.parse_args()

    setup_logger(args.logging_filepath, logger, __name__)
    logger.info(f"\n{"=" * 80}\n{args}\n{"=" * 80}")
    
    # Load documents
    random.seed(args.seed)
    filenames = [fn for fn in os.listdir(args.input_dir) if fn.endswith(".json") and fn.startswith("topic_news") or fn.startswith("topic_conv")]
    for file in sorted(filenames):
        input_filepath = f"{args.input_dir}/{file}"
        logger.info(f"*** Processing file {input_filepath}")
        
        if args.insights_shared_all:
            main(input_filepath, file.replace(".json", ""), args, insights_shared_all=True)
        if args.insights_shared_some:
            main(input_filepath, file.replace(".json", ""), args, insights_shared_some=True)
        if args.insights_adversarial:
            main(input_filepath, file.replace(".json", ""), args, insights_adversarial=True)
        