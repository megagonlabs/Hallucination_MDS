from collections import defaultdict
from dataclasses import dataclass, field
import os
from typing import Any, Dict, List, Optional, Tuple, Union, override
from schema import RawEval, Result


import json
import utils, utils_io, utils_logging, utils_models

# Typing aliases
StrOrList = Union[str, List[str]]
# Default Constants
DEFAULT_NON_COVERAGE = """{"bullet_id": "NA", "coverage": "NO_COVERAGE"}"""

@dataclass
class Processor(utils_logging.LoggingMixin):
    """Post processing class.

    Faciliates the processing of the evaluation data
    in an asynchronous manner. Before running this
    class, the Evaluator class should have been run.
    """
    client: callable
    evals: Optional[Dict[str, Any]]

    cache: Optional[Dict[str, Any]] = field(default_factory=dict)
    eval_prompt_offset: Optional[int] = None
    bidirectional: Optional[int] = None
    n_responses: Optional[int] = None
    aggregate_fn: callable = utils.get_majority_vote
    merge_labels_fn: callable = Result.pessimistic_merge
    batch: bool = False

    def __post_init__(self, *args, **kwargs):
        # extract useful info from eval_kwargs
        self.eval_prompt_offset = self.evals["evaluation_kwargs"]["prompt_offset"]
        self.bidirectional = self.evals["evaluation_kwargs"]["bidirectional"]
        self.n_responses = self.evals["evaluation_kwargs"]["model_kwargs"].get("n", 1)

    @property
    def ref_evals(self) -> Dict[int, dict]:
        return {v["response_id"]: v for v in self.evals["evaluation_assignments"]["refs"]}
        
    @property
    def pred_evals(self) -> Dict[int, dict]:
        if not self.bidirectional:
            return {}
        
        return {v["custom_id"]: v for v in self.evals["evaluation_assignments"]["preds"]}

    @property
    def responses(self):
        return self.evals["assignments"]

    def group_eval_by_uuid(
        self, evals: List[RawEval]
    ) -> Dict[Tuple[int, str], List[str]]:
        if evals is None or len(evals) == 0:
            return {}

        grouped_evals = defaultdict(list)
        for eval in evals:
            resp_group = eval.group_eval_by_uuid(self.eval_prompt_offset, self)
            for unique_id, response in resp_group.items():
                grouped_evals[unique_id].extend(response)

        # Due to errors in the evaluation (if we're using a LLM-based eval)
        # It may happen that we have less responses than expected. In this
        # case, we will fill the missing responses with not-covered labels.
        # (assumption: failure to generate a label implies no coverage)
        for unique_id in grouped_evals.keys():
            n_missing = max(0, self.n_responses - len(grouped_evals[unique_id]))
            grouped_evals[unique_id].extend([DEFAULT_NON_COVERAGE] * n_missing)
        return grouped_evals

    def setup(self, *args, **kwargs):
        pass

    def load_evals(self) -> Tuple[List[RawEval], List[RawEval]]:
        """By default we load the raw evaluation results from the cache."""
        refs = [RawEval(**r) for r in self.ref_evals.values()]
        if self.bidirectional:
            preds = [RawEval(**r) for r in self.pred_evals.values()]
        else:
            preds = []
        return refs, preds

    def post_process(self):
        """Process the raw evaluations and populate the cache with the
        processed evaluations."""

        def _populate_label(label, uuid, ref_uuids, support_uuids) -> Result:
            label.populate_insight_uuid(ref_uuids, self.eval_prompt_offset)
            
            if not label._insight_init:
                label._insight_uuid = uuid

            assert label._insight_uuid == uuid,\
                f"UUID mismatch: {uuid} vs {label._insight_uuid}"
            try:
                label.populate_bullet_uuid(support_uuids, self.eval_prompt_offset)
            except RuntimeError as err:
                self.info(f"Wrong index found: {err}")
                label.coverage = "NO_COVERAGE"
            return label

        self.info("Beginning post_processing step...")
        eval_refs, eval_preds = self.load_evals()
        eval_idx_to_pred_insights = {er.response_id: er.pred_insights_uuids for er in eval_refs}
        eval_refs: Dict[Tuple[int, str], List[str]] = self.group_eval_by_uuid(eval_refs)
        # In case we have sampled multiple completions from the model
        # we'd like to ensure that the aggregated results are consistent.
        # Assumption: If the model does not generate a label for a specific
        # reference insight, we assume it does not cover it.
        # Implementation wise, we will fill the missing evaluations with
        # dummy NO_COVERAGE labels.
        eval_refs_label, eval_refs_metadata = {}, {}
        for (response_idx, ref_uuid), responses in eval_refs.items():
            label, metadata = self.aggregate_fn(responses)
            eval_refs_metadata[(response_idx, ref_uuid)] = metadata

            # Populate labels w/ uuid information
            label = Result(**json.loads(label))
            response = self.responses[response_idx]
            all_ref_uuids = response["all_insights_uuids"]
            all_pred_uuids = eval_idx_to_pred_insights[response_idx]
            label = _populate_label(label, ref_uuid, all_ref_uuids, all_pred_uuids)
            if label.is_covered:
                eval_refs_label[
                    (response_idx, label._insight_uuid, label._bullet_uuid)
                ] = label
        self.cache["refs_support"] = eval_refs_label
        self.cache["refs_metadata"] = eval_refs_metadata

        if self.bidirectional:
            assert eval_preds is not None, \
                "Bidirectional evaluation requested but no predictions found."
            eval_idx_to_pred_insights = {er.response_id: er.pred_insights_uuids for er in eval_preds}
            eval_preds: Dict[Tuple[int, str], List[str]] = self.group_eval_by_uuid(
                eval_preds
            )
            eval_preds_label, eval_preds_metadata = {}, {}
            for (response_idx, pred_uuid), responses in eval_preds.items():
                label, metadata = self.aggregate_fn(responses)
                eval_preds_metadata[(response_idx, pred_uuid)] = metadata

                # Populate labels w/ uuid information
                label = Result(**json.loads(label))
                response = self.responses[response_idx]
                all_ref_uuids = response["all_insights_uuids"]
                all_pred_uuids = eval_idx_to_pred_insights[response_idx]
                label = _populate_label(label, pred_uuid, all_pred_uuids, all_ref_uuids)

                if label.is_covered:
                    eval_preds_label[
                        (response_idx, label._bullet_uuid, label._insight_uuid)
                    ] = label

            self.cache["preds_support"] = eval_preds_label
            self.cache["preds_metadata"] = eval_preds_metadata
        self.info("Post-processing step completed...")

    def merge_results(self):
        """Merges the results of the evaluations.

        It assumes that ``self.post_process()`` was executed previously.
        It will look for the following data structures in the cache:

         * refs_support: dict of tuples (response_idx, ref_uuid, pred_uuid) -> Result
            The `refs_support` data structure shows for each reference present in the
            response, what is the predicted insight that covers it. To unify the
            interface, we've previously grouped the labels by each reference and
            corresponding prediction uuid.

         * preds_support: dict of tuples (response_idx, ref_uuid, pred_uuid) -> Result
            If we're using the bidirectional metric, we will have a similar structure
            for the predictions, where for each prediction associated with a response,
            we will have information about which reference in the response it is
            covering it.

        Using these data structures (or refs_support only if bidirectional is False),
        we will merge the different labels into a single label per (ref_uuid, pred_uuid).
        We will use a worse-case assumption to merge two labels, where the worse-to-best
        order is "NO_COVERAGE" > "PARTIAL_COVERAGE" > "FULL_COVERAGE". In other words,
        the only scenario where we will have a FULL_COVERAGE label is if both results
        in refs_support and preds_support are associated with a FULL_COVERAGE label.

        As a result, we will put in the cache the following data structure:
         * final_evals: dict of tuple (response_idx, ref_uuid, pred_uuid) -> str
            where the string is the coverage label associated with a specific
            reference and prediction in response `response_idx`.
        """
        if not self.bidirectional:
            self.info("Skipping merging results for uni directional metric...")
            self.cache["final_evals"] = {
                k: v.coverage for k, v in self.cache["refs_support"].items()
            }
            return

        ref_supports = self.cache["refs_support"]
        pred_supports = self.cache["preds_support"]

        self.info(
            f"Merging results for bidirectional metric:"
            f"\n-> #References: {len(ref_supports)}"
            f"\n-> #Predicted Insights: {len(pred_supports)}"
        )

        merged_labels: Dict[int, Dict[Tuple[str, str], str]] = defaultdict(dict)
        common_unique_ids = set(ref_supports).intersection(set(pred_supports))
        # ^Note: if there is no match, then we assume there's no valid coverage
        # for that label. In a bidirectional metric, we'd like to have labels
        # in both ways to consider the coverage valid. As such, merged_labels
        # will only have the final valid labels as determined by the metric.
        for response_idx, ref_uuid, pred_uuid in common_unique_ids:
            ref_label = ref_supports[(response_idx, ref_uuid, pred_uuid)]
            pred_label = pred_supports[(response_idx, ref_uuid, pred_uuid)]

            merged_label: str = self.merge_labels_fn(ref_label, pred_label)
            merged_labels[(response_idx, ref_uuid, pred_uuid)] = merged_label

        self.cache["final_evals"] = merged_labels
        if len(merged_labels) == 0:
            self.critical("No common unique IDs found for merging.")
        self.info("Merging step completed.")

    def populate_metadata(self):
        """Populate the metadata for the final evaluations."""

        def to_list(values: Dict[Tuple[int, str, str], str]):
            lst = []
            for (response_idx, ref_uuid, pred_uuid), label in values.items():
                lst.append(
                    {
                        "response_idx": response_idx,
                        "ref_uuid": ref_uuid,
                        "pred_uuid": pred_uuid,
                        "coverage": label.coverage if isinstance(label, Result) else label,
                    }
                )
            return lst

        def metadata_to_list(values: Dict[Tuple[int, str], Dict[str, int]]):
            lst = []
            for (response_idx, uuid), metadata in values.items():
                lst.append(
                    {"response_idx": response_idx, "uuid": uuid, "counter": metadata}
                )
            return lst

        self.evals["evaluation_processor_kwargs"] = {
            "aggregate_fn": utils.get_fully_qualified_function_name(self.aggregate_fn),
            "merge_labels_fn": utils.get_fully_qualified_function_name(
                self.merge_labels_fn
            ),
            "batch": self.batch,
        }

        ref_coverage = to_list(self.cache["refs_support"])
        ref_metadata = metadata_to_list(self.cache["refs_metadata"])

        if self.bidirectional:
            pred_coverage = to_list(self.cache["preds_support"])
            pred_metadata = metadata_to_list(self.cache["preds_metadata"])
        else:
            pred_coverage = {}
            pred_coverage = {}

        labels = to_list(self.cache["final_evals"])
        metric = {
            "ref_coverage": ref_coverage,
            "ref_metadata": ref_metadata,
            "pred_coverage": pred_coverage,
            "pred_metadata": pred_metadata,
            "labels": labels,
        }
        metric_name = "bidirectional" if self.bidirectional else "unidirectional"
        self.evals["evaluation_assignments"][f"metric__{metric_name}"] = metric

    def save(self, output_path: str):
        self.info(f"Saving post_processed data to {output_path}")
        utils_io.to_json(output_path, self.evals)
        self.info(f"Saved!")


@dataclass
class BatchProcessor(Processor):
    """Post processing class tailored for loading batches of the data."""
    open_ai_batch_kwargs: Dict[str, str] = field(
        default_factory=lambda: {
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        }
    )
    evals_dir: Optional[str] = None
    blocking: bool = True
    output_dir: str = "."
    batch: bool = True

    @property
    def evals_refs_dir(self):
        return f"{self.evals_dir}/batches__refs"

    @property
    def evals_preds_dir(self):
        return f"{self.evals_dir}/batches_preds"

    @property
    def output_refs_dir(self):
        return f"{self.output_dir}/batches__refs"

    @property
    def output_preds_dir(self):
        return f"{self.output_dir}/batches_preds"

    @property
    def ref_evals(self) -> Dict[str, dict]:
        return {v["custom_id"]: v for v in self.evals["evaluation_assignments"]["refs"].values()}
        
    @property
    def pred_evals(self) -> Dict[str, dict]:
        if not self.bidirectional:
            return {}
        
        return {v["custom_id"]: v for v in self.evals["evaluation_assignments"]["preds"].values()}
  
    def setup(self, eval_path: str, output_path: str):
        self.info("Setting up the BatchProcessor...")
        if eval_path.endswith("data.json"):
            self.evals_dir = eval_path.rpartition("/")[0]
        elif eval_path.endswith(".json"):
            self.evals_dir = eval_path.rpartition(".json")[0]
        elif os.path.isdir(eval_path):
            self.evals_dir = eval_path
        elif eval_path.endswith(".jsonl"):
            self.evals_dir = eval_path.rpartition("/")[0].rpartition("/")[0]
        else:
            self.evals_dir = eval_path
            
        assert os.path.isdir(self.evals_refs_dir), \
            f"Invalid refs directory: no batches found in {self.evals_refs_dir}"
        if self.bidirectional:
            assert os.path.isdir(self.evals_preds_dir), \
                f"Invalid preds directory: no batches found in {self.evals_preds_dir}"

        self.output_dir = output_path.rpartition(".")[0]
        os.makedirs(self.output_dir, exist_ok=True)
        self.info(f"Created output directory: {self.output_dir}")
        self.info("Setup completed...")

    def _check_batch_status(self, batch_id: str) -> bool:
        batch = utils_models.retrieve_batch(self.client, batch_id, blocking=self.blocking)
        if batch.status in utils_models.FAILED_BATCH_STATUS:
            self.error(f"Batch {batch_id} failed with status {batch.status}. Skipping...")
            return False, batch
        return True, batch

    def retrieve_batch_(self, evals: List[RawEval], tag: str) -> Dict[str, List[RawEval]]:
        """Retrieve the batch from the OpenAI API."""
        
        # 1. Group evals by batch file
        batch2evals: Dict[str, List[RawEval]] = defaultdict(list)
        for eval in evals: 
            batch2evals[eval.openai__input_batch_id].append(eval)
    
        # 2. Determine which batches were completed and which were not
        for batch_id, batch_evals in batch2evals.items():
            batch_completed, batch = self._check_batch_status(batch_id)

            self.info(f"Batch {batch_id} status: {batch}")
            self.info(f"Batch {batch_id} evals: {len(batch_evals)}")
            
            base_dir = self.output_refs_dir if tag == "refs" else self.output_preds_dir
            fp = f"{base_dir}/{batch_id}.jsonl"
            utils_models.dump_batch(self.client, batch, fp)
            
            # Download successful batches
            if batch_completed:
                for eval in batch2evals[batch_id]:
                    eval.batch_local_filepath = fp
                    eval.openai__output_batch_id = batch.id
                    eval.openai__output_file_id = batch.output_file_id
        return batch2evals
        
    def populate_responses_(self, evals: List[RawEval], responses: List[dict]):
        evals: Dict[str, RawEval] = {e.custom_id: e for e in evals}
        responses: Dict[str, dict] = {r["custom_id"]: r for r in responses}
        
        for custom_id in responses:
            eval = evals[custom_id]
            eval.response = responses[custom_id]["text"]
            
    @override
    def load_evals(self) -> Tuple[List[RawEval]]:
        refs = [RawEval(**r) for r in self.ref_evals.values()]
        
        # Step 1. Retrieve the batches that are completed
        batch2refs: Dict[str, List[RawEval]] = self.retrieve_batch_(refs, "refs")
        # ^note: mapping between local batch filepath and the evaluations
        # (the invalid batches will be stored in the "_unsuccessful_" key)
        
        # Step 2. Populate the evaluations w/ responses
        populated_refs = []
        for batch_id, evals in batch2refs.items():
            output_fp = f"{self.output_refs_dir}/{batch_id}.jsonl"
            if os.path.exists(output_fp):
                self.info(f"Populating {len(evals)} refs responses for batch {batch_id}...")
                responses: List[dict] = utils_io.read_jsonlines(output_fp, mode="r")                
                responses: List[dict] = map(utils_models.parse_batch_response, responses)
                self.populate_responses_(evals, responses)
            populated_refs.extend(evals)
        
        # Step 3. Update self.evals with the populated refs
        eval_refs = {e.custom_id: e.model_dump() for e in populated_refs}
        for eval_id, eval_dump in eval_refs.items():
            self.evals["evaluation_assignments"]["refs"][eval_id] = eval_dump
        
        populated_preds = []
        if self.bidirectional:
            preds = [RawEval(**r) for r in self.pred_evals.values()]
            # Step 1. Retrieve the batches that are completed
            batch2preds: Dict[str, List[RawEval]] = self.retrieve_batch_(preds, "preds")
            # ^note: mapping between local batch filepath and the evaluations
            # (the invalid batches will be stored in the "_unsuccessful_" key)
            
            # Step 2. Populate the evaluations w/ responses
            for batch_id, evals in batch2preds.items():
                output_fp = f"{self.output_preds_dir}/{batch_id}.jsonl"
                if os.path.exists(output_fp):
                    self.info(f"Populating {len(evals)} preds responses for batch {batch_id}...")
                    responses: List[dict] = utils_io.read_jsonlines(output_fp, mode="r")                
                    responses: List[dict] = map(utils_models.parse_batch_response, responses)
                    self.populate_responses_(evals, responses)
                populated_preds.extend(evals)
            
            eval_preds = {e.custom_id: e.model_dump() for e in populated_preds}
            for eval_id, eval_dump in eval_preds.items():
                self.evals["evaluation_assignments"]["preds"][eval_id] = eval_dump

        return populated_refs, populated_preds
