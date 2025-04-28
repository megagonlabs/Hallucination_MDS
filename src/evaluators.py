import json, tqdm, os
import utils, utils_io, utils_logging, utils_models

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, override

# Type aliases
StrOrList = Union[str, List[str]]
JSONStr = str


@dataclass
class Evaluator(ABC, utils_logging.LoggingMixin):
    """LM-based evaluator abstract class.

    The goal is to use this class to determine the faithfulness of a
    candidate summary. The evaluation is based on the coverage
    of the insights. In resemblance of NLI-based evaluators,
    this evaluator is bidirectional, which means that a predicted
    insight is only considered covered if it is supported by the
    reference insights and if the same reference insight is also
    supported by the predicted insight.
    """
    client: callable
    eval_kwargs: Dict[str, Any]
    eval_prompt: str = ""
    eval_prompt_offset: int = 1
    bidirectional: bool = True
    prompt_filepath: str = "." # not used...
    
    _num_examples: int = None
    _data: dict = field(default_factory=dict)
    _data_uuid: str = ""
    _eval_prompt_uuid: str = ""
    _cache: dict = field(default_factory=dict)

    @property
    def responses(self):
        if self._num_examples is None:
            return self._data["assignments"]
        return self._data["assignments"][: self._num_examples]

    def ref_insights(self, uuid):
        return self._data["insights"][uuid]

    def pred_insights_uuids(self, preds: List[str]) -> Dict[str, str]:
        return {utils.generate_uuid({"text": pred}): pred for pred in preds}
    
    @classmethod
    def parse_summaries(cls, response: dict) -> List[str]:
        """Parse the responses into a list of strings."""
        bullets = response.get("response__parsed", None)
        if bullets is None:
            bullets = [b.strip() for b in utils.bullet_processor(response["response"])]
        return bullets

    def setup(self, output_path: Optional[str] = None):
        self.info("Starting setup...")
        if output_path is not None:
            self.info(f"Output directory is located at: {output_path}")
            self.output_path = output_path

        self.info(f"Creating results directory at {self.output_path}")
        output_dir = os.path.dirname(self.output_path)
        os.makedirs(output_dir, exist_ok=True)
        self.info("Setup completed...")

    @abstractmethod
    def prepare_prompts(self, response_idx: int, response: dict):
        raise NotImplementedError("Subclasses must implement this method.")

    def prepare_inputs_for_evaluation(self):
        """Parses the responses and creates the necessary prompts.

        The prompts are put into the cache for later use.
        """
        self.info(f"Preparing {len(self.responses)} inputs for evaluation.")

        self._cache["prompts__refs_cov"] = defaultdict(dict)
        self._cache["prompts__preds_cov"] = defaultdict(dict)
        self._cache["pred_insights_uuids"] = defaultdict(dict)
        for response_idx, response in tqdm.tqdm(
            enumerate(self.responses), desc="prepare_inputs_for_evaluation"
        ):
            response["response__parsed"] = self.parse_summaries(response)
            self._cache["prompts__refs_cov"][response_idx] = []
            self._cache["prompts__preds_cov"][response_idx] = []
            self._cache["pred_insights_uuids"][response_idx] = []
            self.prepare_prompts(response_idx, response)

        if not self.bidirectional:
            self._cache.pop("prompts__preds_cov", None)
        self.info("Completed preparation of inputs for evaluations...")

    def _evaluate(self, inputs: List[Tuple[int, Union[str, List[str]], str]], tag: str):
        self._cache[f"raw_eval__{tag}"] = defaultdict(dict)
        for response_idx, unique_id, prompt in tqdm.tqdm(
            inputs, desc=f"Evaluation of {tag}"
        ):
            response = utils_models.generate_chat(
                client=self.client, prompt=prompt, **self.eval_kwargs
            )
            self._cache[f"raw_eval__{tag}"][(response_idx, unique_id)] = response

    def _evaluate_complete(self):
        pass

    def evaluate(self):
        """Consumes the prompts and obtains the responses, associating them with the
        same keys.

        Notes
        -----
        It expects to find the following structures in cache:
         * prompts__refs_cov: dict of lists of tuples ((unique_id, prompt),
         where the unique_id can is a tuple of a string or multiple strings.
         We'll have a prompt per reference insight. The key is the response
         index and should have a direct match to the self.responses[idx].

         * prompts__preds_cov: dict of lists of tuples (unique_ids, prompt),
         where the unique_id can be a tuple of strings. This is only
         defined if we're using the bidirectional metric. The key is the
         response index and should have a direct match to the
         self.responses[idx]. The unique ids represents the uuids of one or
         more predicted insights for the response idx.

         * pred_insights_uuids: dict of lists of strings, where the strings
            are the UUIDs of the predicted insights. The key is the response
            index and should have a direct match to the self.responses[idx].

        Modifications
        -------------
        The method will populate the cache with the following structures:
         * raw_eval__refs: dict of str, where the keys are tuples
            <response_idx, unique_id>. The response will be unprocessed.
            The user should call `post_process` after calling evaluate.

         * raw_eval__preds: dict of str, where the keys are tuples
            <response_idx, unique_id>. The response will be unprocessed.
            The user should call `post_process` after calling evaluate.
        """
        self.info("Beginning evaluation step...")

        evaluations = []
        for response_idx, prompts in self._cache["prompts__refs_cov"].items():
            for unique_id, prompt in prompts:
                evaluations.append((response_idx, unique_id, prompt))
        self._evaluate(evaluations, "refs")

        if self.bidirectional:
            evaluations = []
            for response_idx, prompts in self._cache["prompts__preds_cov"].items():
                for unique_id, prompt in prompts:
                    evaluations.append((response_idx, unique_id, prompt))
            self._evaluate(evaluations, "preds")

        self._evaluate_complete()
        self.info("Evaluation step completed...")

    def populate_metadata(self):
        def __collect_metadata(prompts, raw_evals) -> List[Dict[str, Any]]:
            metadata = []
            for response_idx, prompts in prompts.items():
                for unique_id, prompt in prompts:
                    metadata.append(
                        {
                            "response_id": response_idx,
                            "unique_id": list(unique_id),
                            "prompt": prompt,
                            "response": raw_evals[(response_idx, unique_id)],
                            "pred_insights_uuids": self._cache["pred_insights_uuids"][
                                response_idx
                            ],
                            "_data_uuid": self._data_uuid,
                            "_eval_prompt_uuid": self._eval_prompt_uuid,
                        }
                    )
            return metadata

        self.info("Populating metadata population...")

        # Add evaluation kwargs metadata
        self._data["evaluation_kwargs"] = {
            "batch": False,
            "prompt": self.eval_prompt,
            "prompt_uuid": self._eval_prompt_uuid,
            "prompt_offset": self.eval_prompt_offset,
            "model": self.eval_kwargs["model"],
            "model_kwargs": self.eval_kwargs,
            "bidirectional": self.bidirectional,
            "_generation_data_uuid_": self._data_uuid,
            "_num_examples": self._num_examples,
        }
        refs_metadata = __collect_metadata(
            self._cache["prompts__refs_cov"], self._cache["raw_eval__refs"]
        )
        self._data["evaluation_assignments"] = {"refs": refs_metadata}
        if self.bidirectional:
            preds_metadata = __collect_metadata(
                self._cache["prompts__preds_cov"], self._cache["raw_eval__preds"]
            )
            self._data["evaluation_assignments"]["preds"] = preds_metadata
        self.info("Completed metadata population...")

    def save(self, output_path: str):
        self.info(f"Saving evaluation data to {output_path}")
        utils_io.to_json(output_path, self._data)
        self.info(f"Saved!")


@dataclass
class SingleRequestEvaluator(Evaluator):
    @override
    def prepare_prompts(self, response_idx: int, response: dict) -> List[str]:
        """Create one request per response."""

        def _get_prompt(inputs: List[str], bullets: List[str]) -> str:
            format_kwargs = dict(sort_keys=True, indent=1)
            # structure the inputs into a JSON string
            inputs: List[dict] = [
                {"insight_id": i + self.eval_prompt_offset, "text": input.strip()}
                for i, input in enumerate(inputs)
            ]
            inputs_str = json.dumps({"insights": inputs}, **format_kwargs)

            # structure the support insights into bullet points
            bullets: List[dict] = [
                {"bullet_id": i + self.eval_prompt_offset, "text": bullet.strip()}
                for i, bullet in enumerate(bullets)
            ]
            bullets_str = json.dumps({"bullets": bullets}, **format_kwargs)

            prompt = self.eval_prompt.replace("{{true_insights}}", inputs_str)
            prompt = prompt.replace("{{pred_insights}}", bullets_str)
            assert (
                "{{true_insights}}" not in prompt
                and "{{true_insight}}" not in prompt
                and "{{pred_insights}}" not in prompt
            ), f"Prompt not fully populated: {prompt}"
            assert len(prompt.strip()) > 0, "Prompt cannot be empty."
            return prompt

        # One way coverage (start w/ ref -> pred)
        all_ref_uuids = response["all_insights_uuids"]
        all_ref_texts: Dict[str, str] = {
            uuid: self.ref_insights(uuid)["insight"] for uuid in all_ref_uuids
        }
        all_pred_insights: Dict[str, str] = self.pred_insights_uuids(
            response["response__parsed"]
        )
        self._cache["pred_insights_uuids"][response_idx] = list(
            all_pred_insights.keys()
        )

        if len(all_ref_uuids) > 10 or len(all_pred_insights) > 10:
            self.warn(
                f"Too many insights to evaluate in a single prompt: "
                f"{len(all_ref_uuids)} refs, {len(all_pred_insights)} preds."
            )

        prompt = _get_prompt(all_ref_texts.values(), all_pred_insights.values())
        self._cache["prompts__refs_cov"][response_idx].append(
            (tuple(all_ref_uuids), prompt)
        )

        if response_idx % 10 == 0:
            self.debugging(
                f"*refs -> preds* prompt for response {response_idx}:\n{prompt}"
            )

        if self.bidirectional:
            # The other way coverage (start w/ pred -> ref)
            prompt = _get_prompt(all_pred_insights.values(), all_ref_texts.values())
            self._cache["prompts__preds_cov"][response_idx].append(
                (tuple(all_pred_insights.keys()), prompt)
            )

            if response_idx % 10 == 0:
                self.debugging(
                    f"*preds -> refs* prompt for response {response_idx}:\n{prompt}"
                )


@dataclass
class MultiRequestEvaluator(Evaluator):
    @override
    def prepare_prompts(self, response_idx: int, response: dict) -> List[str]:
        """For every reference insight in the response, create a prompt.

        If the evaluator is bidirectional, create the prompts to get
        the support from the predictions to the references as well."""

        def _get_prompt(input: str, support: List[str]) -> str:
            bullet_dicts = [
                {"bullet_id": i + self.eval_prompt_offset, "text": bullet.strip()}
                for i, bullet in enumerate(support)
            ]
            bullets_str = json.dumps(
                {"bullets": bullet_dicts}, sort_keys=True, indent=1
            )

            # Populate the prompt with the true and predicted insights
            prompt = self.eval_prompt.replace("{{true_insight}}", input)
            prompt = prompt.replace("{{pred_insights}}", bullets_str)
            assert (
                "{{true_insight}}" not in prompt and "{{pred_insights}}" not in prompt
            ), f"Prompt not fully populated: {prompt}"
            assert len(prompt.strip()) > 0, "Prompt cannot be empty."
            return prompt

        # Auxiliary variables
        all_ref_uuids = response["all_insights_uuids"]
        all_pred_insights: Dict[str, str] = self.pred_insights_uuids(
            response["response__parsed"]
        )
        self._cache["pred_insights_uuids"][response_idx] = list(
            all_pred_insights.keys()
        )

        for ref_uuid in all_ref_uuids:
            ref_insight = self.ref_insights(ref_uuid)

            prompt = _get_prompt(ref_insight["insight"], all_pred_insights.values())
            assert prompt != self.eval_prompt, "Prompt not populated: {prompt}"
            self._cache["prompts__refs_cov"][response_idx].append(((ref_uuid,), prompt))

        if not self.bidirectional:
            return

        all_ref_insights = [self.ref_insights(uuid)["insight"] for uuid in all_ref_uuids]
        for pred_uuid, pred_insight in all_pred_insights.items():
            prompt = _get_prompt(pred_insight, all_ref_insights)
            assert prompt != self.eval_prompt, "Prompt not populated: {prompt}"
            self._cache["prompts__preds_cov"][response_idx].append(
                ((pred_uuid,), prompt)
            )


@dataclass
class BatchEvaluator(Evaluator):
    open_ai_batch_kwargs: Dict[str, str] = field(
        default_factory=lambda: {
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        }
    )
    blocking: bool = True
    output_dir: str = "."

    @property
    def output_refs_dir(self):
        return f"{self.output_dir}/batches__refs"

    @property
    def output_preds_dir(self):
        return f"{self.output_dir}/batches_preds"

    @override
    def setup(self, output_path: str):
        self.info("Starting batch setup...")
        assert output_path is not None, "Output path must be provided."

        if not os.path.isdir(output_path):
            self.info("Provided path is a file. Creating the directory...")
            output_path = output_path.rpartition(".json")[0]

        self.output_dir = output_path
        self.info(f"Creating results directory at {self.output_dir}")
        os.makedirs(self.output_dir, exist_ok=True)

        self.debugging("Creating dir for reference batches...")
        os.makedirs(self.output_refs_dir, exist_ok=True)

        if self.bidirectional:
            self.debugging("Creating dir for batch_predictions too...")
            os.makedirs(self.output_preds_dir, exist_ok=True)
        self.info("Setup completed...")

    @override
    def _evaluate(self, inputs: List[Tuple[int, Union[str, List[str]], str]], tag: str):
        """Uses the batch API to run the evaluation."""
        self.info(f" ----------------- Creating data for {tag} batch -----------------")
        batch: List[Dict[str, Any]] = []

        self._cache[f"batch_mapping__{tag}"] = defaultdict(dict)
        for response_idx, unique_id, prompt in tqdm.tqdm(inputs, desc="Evaluation"):
            request_id: str = utils.generate_uuid({"uuid": (
                unique_id if isinstance(unique_id, str) else "_".join(unique_id)
            )})
            request_id = f"{tag}__{response_idx}__{request_id}"

            payload: Dict[str, Any] = utils_models.create_jsonl_batch_format(
                prompt, **self.eval_kwargs
            )
            payload["custom_id"] = request_id
            batch.append(payload)

            self._cache[f"batch_mapping__{tag}"][request_id] = [
                response_idx,
                unique_id,
                prompt,
            ]

        self._cache[f"batch__{tag}"] = batch
        self.info(f"-> Completed... Number of total requests to evaluate: {len(batch)}")
        # Note: We do not submit the batch here as we want to persist the
        # metadata first and thus avoid wasting computation in case of a
        # failure.

    @override
    def _evaluate_complete(self):
        """Dump metadata here and submit the batch to OpenAI."""
        self.info("Submitting batches to OpenAI...")

        def create_batch(group_batch, output_dir: str, tag: str):
            metadata = {}
            for batch_id, batch in group_batch.items():
                path = f"{output_dir}/{tag}__batch_{batch_id}.jsonl"
                utils_io.to_jsonlines(path, batch)
                open_ai_file, open_ai_batch_id = utils_models.create_batch_openai(
                    self.client, path, **self.open_ai_batch_kwargs
                )
                metadata[batch_id] = {
                    "input_file": open_ai_file,
                    "input_batch": open_ai_batch_id,
                    "local_filepath": path,
                }
            return metadata

        def update_metadata_(docs2batch, batches_metadata, tag: str):
            for doc_id, batch_id in docs2batch.items():
                custom_id = self._cache[f"batch__{tag}"][doc_id]["custom_id"]
                batch_metadata = batches_metadata[batch_id]

                metadata = self._cache[f"batch_mapping__{tag}"][custom_id]
                metadata = {
                    "response_id": metadata[0],
                    "unique_id": list(metadata[1]),
                    "prompt": metadata[2],
                    "batch_local_filepath": batch_metadata["local_filepath"],
                    "openai__input_file_id": batch_metadata["input_file"].id,
                    "openai__input_batch_id": batch_metadata["input_batch"].id,
                    "openai__input_batch_status": batch_metadata["input_batch"].status,
                }
                if batch_metadata["input_batch"].status in utils_models.FAILED_BATCH_STATUS:
                    self.warn(
                        f"Batch {batch_metadata["input_batch"].id} failed with status: {batch_metadata['input_batch'].status}"
                    )
                    if batch_metadata["input_batch"].errors is not None:
                        self.debugging(f"Errors: {batch_metadata["input_batch"].errors}")

                    metadata["openai__input_batch_error_file_id"] = batch_metadata[
                        "input_batch"
                    ].error_file_id
                    metadata["openai__input_batch_errors"] = batch_metadata["input_batch"].errors
                
                # Update the metadata! Modifying!! FIXME
                self._cache[f"batch_mapping__{tag}"][custom_id] = metadata

        # step 1. group the requests into batches of 100 mb
        # ------------------------------------------------------------
        grouped_refs_batches, refs2batch = utils_io.get_jsonlines_chunks_of(
            self._cache["batch__refs"], size_in_mb=95, max_lines=50_000, return_doc_to_group_map=True
        )
        refs_batch_metadata: Dict[int, dict] = create_batch(grouped_refs_batches, self.output_refs_dir, tag="refs")
        update_metadata_(refs2batch, refs_batch_metadata, "refs")

        if self.bidirectional:
            grouped_pred_batches, preds2batch = utils_io.get_jsonlines_chunks_of(
                self._cache["batch__preds"], size_in_mb=95, max_lines=50_000, return_doc_to_group_map=True
            )
            batch_pred_metadata = create_batch(
                grouped_pred_batches, self.output_preds_dir, tag="preds"
            )
            update_metadata_(preds2batch, batch_pred_metadata, "preds")

        self.info(f"Submitted batch files to OpenAI.")

    @override
    def populate_metadata(self):
        def __collect_metadata(raw_evals) -> List[Dict[str, Any]]:
            for custom_id, metadata in raw_evals.items():
                response_idx = metadata["response_id"]
                metadata.update(
                    {
                        "custom_id": custom_id,
                        "pred_insights_uuids": self._cache["pred_insights_uuids"][
                            response_idx
                        ],
                        "_data_uuid": self._data_uuid,
                        "_eval_prompt_uuid": self._eval_prompt_uuid,
                    }
                )
            return raw_evals

        self._data["evaluation_kwargs"] = {
            "prompt": self.eval_prompt,
            "prompt_uuid": self._eval_prompt_uuid,
            "prompt_offset": self.eval_prompt_offset,
            "model": self.eval_kwargs["model"],
            "model_kwargs": self.eval_kwargs,
            "bidirectional": self.bidirectional,
            "_generation_data_uuid_": self._data_uuid,
            "_num_examples": self._num_examples,
            "batch": True,
            "open_ai_batch_kwargs": self.open_ai_batch_kwargs,
        }

        refs_metadata = __collect_metadata(self._cache["batch_mapping__refs"])
        self._data["evaluation_assignments"] = {"refs": refs_metadata}

        if self.bidirectional:
            preds_metadata = __collect_metadata(self._cache["batch_mapping__preds"])
            self._data["evaluation_assignments"]["preds"] = preds_metadata

        self.info("Completed metadata population...")
        
    def save(self, output_path: str):
        self.info(f"Saving evaluation data to {self.output_dir}/data.json")
        utils_io.to_json(self.output_dir + "/data.json", self._data)
        self.info(f"Saved!")


class SingleRequestBatch(BatchEvaluator, SingleRequestEvaluator):
    pass


class MultiRequestBatch(BatchEvaluator, MultiRequestEvaluator):
    pass

