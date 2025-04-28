from collections import defaultdict
from pydantic import BaseModel
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, override

import json, copy, traceback


class Result(BaseModel):
    coverage: str
    bullet_id: Union[int, str]
    insight_id: Optional[Union[int, str]] = None
    _insight_init: bool = False
    _insight_uuid: Optional[str] = None
    _bullet_uuid: Optional[str] = None
      
    def __str__(self):
        data = dict(coverage=self.coverage, bullet_id=self.bullet_id if self.bullet_id is not None else "NA")
        if self._insight_init:
            data.update(insight_id=self.insight_id if self.insight_id is not None else "NA")
        return json.dumps(data, sort_keys=True)
    
    @property
    def insight_str(self):
        return str(self.insight_id) if self.insight_id is not None else "NA"
    
    @property
    def bullet_str(self):
        return str(self.insight_id) if self.insight_id is not None else "NA"

    @property
    def is_covered(self):
        return self.coverage in ("FULL_COVERAGE", "PARTIAL_COVERAGE")

    @classmethod
    def pessimistic_merge(cls, res1: "Result", res2: "Result") -> str:
        if not res1.is_covered or not res2.is_covered:
            return "NO_COVERAGE"
        
        # We know that both of res1 and res2 are at least partially covered
        assert res1._bullet_uuid == res2._insight_uuid and res1._insight_uuid == res2._bullet_uuid, "UUID mismatchs."
        if res1.coverage == "PARTIAL_COVERAGE" or res2.coverage == "PARTIAL_COVERAGE":
            return "PARTIAL_COVERAGE"
        else:
            assert "FULL_COVERAGE" == res1.coverage == res2.coverage, "Coverage mismatch."
            return "FULL_COVERAGE"
  
    @override
    def model_post_init(self, __context: Any):
        def _int_from_str(text):
            if isinstance(text, str):
                if text.upper() in ("NA", "N/A", "NONE"):
                    text = None
                else:
                    text = int(text.strip())
            return text

        self.coverage = self.coverage.upper()
        self.bullet_id = _int_from_str(self.bullet_id)
        if self.insight_id is not None: 
            self.insight_id = _int_from_str(self.insight_id)
            self._insight_init = True

    def populate_insight_uuid(self, lst: List[str], offset: int):
        if self.insight_id is not None:
            if self.insight_id < 0 or self.insight_id - offset >= len(lst):
                raise RuntimeError(f"Invalid insight ID found: {self.insight_id - offset} for {len(lst)}.")
  
            self._insight_uuid = lst[self.insight_id - offset]        
            
    def populate_bullet_uuid(self, lst: List[str], offset: int):
        if self.bullet_id is not None:
            if self.bullet_id < 0 or self.bullet_id - offset >= len(lst):
                raise RuntimeError(f"Invalid bullet ID found: {self.bullet_id - offset} for {len(lst)}.")
            self._bullet_uuid = lst[self.bullet_id - offset]
                

class Results(BaseModel):
    results: List[Result]
    
    def __getitem__(self, idx: int) -> Result:
        return self.results[idx]

    def __len__(self):
        return len(self.results)
    
    def __iter__(self):
        return self.results.__iter__()
    
    def __next__(self):
        return self.results.__next__()
    
    @classmethod
    def from_json_str(self, json_str: str):
        json_obj = json.loads(json_str.strip())
        
        if json_obj.get("results") is not None:
            return Results(**json_obj)
        else:
            return Results(results=[json_obj])



class RawEval(BaseModel):
    response_id: int
    unique_id: List[str]
    prompt: str
    pred_insights_uuids: List[str]
    _data_uuid: str
    _eval_prompt_uuid: str
    response: Union[str, List[str]] = None
    
    # The fields below are defined for the batch evaluation
    custom_id: str = None
    batch_local_filepath: str = None
    openai__input_file_id: str = None
    openai__input_batch_id: str = None
    openai__input_batch_status: str = None
    openai__output_file_id: str = None
    openai__output_batch_id: str = None
    openai__last_updated: int = None
    openai__input_batch_error_file_id: str = None
    openai__input_batch_errors: str = None
    
    def __post_init__(self):
        if isinstance(self.response, str):
            self.response = [self.response]
    
    @classmethod
    def from_object(cls, obj: "RawEval") -> "RawEval":
        values = copy.deepcopy(obj.model_dump())
        return cls(**values)
    
    def extract_insights(self, response_str: str, uuids: Iterable[str], offset: int, logger) -> Dict[str, List[Results]]:
        """Extracts insights from a JSON response and groups them by UUID."""
        response = Results.from_json_str(response_str)
        
        grouped_results: Dict[str, List[Result]] = defaultdict(list)
        for result in response:
            # -----------------------------------------------------------
            # For multi-requests, the insight id will be the same as the 
            # unique_id of the evaluation (since there was a single
            # request for each unique_id)
            # -----------------------------------------------------------
            if not result._insight_init:
                assert len(uuids) == len(self.unique_id) == 1 and uuids[0] == self.unique_id[0],\
                    "Unexpected error for multi-request: {self.unique_id} vs {uuids}"
                result._insight_uuid = uuids[0]
                grouped_results[uuids[0]].append(result)
            # -----------------------------------------------------------
            # For single requests, the insight id will be determined
            # based on the list of uuids
            # -----------------------------------------------------------
            elif result.insight_id - offset >= len(uuids) or result.insight_id < 0:
                logger.error(f"Invalid insight ID found for single-request: {result.insight_id} in {result}. Skipping...")
                continue
            else:
                uuid = uuids[result.insight_id - offset]
                if uuid in grouped_results:
                    logger.warn(f"Duplicate UUID  {uuid} found: {response_str}. Adding it to the results...")
                grouped_results[uuid].append(result)
        return grouped_results
    
    def group_eval_by_uuid(self, offset: int, logger) -> Dict[Tuple[int, str], List[str]]:
        """Group the evaluation in terms of the unique IDs."""
        grouped_eval: Dict[Tuple[int, str], List[str]] = defaultdict(list)
        logger.debugging("Multiple unique IDs found. Grouping the evaluation...")
        for resp in self.response:
            try:
                insights: Dict[str, List[Result]] = self.extract_insights(resp, self.unique_id, offset, logger)
                for uuid, insight in insights.items():
                    grouped_eval[(self.response_id, uuid)].extend([str(r) for r in insight])
            except Exception as e:
                logger.error(f"Error while grouping the evaluation for response: {traceback.format_exception(e)}"
                             + f"\n\n- Unique IDs: {self.unique_id}"
                             + f"- Response: '{resp}'\n" + "*" * 80)
                
        return grouped_eval
    