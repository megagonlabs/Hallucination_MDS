
import abc, copy, logging, tqdm
import os
import numpy as np
import entailment as ent
import contamination_base as cont
import utils_io

from collections import defaultdict
from typing import Any, Dict, List, override
from utils import generate_uuid


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class FilterBase(abc.ABC):
    @abc.abstractmethod
    def run(self, data):
        raise NotImplementedError("FilterBase.run() is not implemented")
        
    def __call__(self, data):
        return self.run(data)
    
    def get_filtered_response(self, response: Dict[str, Any], response_id: int|None=None) -> List[Dict[str, Any]]:
        response_filtered = response.get("response__parsed__filtered", None)
        
        if response_filtered is None:
            return [{
                "response_id": response.get("response_id", response_id),
                "pred_rank": pred_rank, 
                "pred_text": pred,
                "pred_uuid": generate_uuid({"text": pred}),
                "removed": False,
                "removed_reason": None,
            } for pred_rank, pred in enumerate(response["response__parsed"])]
        else:
            return [resp for resp in response_filtered if not resp["removed"]]
    
    def init_cache(self):
        logging.warning(f"{self.__class__.__name__} has no cache to initialize.")
    
    def save_cache(self):
        logging.warning(f"{self.__class__.__name__} has no cache to dump.")

    def save(self, filepath: str, data: dict):
        self.save_cache()

        if os.path.isdir(filepath):
            filepath = f"{filepath}/{self.__class__.__name__}.json"
        utils_io.to_json(filepath, data)
        
    
class RankingTrimmingFilter(FilterBase):
    def __init__(self, start: int=0, end: int|None=None, **kwargs):
        self.start = start
        self.end = end
        
    def is_in_valid_range(self, rank: int) -> bool:
        end = rank+1 if self.end is None else self.end
        return self.start <= rank < end

    def run(self, data: dict, **kwargs):
        data = copy.deepcopy(data)
        
        statistics = defaultdict(list)
        for response_id, response in tqdm.tqdm(enumerate(data["assignments"])):
            
            response_filtered = self.get_filtered_response(response, response_id=response_id)
            response["response__parsed__filtered"] = response_filtered
                        
            trimmed_counts = 0
            for bullet in response_filtered:
                if not self.is_in_valid_range(bullet["pred_rank"]):
                    bullet["removed"] = True
                    bullet["removed_reason"] = "ranking_trimming"
                    trimmed_counts += 1
                            
            n_preds = len(response_filtered)
            statistics["response_id"].append(response_id)
            statistics["num_preds"].append(n_preds)
            statistics["num_trimmed"].append(trimmed_counts)
            statistics["fraction"].append(trimmed_counts / n_preds)
            
        data["postprocessing__RankingTrimmingFilter"] = statistics
        data["postprocessing__RankingTrimmingFilter__kwargs"] = {"start": self.start, "end": self.end}
        
        logging.info("-------------------- Statistics ------------------")
        logging.info(f"Number of summaries that were trimmed: "
                     f"{np.count_nonzero(statistics["num_trimmed"])}"
                     f"/{len(statistics["num_trimmed"])}")
        logging.info(f"Avg trimmed bulletpoints per summary: "
                     f"{np.mean(statistics['fraction']):.2%}")
        return data 

class RedundantBulletsFilter(FilterBase):
    """Filters:"""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model: ent.BaseEntailment = ent.BaseEntailment.from_kwargs(prefix="paraphrase_preds", **kwargs)
        
    @override
    def init_cache(self):
        return self.model.init_prediction_cache()
       
    @override
    def save_cache(self):
        self.model.save_prediction_cache()
        
    @override
    def run(self, data: dict, **kwargs):
        data = copy.deepcopy(data)
        
        statistics = defaultdict(list)
        for response_id, response in tqdm.tqdm(enumerate(data["assignments"])):
            response_filtered = self.get_filtered_response(response, response_id=response_id)
            response["response__parsed__filtered"] = response_filtered
            response_preds = [r["pred_text"] for r in response_filtered]

            subtopic_uuid = response["subtopic_uuid"]
            subtopic = data["subtopics"][subtopic_uuid]["subtopic"]
            
            response_meaning_ids = ent.get_semantic_ids(strings_list=response_preds,
                                                        model=self.model,
                                                        subtopic=subtopic,
                                                        topic=data["topic"],
                                                        **kwargs)
            # ^Note: This is a list of integers, where each integer represents a unique semantic meaning.2
            # If two references are semantically equivalent, they will have the same integer.
            # If two references are not semantically equivalent, they will have different integers.
            # ------------------------------------------------------------------------------
            n_preds = len(response_preds)
            n_redundant = n_preds - len(set(response_meaning_ids))
            statistics["response_id"].append(response_id)
            statistics["num_preds"].append(n_preds)
            statistics["num_redundant"].append(n_redundant)
            statistics["fraction"].append(n_redundant / n_preds)
            statistics["response_meaning_ids"].append(response_meaning_ids)
            # ------------------------------------------------------------------------------             
            exists = {}
            for semantic_id, resp in zip(response_meaning_ids, response_filtered):
                if exists.get(semantic_id, None) is None:
                    exists[semantic_id] = True
                else:
                    resp["removed"] = True
                    resp["removed_reason"] = "redundant_bullets"
            # ^Note: given our finding that predictions made earlier on are usually correct,
            # we decide to keep the first prediction made for each unique semantic meaning.
            
        data["postprocessing__RedundantBulletsFilter"] = statistics
        data["postprocessing__RedundantBulletsFilter__kwargs"] = self.kwargs
        
        logging.info(
            f"Avg fraction of redundant bulletpoints: {np.mean(statistics['fraction']):.2f}" 
            + f" (± {np.std(statistics['fraction']):.2f})"
        )
        return data


class SubTopicNLIParaphraseFilter(FilterBase):
    """Filters:
    
    1. Load the data
    2. For each response, iterate the bulletpoints
    3. For each bullet point, determine whether it is a paraphrase of the subtopic
    or not.
    4. If it is a paraphrase, remove it.
    5. Save the data (cache)
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model: ent.BaseEntailment = ent.BaseEntailment.from_kwargs(**kwargs)
        
    @override
    def init_cache(self):
        return self.model.init_prediction_cache()
    
    @override
    def save_cache(self):
        self.model.save_prediction_cache()
    
    @override
    def run(self, data, **kwargs):
        data = copy.deepcopy(data)
        
        statistics = defaultdict(list)
        for response_id, response in tqdm.tqdm(enumerate(data["assignments"])):
            response_filtered = self.get_filtered_response(response, response_id=response_id)
            response["response__parsed__filtered"] = response_filtered
            response_preds = [r["pred_text"] for r in response_filtered]
            response_preds = [r[2:] if r.startswith("- ") else r for r in response_preds]
            
            subtopic_uuid = response["subtopic_uuid"]
            subtopic = data["subtopics"][subtopic_uuid]["subtopic"]
            
            response_meaning_ids = ent.get_semantic_ids(strings_list=[subtopic]+response_preds, 
                                                        model=self.model,
                                                        num_iter=1,
                                                        **kwargs)
            # ^Note: this is a list of integers. The first integer represents the subtopic, 
            # while the rest represent the bulletpoints. We only run it over one iteration
            # since we merely want to compare the subtopic w/ the bulletpoints.
            response_meaning_ids = [id == response_meaning_ids[0] for id in response_meaning_ids[1:]]
            
            # Metadata: 
            n_preds = len(response_preds)
            n_redundant = len([r for r in response_meaning_ids if r])
            
            statistics["response_id"].append(response_id)
            statistics["num_preds"].append(n_preds)
            statistics["num_redundant"].append(n_redundant)
            statistics["fraction"].append(n_redundant / n_preds)
            statistics["subtopic"].append(subtopic)
            statistics["subtopic_uuid"].append(subtopic_uuid)
            statistics["is_paraphrase_of_subtopic"].append(response_meaning_ids)

            for is_paraphrase, resp in zip(response_meaning_ids, response_filtered):
                if is_paraphrase:
                    logging.info(f"Paraphrase found!")
                    logging.info(f"-> Subtopic: {subtopic}")
                    logging.info(f"Pred: {resp['pred_text']}")
                    logging.info("--------------------------------------")
                    resp["removed"] = True
                    resp["removed_reason"] = "subtopic_paraphrase"
        
        data["postprocessing__SubtopicParaphraseFilter"] = statistics
        data["postprocessing__SubtopicParaphraseFilter__kwargs"] = self.kwargs
        
        logging.info(
            f"Avg fraction of bulletpoints (per summary) that paraphrase the subtopic:{np.mean(statistics['fraction']):.2f}" 
            + f"(± {np.std(statistics['fraction']):.3f})"
        )
        return data
    
    
class UnrelatedSubtopicFilter(FilterBase):
    """Filters:
    
    1. Load the data
    2. For each response, iterate the bulletpoints
    3. For each bullet point, use LLM as a judge to determine whether it is
    related to the queried subtopic or not.
    4. If it is a paraphrase, remove it.
    5. Save the data (cache)
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = ent.RelatedSubtopicLLM(prefix="unrelated_subtopic_llm", **kwargs)
        files = os.listdir(".")
        files = [f for f in files if "__unrelated_subtopic.txt" in f]

        self.temp_file = f"{len(files)}__unrelated_subtopic.txt"
        with open(self.temp_file, "w") as f:
            pass
        
        
    @override
    def init_cache(self):
        return self.model.init_prediction_cache()
    
    @override
    def save_cache(self):
        self.model.save_prediction_cache()

    @override
    def run(self, data, **kwargs):
        data = copy.deepcopy(data)
        
        statistics = defaultdict(list)
        for response_id, response in tqdm.tqdm(enumerate(data["assignments"])):
            subtopic_uuid = response["subtopic_uuid"]
            subtopic = data["subtopics"][subtopic_uuid]["subtopic"]
            
            response_filtered = self.get_filtered_response(response, response_id=response_id)
            response["response__parsed__filtered"] = response_filtered
            for bullet in response_filtered:
                is_same_subtopic, full_response = self.model.check_subtopic(
                    topic=data["topic"],
                    subtopic=subtopic,
                    insight=bullet["pred_text"],
                    **kwargs,
                )
                
                if is_same_subtopic == "no":
                    with open(self.temp_file, "a") as f:
                        f.write("----------------------------------------------------\n")
                        f.write(f"Response_id: {response_id}\n")
                        f.write(f"Subtopic: {subtopic}\n")
                        f.write(f"Pred: {bullet['pred_text']}\n")
                        f.write(f"LLM-judge response: {full_response}\n")
                    bullet["removed"] = True
                    bullet["removed_reason"] = "unrelated_subtopic"
                    bullet["removed_metadata"] = full_response
                    
                statistics["response_id"].append(response_id)
                statistics["subtopic_uuid"].append(subtopic_uuid)
                statistics["subtopic"].append(subtopic)
                statistics["is_same_subtopic"].append(is_same_subtopic == "yes")
                statistics["pred_text"].append(bullet["pred_text"])
                statistics["filter_response"].append(full_response)

        data["postprocessing__UnrelatedSubtopicFilter"] = statistics
        data["postprocessing__UnrelatedSubtopicFilter__kwargs"] = self.kwargs
        
        logging.info(
            f"Number of bulletpoints from different subtopic: "
            f"{len(statistics["is_same_subtopic"]) - np.sum(statistics['is_same_subtopic'])}" 
            + f" (out of {len(statistics["is_same_subtopic"])})"
        )
        return data
    

class ParaphraseSTSSubtopicFilter(FilterBase):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = ent.RelatedSubtopicLLM(prefix="paraphrase_llm", **kwargs)
        
        files = os.listdir(".")
        files = [f for f in files if "__subtopic_paraphrase.txt" in f]

        self.temp_file = f"{len(files)}__subtopic_paraphrase.txt"
        with open(self.temp_file, "w") as f:
            pass

    @override
    def init_cache(self):
        return self.model.init_prediction_cache()
    
    @override
    def save_cache(self):
        self.model.save_prediction_cache()

    @override
    def run(self, data, **kwargs):
        data = copy.deepcopy(data)
        
        statistics = defaultdict(list)
        for response_id, response in tqdm.tqdm(enumerate(data["assignments"])):
            subtopic_uuid = response["subtopic_uuid"]
            subtopic = data["subtopics"][subtopic_uuid]["subtopic"]
            
            response_filtered = self.get_filtered_response(response, response_id=response_id)
            response["response__parsed__filtered"] = response_filtered
            for bullet in response_filtered:
                is_paraphrase, full_response = self.model.check_subtopic(
                    topic=data["topic"],
                    subtopic=subtopic,
                    insight=bullet["pred_text"],
                    **kwargs,
                )
                
                if is_paraphrase == "yes":
                    with open(self.temp_file , "a") as f:
                        f.write("----------------------------------------------------\n")
                        f.write(f"Response_id: {response_id}\n")
                        f.write(f"Subtopic: {subtopic}\n")
                        f.write(f"Pred: {bullet['pred_text']}\n")
                        f.write(f"LLM-judge: {full_response}\n")
                    bullet["removed"] = True
                    bullet["removed_reason"] = "paraphrase_subtopic_sts"
                    bullet["removed_metadata"] = full_response
                    
                statistics["response_id"].append(response_id)
                statistics["subtopic_uuid"].append(subtopic_uuid)
                statistics["subtopic"].append(subtopic)
                statistics["is_paraphrase"].append(is_paraphrase == "yes")
                statistics["pred_text"].append(bullet["pred_text"])
                statistics["filter_response"].append(full_response)

        data["postprocessing__ParaphraseSTSSubtopicFilter"] = statistics
        data["postprocessing__ParaphraseSTSSubtopicFilter__kwargs"] = self.kwargs
        
        logging.info(
            f"Number of bulletpoints that are considered paraphrases: "
            f"{np.mean(statistics['is_paraphrase']):.2%}" 
            + f" ({np.sum(statistics['is_paraphrase'])} out of {len(statistics["is_paraphrase"])})"
        )
        return data
    


class SharedOnlyFilter(FilterBase):
    def __init__(self, shared_by_or_more=2, **kwargs):
        self.kwargs = kwargs
        self.model: cont.ContaminationBase = cont.ContaminationBase.from_kwargs(**kwargs)
        self.shared_by_or_more = shared_by_or_more
        
    @override
    def init_cache(self):
        self.model.init_prediction_cache()
    
    @override
    def save_cache(self):
        return self.model.save_prediction_cache()
    
    @override 
    def run(self, data, **kwargs):
        data = copy.deepcopy(data)
        uuid2docs = {d: dmeta["document_text"] for d, dmeta in data["documents"].items()}
        # preload documents in cache
        self.model.index_documents(uuid2docs)
        
        statistics = defaultdict(list)
        for response_id, response in tqdm.tqdm(enumerate(data["assignments"])):
            docs_uuid_ordered = [response["docs_uuids"][ord] for ord in response["docs_order"]]    

            response_filtered = self.get_filtered_response(response, response_id=response_id)
            response["response__parsed__filtered"] = response_filtered

            for bullet in response_filtered:
                matches = []
                for pos, doc_uuid in enumerate(docs_uuid_ordered):
                    if self.model.check_match(insight=bullet["pred_text"], doc_uuid=doc_uuid):
                        matches += [pos] 
                        
                if len(matches) < self.shared_by_or_more:
                    bullet["removed"] = True
                    bullet["removed_reason"] = "not_shared_by_enough"
                    bullet["removed_metadata"] = {
                        "num_matches": len(matches), 
                        "num_matches_expected": f">={self.shared_by_or_more}",
                        "matches": matches,
                    }
                    
                statistics["response_id"].append(response_id)
                statistics["pred_uuid"].append(bullet["pred_uuid"])
                statistics["pred_rank"].append(bullet["pred_rank"])
                statistics["pred_text"].append(bullet["pred_text"])
                statistics["num_documents"].append(len(docs_uuid_ordered))
                statistics["num_preds"].append(len(response_filtered))
                statistics["num_matches"].append(len(matches))
                statistics["is_shared"].append(len(matches) >= self.shared_by_or_more)
                statistics["matches"].append(matches)

        data["postprocessing__SharedOnlyFilter"] = statistics
        data["postprocessing__SharedOnlyFilter__kwargs"] = {"shared_by_or_more": self.shared_by_or_more}
        data["postprocessing__SharedOnlyFilter__kwargs"].update(self.kwargs)
        
        logging.info(
            f"Number of bulletpoints that are shared by {self.shared_by_or_more} or more documents: "
            f"{np.count_nonzero(statistics['is_shared'])} (out of {len(statistics["num_matches"])})")
        return data    

class LocalityFilter(FilterBase):
    """Filters:
    
    1. Load the data
    2. Determine longest common subsequence between the subtopic and the bulletpoints
        - Can use the same approach as the gpt-4 paper (break down pred into sequences of characters)
        - Map the index to the original documents and remove bulletpoints whose indexes are more than X chars away.
    """
    def run(self, data):
        raise NotImplementedError("MergePhrasesFilter.run() is not implemented")