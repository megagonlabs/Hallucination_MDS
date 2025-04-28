"""Implement semantic entropy."""
import os
import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils import generate_uuid, create_object_from_class_string
from utils_mixins import CacheMixin, LLMJudgeMixin

class BaseEntailment(CacheMixin):
    def __init__(self, entailment_mapping: Dict[str, int] | None=None, strict_entailment: bool=False, entailment_threshold: float | None=None, **kwargs):
        super().__init__(**kwargs)
        if entailment_mapping is None:
            entailment_mapping = {
                'contradiction': 0,
                'neutral': 1,
                'entailment': 2,
            }
        self.entailment_mapping = entailment_mapping
        self.entailment_classes = list(entailment_mapping.keys())
        self.strict_entailment = strict_entailment
        self.entailment_threshold = entailment_threshold

    def satisfy_threshold(self, prob: float) -> bool:
        assert 0 <= prob <= 1, f"Probability must be between 0 and 1, but got {prob}."
        if self.entailment_threshold is None:
            return True
        return prob >= self.entailment_threshold 

    def check_implication(self, *args, **kwargs):
        pass
    
    def are_equivalent(self, text1, text2, **kwargs):
        implication_1 = self.check_implication(text1, text2, **kwargs)
        implication_2 = self.check_implication(text2, text1, **kwargs)  # pylint: disable=arguments-out-of-order
        assert (implication_1 in self.entailment_classes) and (implication_2 in self.entailment_classes)

        if self.strict_entailment:
            semantically_equivalent = (implication_1 == "entailment") and (implication_2 == "entailment")

        else:
            implications = [implication_1, implication_2]
            # Check if none of the implications are 0 (contradiction) and not both of them are neutral.
            semantically_equivalent = ("contradiction" not in implications) and (["neutral", "neutral"] != implications)
    
        return semantically_equivalent
    
    @staticmethod
    def from_kwargs(**kwargs):
        classname = kwargs.pop("classname", None)
        module, _, name = classname.rpartition(".")
        obj = create_object_from_class_string(module, name, kwargs)
        kwargs["classname"] = classname
        return obj

class EntailmentHF(BaseEntailment):
    def __init__(self, model_name: str="microsoft/deberta-v2-xlarge-mnli", device: str | None=None, **kwargs):
        super().__init__(**kwargs)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.default_cache_file = f'entailment_hf/{model_name.replace("/", "__")}__cache.pkl'        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def check_implication(self, text1, text2, *args, **kwargs):
        hashed = generate_uuid(f"{text1}, {text2}")
        if hashed in self.prediction_cache:
            logging.info('Restoring hashed instead of predicting with model.')
            pred, probs = self.prediction_cache[hashed]
        else:
            if self.cache_only:
                raise ValueError
            
            inputs = self.tokenizer(text1, text2, return_tensors="pt").to(self.device)
            if inputs["input_ids"].shape[1] > 512:
                logging.warning(f"Input length is {inputs['input_ids'].shape[1]}. Truncating to 512.\n"
                                f"-> Text1: {text1}\nText2: {text2}")
                inputs = {k: v[:, :512] for k, v in inputs.items()}
            # The model checks if text1 -> text2, i.e. if text2 follows from text1.
            # check_implication('The weather is good', 'The weather is good and I like you') --> 1
            # check_implication('The weather is good and I like you', 'The weather is good') --> 2
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            # Deberta-mnli returns `neutral` and `entailment` classes at indices 1 and 2.
            pred = torch.argmax(probs).cpu().item()
            probs = probs.tolist()[0]            
            self.prediction_cache[hashed] = (pred, probs) 

        index2label = {i: l for l, i in self.entailment_mapping.items()}
        label = index2label[pred] 
        if label == "entailment" and not self.satisfy_threshold(probs[pred]):
            label = "neutral"
        # By storing it in the cache, we avoid re-running the model.
        # even if we just want to change the threshold
        return label


class EntailmentLLM(LLMJudgeMixin, BaseEntailment):
    def __init__(self, prompt: str, **kwargs):
        super(EntailmentLLM, self).__init__(**kwargs)
        self.default_cache_file = f'entailment_llm/{self.model_name.replace("/", "__")}_cache.pkl'
        self._prompt = prompt
        
    def get_entailment_label(self, response: str, token_logprobs: List[Tuple[str, float]]) -> str:
        for label_type in self.entailment_mapping:
            if label_type in response:
                label = label_type
                break
        else:
            label = "neutral"
                
        if label == "entailment" and self.entailment_threshold is not None:
            prob, _ = self.get_label_prob(label, response, token_logprobs)
            if prob < self.entailment_threshold:
                label = "neutral"
        
        return label
            
    def check_implication(self, text1, text2, **kwargs):
        prompt = self.get_prompt(text1=text1, text2=text2, **kwargs)
        
        logging.debug('%s input: %s', self.__class__.__name__, prompt)
        hashed = generate_uuid(prompt)
        if hashed in self.prediction_cache:
            logging.info('Restoring hashed instead of predicting with model.')
            response, r_logprob, tw_logprobs = self.prediction_cache[hashed]
        else:
            if self.cache_only:
                raise ValueError
            response, r_logprob, tw_logprobs = self.predict(prompt)        
            self.prediction_cache[hashed] = (response, r_logprob, tw_logprobs)
            #^Note: 
            # response is the text, 
            # r_logprob is the cumulative logprob,
            # tw_logprobs is the token-wise logprobs
            logging.debug(f'{self.__class__.__name__} prediction: {response}')
            logging.debug(f'{self.__class__.__name__} response_logprob: {r_logprob}')
            logging.info(f'{self.__class__.__name__} tokenwise_logprobs: {tw_logprobs}')
        
        # We prefer to determine the value after we extract it from cache
        # since it enable us avoid re-running the evaluation model, if we
        # have already done so.            
        label = self.get_entailment_label(response.lower()[:30], tw_logprobs)
        return label
        

class EntailmentGPT4oMini(EntailmentLLM):
    def __init__(self, **kwargs):
        super().__init__(model_name='gpt-4o-mini-2024-07-18', **kwargs)


class ParaphraseLLM(LLMJudgeMixin, CacheMixin):
    def __init__(self, prefix: str="llm_judge", prompt: str | None=None, threshold: float|None=None, confident_class: str = "no", **kwargs):
        super().__init__(**kwargs)
        self.default_cache_file = f'{prefix}/{self.model_name.replace("/", "__")}_cache.pkl'        
        self.confident_threshold = threshold
        self._prompt = prompt
        self.default_class = "yes" if confident_class == "no" else "no"
        self.confident_class = confident_class
        
    def are_equivalent(self, text1: str, text2: str, **kwargs) -> Tuple[str, str]:
        text1 = text1[2:] if text1.startswith("- ") else text1
        text2 = text2[2:] if text2.startswith("- ") else text2
        
        prompt = self.get_prompt(text1=text1, text2=text2, **kwargs)
        logging.debug(f'{self.__class__.__name__} input: {prompt}')
        hashed = generate_uuid(prompt)
        if hashed in self.prediction_cache:
            logging.info('Restoring hashed instead of predicting with model.')
            response, r_logprob, tw_logprobs = self.prediction_cache[hashed]
        else:
            if self.cache_only:
                raise ValueError
            response, r_logprob, tw_logprobs = self.predict(prompt)        
            self.prediction_cache[hashed] = (response, r_logprob, tw_logprobs)
            #^Note: 
            # response is the text, 
            # r_logprob is the cumulative logprob,
            # tw_logprobs is the token-wise logprobs
            logging.debug(f'{self.__class__.__name__} prediction: {response}')
            logging.debug(f'{self.__class__.__name__} response_logprob: {r_logprob}')
            logging.debug(f'{self.__class__.__name__} tokenwise_logprobs: {tw_logprobs}')

        logging.info(f'{self.__class__.__name__} prediction: {response}')
        # We will be conservative. If we're not sure whether an insight is related to
        # the subtopic or not, we will keep it as related.
        if response.lower().startswith(self.confident_class):
            final_response = self.confident_class
            if self.confident_threshold is not None:
                prob, _ = self.get_label_prob(self.confident_class, response.lower()[:5], tw_logprobs)
                if prob < self.confident_threshold:
                    final_response = self.default_class
                    
        elif response.lower().startswith(self.default_class):
            final_response = self.default_class
        else:
            final_response = self.default_class
            
        return final_response == "yes"


def get_semantic_ids(strings_list, model: BaseEntailment, num_iter: int | None=None, **kwargs):
    """Group list of predictions into semantic meaning."""

    # Initialise all ids with -1.
    semantic_set_ids = [-1] * len(strings_list)
    # Keep track of current id.
    next_id = 0
    for i, string1 in enumerate(strings_list):
        if num_iter is not None and i >= num_iter:
            break
        # Check if string1 already has an id assigned.
        if semantic_set_ids[i] == -1:
            # If string1 has not been assigned an id, assign it next_id.
            semantic_set_ids[i] = next_id
            for j in range(i+1, len(strings_list)):
                # Search through all remaining strings. If they are equivalent to string1, assign them the same id.
                if string1 == strings_list[j] or model.are_equivalent(string1, strings_list[j], **kwargs):
                    semantic_set_ids[j] = next_id
            next_id += 1

    assert num_iter is not None or -1 not in semantic_set_ids
    return semantic_set_ids

class RelatedSubtopicLLM(LLMJudgeMixin, CacheMixin):
    def __init__(self, prompt: str | None=None, confident_threshold: float|None=None, confident_class: str = "no", prefix="unrelated_subtopic_llm", **kwargs):
        super().__init__(**kwargs)
        self.default_cache_file = f'{prefix}/{self.model_name.replace("/", "__")}_cache.pkl'        
        self.confident_threshold = confident_threshold
        self._prompt = prompt
        
        self.default_class = "yes" if confident_class == "no" else "no"
        self.confident_class = confident_class
        
    def check_subtopic(self, subtopic: str, insight: str, **kwargs) -> Tuple[str, str]:
        insight = insight[2:] if insight.startswith("- ") else insight
        prompt = self.get_prompt(insight=insight, subtopic=subtopic, **kwargs)

        logging.debug(f'{self.__class__.__name__} input: {prompt}')
        hashed = generate_uuid(prompt)
        if hashed in self.prediction_cache:
            logging.info('Restoring hashed instead of predicting with model.')
            response, r_logprob, tw_logprobs = self.prediction_cache[hashed]
        else:
            if self.cache_only:
                raise ValueError
            response, r_logprob, tw_logprobs = self.predict(prompt)        
            self.prediction_cache[hashed] = (response, r_logprob, tw_logprobs)
            #^Note: 
            # response is the text, 
            # r_logprob is the cumulative logprob,
            # tw_logprobs is the token-wise logprobs
            logging.debug(f'{self.__class__.__name__} prediction: {response}')
            logging.debug(f'{self.__class__.__name__} response_logprob: {r_logprob}')
            logging.debug(f'{self.__class__.__name__} tokenwise_logprobs: {tw_logprobs}')

        logging.info(f'{self.__class__.__name__} prediction: {response}')
        # We will be conservative. If we're not sure whether an insight is related to
        # the subtopic or not, we will keep it as related.
        if response.lower().startswith(self.confident_class):
            final_response = self.confident_class
            if self.confident_threshold is not None:
                prob, _ = self.get_label_prob(self.confident_class, response.lower()[:5], tw_logprobs)
                if prob < self.confident_threshold:
                    final_response = self.default_class
        else:
            final_response = self.default_class
            
        return final_response, response
