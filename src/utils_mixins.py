
import logging, os, pickle
import numpy as np
from typing import List, Tuple

from utils_io import save_pickle
from utils_models import load_model, generate_chat_with_logprobs


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class CacheMixin:
    def __init__(self, cache_only: bool, cache_dir: str="./caches_dir", **kwargs):
        self.cache_only = cache_only
        self.default_cache_file = 'default_cache.pkl'
        self.cache_dir = cache_dir
        self.prediction_cache = {}

    @property
    def cache_filepath(self):
        return f"{self.cache_dir}/{self.default_cache_file}"

    def init_prediction_cache(self, filepath: str | None=None):
        if filepath is None:
            filepath = self.cache_filepath
        
        if os.path.exists(filepath):
            logging.info('Restoring prediction cache from %s', filepath)
            with open(filepath, "rb") as infile:
                self.prediction_cache = pickle.load(infile)
        else:
            logging.info(f"Prediction cache file not found: {filepath}")
            self.prediction_cache = {}

    def save_prediction_cache(self):
        # Write the dictionary to a pickle file.
        save_pickle(self.prediction_cache, self.cache_filepath)


class LLMJudgeMixin:
    def __init__(self, model_name: str, prompt: str | None=None, **kwargs):
        super().__init__(**kwargs)
        self.client = load_model(model=model_name, **kwargs)["client"]
        self.model_name = model_name
        self._prompt = prompt
        self.generation_kwargs = kwargs.pop("generation_kwargs", {})
    
    def get_label_prob(self, label: str, response: str, token_logprobs: List[Tuple[str, float]]) -> str:
        if label in response:
            resp, resp_logprob, num_tokens = "", 0, 0 # TODO: We can add a normalization mechanism here.
            for (token, logprob) in token_logprobs:
                if label not in resp:
                    resp += token.lower()
                    resp_logprob += logprob
                    num_tokens += 1
    
            if num_tokens >= 1:
                logging.warning(f"Entailment threshold being computed with " 
                                f"more than 1 token: {token_logprobs}\n"
                                f"(p({resp}|prompt)={np.exp(resp_logprob)})")
                
            return np.exp(resp_logprob), num_tokens
        else:
            return 0, 0

    def predict(self, prompt):
        return generate_chat_with_logprobs(client=self.client, model=self.model_name, prompt=prompt, **self.generation_kwargs)
    
    def get_prompt(self, orig_prompt: str | None=None, **kwargs):
        if orig_prompt is None:
            orig_prompt = self._prompt

        prompt = orig_prompt
        for param_name, param_value in kwargs.items():
            assert isinstance(param_value, (int, float, str)), F"Invalid type for param '{param_name}': {type(param_value)}: {param_value}"
            prompt = prompt.replace("{{" + param_name + "}}", str(param_value))
            
        if "{{" in prompt or "}}" in prompt:
            raise ValueError(F"Prompt still contains placeholders: {prompt}")

        if prompt == orig_prompt:
            raise ValueError(F"Prompt was not modified: {prompt}")

        return prompt
