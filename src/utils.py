from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

import hashlib, importlib, json, random, re
import omegaconf


def generate_uuid(content, indent: int = 2) -> str:
    """Deterministic uuid generator of the `content`."""
    content = json.dumps(content, sort_keys=True, indent=indent).encode("utf-8")
    return hashlib.md5(content).hexdigest()


def sample_list(lst: Iterable, n: int) -> List[Dict[str, Any]]:
    ids = [i for i in range(len(lst))]
    random.shuffle(ids)
    ids = ids[:n]
    return [lst[indx] for indx in ids]


def bullet_processor(text: str) -> List[str]:
    lines = text.split("\n")
    lines = [l.strip() for l in lines if l.strip() != ""]
    return lines


def get_majority_vote(texts: List[str], regex: Optional[str] = None) -> Tuple[str, Counter]:
    """Aggregate multiple generations into a single string.
    
    Regex can be useful in the context of the use of self-consistency
    with different reasoning chains.
    """
    def canonic(text: str) -> List[str]:
        if regex is not None: # returns the first match
            pattern = re.compile(regex)
            if (match := re.search(pattern, text)) is not None:
                return match.group()
            else:
                raise ValueError(f"Regex {regex} not found in text: {text}")
        return text
        
    counter = Counter()
    for text in texts:
        text = canonic(text)
        canonic_text = json.loads(text.strip())
        canonic_text = json.dumps(canonic_text, sort_keys=True)
        counter.update([canonic_text])
        
    # What if there's a tie? Break it randomly
    common = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    common = [(text, counts) for text, counts in common if counts == common[0][1]]
    random.shuffle(common)
    return common[0][0], counter


def get_fully_qualified_function_name(func: callable):
    module = func.__module__
    qualname = func.__qualname__
    return f"{module}.{qualname}"



def create_object_from_class_string(module_name, class_name, parameters):
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    instance = class_(**parameters)
    return instance


def load_object_from_dict(parameters: dict, **kwargs):
    if isinstance(parameters, omegaconf.DictConfig):
        parameters = omegaconf.OmegaConf.to_container(parameters, resolve=True)
        
    parameters.update(kwargs)
    type = parameters.pop('_target_')
    if type is None:
        return None
    else:
        type = type.split('.')
        module_name, class_name = '.'.join(type[:-1]), type[-1]
        return create_object_from_class_string(module_name, class_name, parameters)