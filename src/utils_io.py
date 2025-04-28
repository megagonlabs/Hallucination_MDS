import json, jsonlines, os
import pickle
import shutil
import yaml
from typing import Any, Dict, List, Union


def read_json(filepath: str) -> Dict[str, Any]:
    """Wrapper function to load a JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def read_yaml(filepath: str) -> Dict[str, Any]:
    """Wrapper function to load a YAML file."""
    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def read_jsonlines(filepath: str, mode: str = "r", **kwargs) -> jsonlines.Reader:
    """Wrapper function to load a JSONLines file."""
    with jsonlines.open(filepath, mode=mode, **kwargs) as f:
        return [line for line in f]


def read_prompt(filepath: str):
    with open(filepath) as f:
        if filepath.endswith(".txt"):
            return f.read()
        else:
            return json.load(f)


def to_json(filepath: str, obj: Dict[str, Any]):
    """Wrapper function to dump a JSON file."""
    dir = os.path.dirname(filepath)
    os.makedirs(dir, exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def to_jsonlines(filepath: str, obj_lst: List[Dict[str, Any]], mode: str="w", **kwargs):
    dir = os.path.dirname(filepath)
    os.makedirs(dir, exist_ok=True)
    
    with jsonlines.open(filepath, mode=mode, **kwargs) as writer:
        writer.write_all([obj_lst] if isinstance(obj_lst, dict) else obj_lst)


def to_txt(filepath: str, content: str, mode="a"):
    """Wrapper function to dump a text file."""
    dir = os.path.dirname(filepath)
    os.makedirs(dir, exist_ok=True)

    with open(filepath, mode) as f:
        f.write(content + "\n")


def create_directory(path: str, force_reset=False):
    if os.path.exists(path) and len(os.listdir(path)) == 0:
        return 

    elif not force_reset and os.path.exists(path):
        raise ValueError(f"Output path {path} already exists"
                         " but you're trying to run a new evaluation."
                         "Remove the directory or change the stage.")
    elif force_reset:
        shutil.rmtree(path)
    
    os.makedirs(path, exist_ok=True)
    
    
def estimate_json_mb_size(data: Union[List[dict], dict]) -> float:
    # Serialize the list of dictionaries to a JSON string
    if isinstance(data, list):
        json_str = "\n".join([json.dumps(d) for d in data])
        
    json_str = json.dumps(data)
    
    # Get the byte size of the JSON string
    byte_size = len(json_str.encode('utf-8'))
    
    # Convert byte size to megabytes
    mb_size = byte_size / 1_048_576  # 1024 * 1024
    
    return mb_size


def get_jsonlines_chunks_of(lst_dicts: List[dict], size_in_mb: int=100, max_lines=50_000, return_doc_to_group_map=False):
    """Split a list of dictionaries into chunks of size `size_in_mb` in megabytes."""
    sizes = 0
    chunks = [0]
    for i, d in enumerate(lst_dicts):
        d_size = estimate_json_mb_size(d)
        
        # Start new chunk        
        if sizes + d_size > size_in_mb or i > max_lines:
            sizes = d_size 
            chunks.append(i)
        else:
            sizes += d_size
    
    if len(chunks) == 1:
        groups = {0: lst_dicts}
        doc_to_group_map = {i: 0 for i in range(len(lst_dicts))}
    else:    
        groups = {}
        doc_to_group_map = {}
        for chunk_start, chunk_end in zip(chunks[0:-1], chunks[1:]):
            group_id = len(groups)
            groups[group_id] = lst_dicts[chunk_start:chunk_end]
            
            for i in range(chunk_start, chunk_end):
                doc_to_group_map[i] = group_id
            
    if return_doc_to_group_map:
        return groups, doc_to_group_map
    
    return groups


def save_pickle(object, filepath: str):
    dirname = os.path.dirname(filepath)
    os.makedirs(dirname, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(object, f)