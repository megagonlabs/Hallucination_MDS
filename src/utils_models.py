import os, json
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential, 
)
from typing import Any, Dict, List, Union
import openai, together
import google.generativeai as genai


FAILED_BATCH_STATUS = ("failed", "expired", "cancelling", "cancelled")
# from langsmith.wrappers import wrap_openai
# from langsmith import traceable

def load_model(config_path: str, model: str, **kwargs) -> openai.OpenAI:
    configs = {}
    if config_path.endswith(".txt"):
        with open(config_path) as f:
            configs["api_key"]= f.read().strip()
    else:
        with open(config_path) as f:
            configs = json.load(f)
    
    if model.startswith("gpt-") or "fireworks" in config_path:
        with open(config_path) as f:
            kwargs = {k: v for k,v in kwargs.items() if k in ("api_key", "base_url", "project", "timeout")}
            client = openai.OpenAI(**configs, **kwargs)
    elif "gemini" in model:
        genai.configure(**configs)
        client = genai.GenerativeModel(model)
    else:
        with open(config_path) as f:
            client = together.Together(**configs, **kwargs)
    return {
        "client": client,
    }
    
def generate_chat_gemini(client, prompt: str, max_tokens: int=800, model=None, **kwargs):
    assert isinstance(prompt, str), "Prompt must be a string."
    assert len(prompt.strip()) > 0, "Prompt cannot be empty."
    response = client.generate_content(prompt, 
                                       generation_config=genai.types.GenerationConfig(
                                           max_output_tokens=max_tokens, 
                                           **kwargs))
    return response.text
    
# @traceable
@retry(wait=wait_random_exponential(min=1, max=3), stop=stop_after_attempt(30))
def generate_chat(client, prompt: Union[str, dict], **kwargs):
    """OpenAI-compatible chat completion API."""
    if isinstance(client, genai.GenerativeModel):
        return generate_chat_gemini(client, prompt, **kwargs)
    
    if isinstance(prompt, str):
        assert len(prompt.strip()) > 0, "Prompt cannot be empty."
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = prompt

    response = client.chat.completions.create(
        messages=messages,
        **kwargs
    )
    
    if "n" in kwargs:
        return [r.message.content for r in response.choices]
    else:
        return response.choices[0].message.content


@retry(wait=wait_random_exponential(min=1, max=3), stop=stop_after_attempt(30))
def generate_chat_with_logprobs(client, prompt: Union[str, dict], **kwargs):
    """OpenAI-compatible chat completion API."""
    if isinstance(prompt, str):
        assert len(prompt.strip()) > 0, "Prompt cannot be empty."
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = prompt

    response = client.chat.completions.create(
        messages=messages,
        logprobs=True,
        **kwargs
    )
    text = response.choices[0].message.content
    logprobs = []
    cumulative_logprob = 0
    for logprob in response.choices[0].logprobs.content:
        logprobs.append((logprob.token, logprob.logprob))
        cumulative_logprob += logprob.logprob
        
    return text, cumulative_logprob, logprobs
    

def create_jsonl_batch_format(prompt: Union[str, dict], url="/v1/chat/completions", **kwargs) -> Dict[str, Any]:
    """OpenAI-compatible chat completion Batch API."""        
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = prompt

    kwargs = {k: v for k, v in kwargs.items()}
    kwargs["messages"] = messages
    return { "method": "POST", "url": url, "body": kwargs}


def create_batch_openai(client: openai.OpenAI, filepath: str, **kwargs) -> Dict[str, Any]:
    """Create a file in the OpenAI API."""
    input_file = client.files.create(file=open(filepath, "rb"), purpose="batch")
    batch = client.batches.create(input_file_id=input_file.id, **kwargs)
    return input_file, batch

def parse_batch_response(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Parse the response from the batch API.

    Example of a successful request
    ```
    {
        "custom_id": "request-1",
        "error": null,
        "id": "batch_req_uXlU2g24rQcX1IlSYx5kuYmw",
        "response": {
            "body": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "logprobs": null,
                        "message": {
                            "content": "Hello! How can I assist you today?",
                            "role": "assistant"
                        }
                    },
                    ...
                }
                ],
                "created": 1721433790,
                "id": "chatcmpl-9mrr4IiMKZaNPC8JozMfgaSTeyXD3",
                "model": "gpt-4o-mini-2024-07-18",
                "object": "chat.completion",
                "system_fingerprint": "fp_8b761cb050",
                "usage": {
                    "completion_tokens": 9,
                    "prompt_tokens": 20,
                    "total_tokens": 29
                }
            },
            "request_id": "6baa9d95933a65c1c8529b5311572840",
            "status_code": 200
        }
    }
    ```
    """
    results = []
    choices = batch["response"]["body"]["choices"]    
    for choice in choices:
        results.append(choice["message"]["content"])
    return {
        "id": batch["id"],
        "request_id": batch["response"]["request_id"], 
        "custom_id": batch["custom_id"],
        "text": results,
    }


def is_not_cancelled(batch):
    """Return True if value is None"""
    return batch.status not in FAILED_BATCH_STATUS        


# @retry(retry=retry_if_result(is_not_cancelled), stop=stop_after_delay(3600*24), wait=wait_chain(*[wait_fixed(60) for i in range(3)] + [wait_fixed(300)]))
def sync_retrieve_batch(client: openai.OpenAI, batch_id: str) -> Dict[str, Any]:
    """Retrieve the batch from the OpenAI API."""
    return client.batches.retrieve(batch_id)


def retrieve_batch(client: openai.OpenAI, batch_id: str, blocking=False) -> Dict[str, Any]:
    """Retrieve the batch from the OpenAI API."""
    if not blocking:
        return client.batches.retrieve(batch_id)
    else:
        return sync_retrieve_batch(client, batch_id)
    
    
def dump_batch(client, batch, output_path: str):
    """Dump the batch to a JSON file."""
    out_dir = os.path.dirname(output_path)
    os.makedirs(out_dir, exist_ok=True)
    
    if batch.output_file_id is not None:
        result = client.files.content(batch.output_file_id).content
    
        with open(output_path, 'wb') as file:
            file.write(result)
            
    if batch.error_file_id is not None:
        error = client.files.content(batch.error_file_id).content
        with open(output_path.replace(".jsonl", ".error.jsonl"), 'wb') as file:
            file.write(error)