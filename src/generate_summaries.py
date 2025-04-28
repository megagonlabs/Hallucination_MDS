import json, logging, os, random, tqdm
import argparse

from typing import Any, Dict, List, Optional
import utils_models as models
import utils, utils_logging, utils_io

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def populate_prompt_(
    data: dict,
    prompt: str,
    prompt_uuid: str,
    examples_range: callable,
    frequency: Optional[int] = 50,
    is_conversation_domain: bool = False,
) -> list:
    documents = data["documents"]
    subtopics = data["subtopics"]
    
    examples = []
    for i in tqdm.tqdm(range(*examples_range)):
        if i >= len(data["assignments"]):
            break
        
        example = data["assignments"][i]
        example["__response_idx__"] = i
        n_docs = example["num_documents"]
        subtopic_uuid = example["subtopic_uuid"]
        pop_prompt = prompt\
            .replace("{{n_articles}}", str(int(n_docs)))\
            .replace("{{topic}}", data["topic"])\
            .replace("{{subtopic}}", subtopics[subtopic_uuid]["subtopic"])

        if is_conversation_domain:
            pop_prompt = pop_prompt.replace(
                "{{participants}}",  ", ".join(data["topic_metadata"]["participants"])
            )

        # ------------------------------------------
        #       Infill articles text in the prompt
        # ------------------------------------------
        # 1. Randomize order to avoid any artifacts
        ids = list(range(n_docs)); random.shuffle(ids)
        example["docs_order"] = ids
        
        # 2. Populate articles
        docs_uuid = [example[f"doc_uuid_{i}"] for i in ids]
        docs_content = [documents[uuid]["document_text"] for uuid in docs_uuid] 

        prefix = "Conversation" if is_conversation_domain else "Article"
        docs_content = [f"{prefix} {i+1}:\n{doc_content}\n\n" for i, doc_content in enumerate(docs_content)]
        
        pop_prompt = pop_prompt.replace("{{articles}}", "".join(docs_content).strip())

        for expr in ("{{", "}}"):
            assert expr not in pop_prompt, f"Prompt not fully populated: {pop_prompt}"

        example["prompt"] = pop_prompt
        example["prompt_uuid"] = prompt_uuid
        examples.append(example)
        
        if frequency is not None and i % frequency == 0:
            logger.debug(f"Example {i}:\n{json.dumps(example, sort_keys=True, indent=1)}")
        
    data["assignments"] = examples


def generate_(examples: List[Dict[str, Any]], **model_kwargs):
    for example in tqdm.tqdm(examples):
        example["response"] = models.generate_chat(prompt=example["prompt"], **model_kwargs)
        example["response_uuid"] = utils.generate_uuid({"response": example["response"]})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_filepath", type=str, required=True)
    parser.add_argument("--output_filepath", type=str, required=True)
    parser.add_argument("--config_filepath", type=str, required=True)
    parser.add_argument("--prompt_filepath", type=str, required=True)
    parser.add_argument("--range", type=str, default="*")
    parser.add_argument("--model", type=str, default="gpt-4o-mini-2024-07-18") # cheapest model
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--max_tokens", type=int, default=800)
    parser.add_argument("--seed", type=int, default=91237)
    parser.add_argument("--conversation", action="store_true")
    args = parser.parse_args()
    
    output_fp = args.output_filepath\
        .replace("{model}", args.model)\
        .replace(".json", f"__examples{args.range}.json")
    os.makedirs(os.path.dirname(output_fp), exist_ok=True)

    random.seed(args.seed)
    logging_fp = args.output_filepath\
        .replace(".json", ".log")\
        .replace("results/", f"logs/")
    utils_logging.setup_logger(logging_fp, logger, __name__)
    logger.info(f"\n{"=" * 80}\n{args}\n{"=" * 80}")
    
    logger.info(f"*** Generating summaries for examples @ {args.input_filepath}")
    data = utils_io.read_json(args.input_filepath)

    logger.info(f"*** Generating summaries using prompt @ {args.prompt_filepath}")
    prompt = utils_io.read_prompt(args.prompt_filepath) 
    prompt_uuid = utils.generate_uuid({"prompt": prompt})
    logger.debug(f"Prompt:\n'''{prompt}'''")
        
    ex_range = [int(e.strip()) for e in args.range.split(",")]
    populate_prompt_(data, prompt, prompt_uuid, ex_range, is_conversation_domain=args.conversation)

    # Load model
    openai_kwargs = models.load_model(args.config_filepath, model=args.model)
    openai_kwargs.update({
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    })
    
    generate_(examples=data["assignments"], **openai_kwargs)
    openai_kwargs.pop("client")
    data["generation_kwargs"] = {
        "prompt": prompt,
        "prompt_uuid": prompt_uuid,
        "prompt_filepath": args.prompt_filepath.rpartition("/")[-1],
        "model": openai_kwargs["model"],
        "model_kwargs": openai_kwargs,
        "range": args.range,
        "seed": args.seed,
    }
    
    logger.info(f"-> Summaries dumped @ {args.output_filepath}")
    with open(output_fp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
