import hydra, logging, omegaconf, os, random, time

import tqdm
import utils_io
import utils_logging as log_utils
import utils_models as models
import processors


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(version_base=None, config_path="../configs/eval_configs", config_name="eval_pprocess_base")
def main(cfg: omegaconf.DictConfig):
    random.seed(cfg["setup"]["seed"])
    logging_path = cfg.setup.output_filepath.replace(".json", ".log")
    log_utils.setup_logger(logging_path, log, __name__)
        
    log.info("=" * 80)
    log.info(omegaconf.OmegaConf.to_yaml(cfg, resolve=True))
    log.info("=" * 80)
    filepaths = [cfg.setup.evaluations_filepath]
    output_paths = [cfg.setup.output_filepath]

    if os.path.isdir(cfg.setup.evaluations_filepath):
        epath = cfg.setup.evaluations_filepath
        basenames = os.listdir(cfg.setup.evaluations_filepath)
        basenames = [b for b in basenames if os.path.exists(f"{epath}/{b}/data.json")]
        filepaths = [f"{epath}/{b}/data.json" for b in basenames]
        output_paths = [f"{cfg.setup.output_filepath}/{b}.json" for b in basenames]

    for filepath, output_path in tqdm.tqdm(zip(filepaths, output_paths)):
        log.info("\n" * 4 + "=" * 80)
        log.info(f"filepath: {filepath}\noutput_path: {output_path}")
        log.info("=" * 80 + "\n" * 4)

        if cfg.setup.get("batch", "False"):
            processor_class = processors.BatchProcessor
            processor_kwargs = cfg.requests
            if not filepath.endswith("data.json"):
                filepath = filepath.split(".json")[0] + "/data.json"
        else:
            processor_class = processors.Processor
            processor_kwargs = {}
            
        data = utils_io.read_json(filepath)
        eval_model = data["evaluation_kwargs"]["model_kwargs"]["model"]
        # Load model and evaluate
        model_kwargs = models.load_model(cfg.setup.config_filepath, eval_model)
        processor: processors.Processor = processor_class(evals=data, logger=log, **model_kwargs, **processor_kwargs)
        processor.setup(filepath, output_path)
        processor.post_process()
        processor.merge_results()
        processor.populate_metadata()
        processor.save(output_path)
        log.info(f"Finished processing {filepath} and saved to {output_path}")
    log.info(f"Waiting 5s before starting the next post-processing script.")
    time.sleep(5)

if __name__ == "__main__":
    main()
