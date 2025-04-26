import os, time
import hydra, logging, omegaconf, random
import utils
import utils_io
import utils_logging as log_utils
import utils_models as models
import evaluators


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(
    version_base=None, config_path="../configs/eval_configs", config_name="eval_base"
)
def main(cfg: omegaconf.DictConfig):
    print("=" * 80)
    print(omegaconf.OmegaConf.to_yaml(cfg, resolve=True))
    print("=" * 80)

    # Experiment setup
    random.seed(cfg["setup"]["seed"])
    logging_path = cfg.setup.output_filepath + ".log"
    log_utils.setup_logger(logging_path, log, __name__)

    # Load data and prompt
    eval_prompt = utils_io.read_prompt(cfg.requests.prompt_filepath)
    eval_prompt_uuid = utils.generate_uuid({"eval_prompt": eval_prompt})
    log.info(f"cfg.setup.data_filepath: {cfg.setup.data_filepath}")

    if os.path.isdir(cfg.setup.data_filepath):
        basenames = os.listdir(cfg.setup.data_filepath)
        basenames = sorted([fn for fn in basenames if fn.endswith(".json")])
        filepaths = [f"{cfg.setup.data_filepath}/{fn}" for fn in basenames]
        output_paths = [f"{cfg.setup.output_filepath}/{fn}" for fn in basenames]
    else:
        filepaths = [cfg.setup.data_filepath]
        output_paths = [cfg.setup.output_filepath]

    for filepath, output_path in zip(filepaths, output_paths):
        log.info("\n" * 4)
        log.info("=" * 80)
        log.info(f"\nfilepath: {filepath}\noutput_path: {output_path}")
        log.info("=" * 80)
        log.info("\n" * 4)

        data = utils_io.read_json(filepath)
        data_uuid = utils.generate_uuid(data)

        # Load model and evaluate
        model_kwargs = models.load_model(
            cfg.setup.config_filepath, cfg.requests.eval_kwargs.model
        )
        evaluator: evaluators.Evaluator = utils.load_object_from_dict(
            cfg.requests,
            eval_prompt=eval_prompt,
            _eval_prompt_uuid=eval_prompt_uuid,
            _data=data,
            _data_uuid=data_uuid,
            **model_kwargs,
            logger=log,
        )
        evaluator.setup(output_path)
        evaluator.prepare_inputs_for_evaluation()
        evaluator.evaluate()
        evaluator.populate_metadata()
        evaluator.save(output_path)
        log.info(f"Finished processing {filepath} and saved to {output_path}")
        log.info(f"Waiting 5s before starting the next file...")
        time.sleep(5)


if __name__ == "__main__":
    main()
 