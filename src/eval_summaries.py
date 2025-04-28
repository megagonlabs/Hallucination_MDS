import functools, hydra, logging, omegaconf, random
import utils_logging as log_utils
import utils_models as models
import evaluators


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

@hydra.main(version_base=None, config_path="../configs/evaluation", config_name="config") 
def main(cfg : omegaconf.DictConfig):
    # use syntax: `python eval_summaries.py +evaluator=multi_requests`
    # to select configuration
    print(omegaconf.OmegaConf.to_yaml(cfg, resolve=True))

    # Experiment setup
    random.seed(cfg.setup.seed)
    log_utils.setup_logger(cfg.setup.output_filepath.replace(".json", ".log"), log, __name__)
    logger_kwargs = {"logger": log}
    # Load model and evaluate
    eval_model = cfg.requests.eval_kwargs.get("model", "gpt-4o-mini-2024-07-18")
    model_kwargs = models.load_model(cfg.setup.config_filepath, eval_model)

    # Model
    evaluator: evaluators.Evaluator = hydra.utils.instantiate(
        cfg.requests,
        **model_kwargs,
        **logger_kwargs,
        _to_dict_fn=functools.partial(omegaconf.OmegaConf.to_container, resolve=True),
    )
    evaluator.setup(cfg.setup.summary_filepath, cfg.setup.output_filepath)
    evaluator.prepare_inputs_for_evaluation()
    evaluator.evaluate()
    evaluator.post_process()
    evaluator.merge_results() 
    evaluator.populate_metadata()
    evaluator.save()


if __name__ == "__main__":
    main()
    