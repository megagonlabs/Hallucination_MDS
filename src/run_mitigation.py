import hydra, logging, omegaconf, os

import tqdm
import utils
import utils_io
import utils_logging as log_utils
import mitigation_base as mitigation

from typing import Dict, List


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class Pipeline:
    def __init__(self, filters, data, temp_path: str="./temp"):
        self.steps: List[mitigation.FilterBase] = filters
        self.data = data
        self.temp_dir = temp_path
        
    def run(self):
        for step_name, step in tqdm.tqdm(self.steps.items(), "Mitigation Steps"):
            logging.info(f"Running step: {step_name}")
            step.init_cache()
            self.data = step.run(self.data)
            step.save(f"{self.temp_dir}/{step_name}.json", self.data)
            del step
        return self.data


def create_filters(cfg, **kwargs) -> Dict[str, mitigation.FilterBase]:
    filters: Dict[str, mitigation.FilterBase] = {} 
    for filter_name, filter_cfgs in cfg.items():
        filters[filter_name] = utils.load_object_from_dict(filter_cfgs, **kwargs)
    return filters


@hydra.main(version_base=None, config_path="../configs/mitigation", config_name="base")
def main(cfg: omegaconf.DictConfig):
    print("=" * 80)
    print(omegaconf.OmegaConf.to_yaml(cfg, resolve=True))
    print("=" * 80)

    # Experiment setup
    logging_path = cfg.filepaths.output + ".log"
    log_utils.setup_logger(logging_path, log, __name__)

    if os.path.isdir(cfg.filepaths.input):
        basenames = sorted(os.listdir(cfg.filepaths.input))
        basenames = sorted([fn for fn in basenames if fn.endswith(".json")])
        filepaths = [f"{cfg.filepaths.input}/{fn}" for fn in basenames]
        output_paths = [f"{cfg.filepaths.output}/{fn}" for fn in basenames]
        temp_paths = [f"{cfg.filepaths.temp}/{fn}" for fn in basenames]
    else:
        filepaths = [cfg.filepaths.input]
        output_paths = [cfg.filepaths.output]
        temp_paths = [cfg.filepaths.temp]
    
    # Create filters for each filepath
    for filepath, output_path, temp_path in zip(filepaths, output_paths, temp_paths):
        log.info("=" * 80)
        log.info(f"input filepath: {filepath}")
        log.info(f"output path: {output_path}")
        log.info(f"temp path: {temp_path}")
        log.info("=" * 80)

        temp_path = temp_path.rpartition(".")[0]
        # Load data
        data = utils_io.read_json(filepath)
        filters = create_filters(cfg.filters, cache_dir=temp_path + "/cache", config_filepath=cfg.filepaths.config)
        
        pipeline = Pipeline(filters, data, temp_path=temp_path + "/out")
        filtered_data = pipeline.run()
        utils_io.to_json(output_path, filtered_data)
        log.info(f"Finished processing {filepath} and saved to {output_path}")
        

if __name__ == "__main__":
    main()