from dataclasses import dataclass

import os, logging, json


def setup_logger(path: str, logger, name: str): 
    path_dir = os.path.dirname(path)
    os.makedirs(path_dir, exist_ok=True)
    
    if not path.endswith(".log"):
        path = path.rpartition(".")[0] + ".json"

    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s]: %(message)s', datefmt='%Y/%m/%d %I:%M:%S')    
    # Create a file handler and set the formatter
    file_handler = logging.FileHandler(path, "w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Create a stream handler (output to terminal) and set the formatter
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
        
    logger = logging.getLogger(name)
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
def with_sep(fn:callable, msg: str, sep="-"):
    sep = sep * min(len(msg), 80)
    if isinstance(msg, dict):
        msg = json.dumps(msg, indent=2, sort_keys=True)
    fn("\n" + sep + "\n" + msg, "\n" + sep + "\n")


@dataclass
class LoggingMixin:
    logger: logging.Logger
    
    def info(self, msg, **kwargs):
        self.logger.info(msg, **kwargs)
        
    def debugging(self, msg, **kwargs):
        self.logger.debug(msg, **kwargs)
    
    def warn(self, msg, **kwargs):
        self.logger.warn(msg, **kwargs)
        
    def error(self, msg, **kwargs):
        self.logger.error(msg, **kwargs)
    
    def critical(self, msg, **kwargs):
        self.logger.critical(msg, **kwargs)
        
    def with_sep(self, msg, sep="-"):
        sep = sep * min(len(msg), 80)
        if isinstance(msg, dict):
            msg = json.dumps(msg, indent=2, sort_keys=True)
        self.logger.info(f"""\n{sep}\n{msg}\n{sep}\n""")

