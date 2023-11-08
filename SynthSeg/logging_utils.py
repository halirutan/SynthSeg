import logging
import sys
import inspect


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a unique logger for the SynthSeg module.
    If no name is provided, the default is to use the name of the module where this function was
    called from.
    """
    if name is None:
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        name = module.__name__

    logger_name = "SynthSeg" if name == "__main__" else name
    our_logger = logging.getLogger(logger_name)
    our_logger.setLevel(logging.DEBUG)
    log_stdout_handler = logging.StreamHandler(sys.stdout)
    log_formatter = logging.Formatter("%(name)s - %(levelname)s: %(message)s")
    log_stdout_handler.setFormatter(log_formatter)
    our_logger.addHandler(log_stdout_handler)
    return our_logger
