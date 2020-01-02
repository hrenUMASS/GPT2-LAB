import logging

import torch


def get_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def log_info(logger, msg):
    print(msg)
    if logger is not None:
        logger.info(msg)


def cuda_mem_in_mb():
    return torch.cuda.memory_allocated() / 2 ** 20
