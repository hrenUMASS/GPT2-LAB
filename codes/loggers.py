import logging
import os

import torch

prepare_logger, train_logger, validation_logger, final_logger, cuda_logger, loss_logger = None, None, None, None, None, None


def get_logger(name, log_file, level=logging.INFO, clear=True):
    if clear:
        open(log_file, 'w').close()
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    for hand in logger.handlers:
        logger.removeHandler(hand)
    logger.addHandler(handler)
    return logger


def log_info(logger, msg):
    print(msg)
    if logger is not None:
        logger.info(msg)


def cuda_mem_in_mb():
    return torch.cuda.memory_allocated() / 2 ** 20


def initial_loggers(log_path):
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    global prepare_logger, train_logger, validation_logger, final_logger, cuda_logger, loss_logger
    prepare_logger = get_logger('gpt2 prepare logger', log_path + 'pre.log')
    train_logger = get_logger('gpt2 train logger', log_path + 'train.log')
    validation_logger = get_logger('gpt2 validation logger', log_path + 'val.log')
    final_logger = get_logger('gpt2 final logger', log_path + 'fin.log')
    cuda_logger = get_logger('cuda logger', log_path + 'cuda.log')
    loss_logger = get_logger('loss logger', log_path + 'los.log')
    if None not in (prepare_logger, train_logger, validation_logger, final_logger, cuda_logger, loss_logger):
        log_info(prepare_logger, 'loggers successfully initialized')
