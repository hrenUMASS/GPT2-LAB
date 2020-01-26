import logging
import os
import sys

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


def initial_loggers(log_path, module_name=None):
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    global prepare_logger, train_logger, validation_logger, final_logger, cuda_logger, loss_logger
    if not (None not in (prepare_logger, train_logger, validation_logger, final_logger, cuda_logger, loss_logger)):
        pre = get_logger('prepare logger', log_path + 'pre.log')
        tra = get_logger('gpt2 train logger', log_path + 'train.log')
        val = get_logger('gpt2 validation logger', log_path + 'val.log')
        fin = get_logger('final logger', log_path + 'fin.log')
        cud = get_logger('cuda logger', log_path + 'cuda.log')
        los = get_logger('loss logger', log_path + 'los.log')
        result = {'prepare_logger': pre, 'train_logger': tra, 'validation_logger': val, 'final_logger': fin,
                  'cuda_logger': cud, 'loss_logger': los}
        main = sys.modules['__main__']
        for name, log in result.items():
            if hasattr(main, name):
                setattr(main, name, log)

        prepare_logger = pre
        train_logger = tra
        validation_logger = val
        final_logger = fin
        cuda_logger = cud
        loss_logger = los
    if module_name is not None:
        result = {'prepare_logger': prepare_logger, 'train_logger': train_logger,
                  'validation_logger': validation_logger, 'final_logger': final_logger,
                  'cuda_logger': cuda_logger, 'loss_logger': loss_logger}
        module = sys.modules[module_name]
        for name, log in result.items():
            if hasattr(module, name):
                setattr(module, name, log)
