import time
from datetime import datetime

import numpy as np
import torch
import transformers as tfm
from torch import nn
from torch.utils.data import DataLoader

from . import loggers
from .loggers import log_info
from .util import train_one_epoch, get_module_from_parallel, save_checkpoint, load_checkpoint


def train(model, dataset, batch_size, epochs, epoch_iter, learning_rate=1e-2, weight_decay=1e-4, n_gpus=1,
          save_path=None, from_checkpoint=False, continue_train=False, tokenizer=None, data_func=lambda x: x):
    loss_logger, train_logger = loggers.loss_logger, loggers.train_logger
    no_decay = ['bias', 'LayerNorm.weight']
    device_ids = list(range(n_gpus))
    # SGD
    # fix learning rate
    #

    optimizer_params = [{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                         'weight_decay': weight_decay},
                        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                         'weight_decay': 0.0}]
    optimizer = tfm.AdamW(optimizer_params, lr=learning_rate)
    scheduler = tfm.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100,
                                                    num_training_steps=epochs * epoch_iter)
    losses = []
    # gpu = GPUtil.getGPUs()[0]
    # n = max(len(dataset) // 9000, 1)
    n = 5
    datasets = dataset.split(n)
    data_loaders = [DataLoader(i, shuffle=False, batch_size=batch_size, collate_fn=lambda x: x) for i in datasets]
    print([k.get_loaded_length() for k in datasets])
    mini_epoch = 0
    if from_checkpoint:
        epoch, mini_epoch, model_state, optimizer_state, scheduler_state, loss = load_checkpoint(
            save_path + 'checkpoint.pt')
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(scheduler_state)
        if continue_train:
            epochs = epochs - epoch + 1
    model = nn.DataParallel(model, device_ids)
    model.train()
    for e in range(epochs):
        epoch_start = time.time()
        for ed, data_loader in enumerate(data_loaders[mini_epoch:]):
            mini_epoch = 0
            loss_value, loss = train_one_epoch(data_loader, model, optimizer, scheduler, data_process_func=data_func)
            losses.extend(loss_value)
            if save_path is not None:
                get_module_from_parallel(model).save_pretrained(save_path)
                if tokenizer is not None:
                    tokenizer.save_pretrained(save_path)
                save_checkpoint(save_path + 'checkpoint.pt', model, e, ed + mini_epoch, optimizer, scheduler, loss)
                log_info(loss_logger, 'saved models for mini epoch {} in epoch {}'.format(ed + mini_epoch, e))
        loss_seg = losses[e * epoch_iter:]
        log_info(train_logger, '-' * 50)
        log_info(train_logger, 'Epoch {}, Mean Loss {}, Min Loss {}'.format(e, np.mean(loss_seg), np.min(loss_seg)))
        time_diff = time.time() - epoch_start
        log_info(train_logger,
                 'Time {}, Epoch Time {}, Avg Iter Time {}'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                                                                   time_diff, time_diff / epoch_iter))
    return model, torch.tensor(losses)
