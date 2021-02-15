import time
from typing import List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import transformers as tfm
from torch import nn, Tensor
from torch.utils.data import DataLoader

from libs import get_module_from_parallel, save_checkpoint, load_checkpoint, get_model_output
from libs import log_info, cuda_mem_in_mb
from libs import loggers


def train_one_epoch(dataloader, model, optimizer, scheduler, data_process_func, tok):
    losses = []
    cuda_logger, train_logger = loggers.cuda_logger, loggers.train_logger
    loss = None
    res_data = {'id': [], 'gt_label': [], 'pred_label': [], 'pred_logit': [], 'prob': [], 'losses': []}
    for step, raw in enumerate(dataloader):
        step_time = time.time()
        data = data_process_func(raw)
        if data is None:
            log_info(cuda_logger, 'Empty data {} Iter'.format(step))
            continue
        log_info(cuda_logger,
                 'Allocated batches {}, {}'.format(cuda_mem_in_mb(), {k: v.shape for k, v in data.items()}))
        # loss = get_model_output(model, data)[0].mean()
        output = get_model_output(model, data)
        if hasattr(output, 'loss'):
            loss = output.loss.mean()
        else:
            loss = output[0].mean()
        loss_value = loss.item()
        if hasattr(output, 'logits'):
            logit = output.logits.softmax(-1)
            # print(logit.shape)
            maxes = logit.max(-1)
            # print(raw)
            from bethesda_structs.plugin import FNVPlugin
            res_data['id'].extend([x[-1] for x in raw])
            res_data['gt_label'].extend([x[1] for x in raw])
            res_data['pred_label'].extend(maxes.indices.detach().cpu().tolist())
            res_data['pred_logit'].extend(logit.detach().cpu().tolist())
            res_data['prob'].extend(maxes.values.detach().cpu().tolist())
            res_data['losses'].extend([loss_value for _ in range(len(raw))])

        # if len(losses) > 0 and abs(loss_value - losses[-1]) > 0.5:
        #     log_info(loggers[2], 'Huge Loss Change Detected {}\n{}'.format(loss_value - losses[-1], raw))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        losses.append(loss_value)
        log_info(train_logger, '{} Iter Loss {} Time {}'.format(step, loss_value, time.time() - step_time))
    # print('logit', len(logits))
    return losses, loss, pd.DataFrame(res_data)


def train(model, dataset, batch_size, epochs, epoch_iter, learning_rate=1e-2, weight_decay=1e-4,
          save_path=None, from_checkpoint=False, continue_train=False, tokenizer=None, data_func=lambda x: x,
          all_config=None):
    from global_constants import ConfigEnums as ce
    loss_logger, train_logger = loggers.loss_logger, loggers.train_logger
    no_decay = ['bias', 'LayerNorm.weight']
    # device_ids = list(range(n_gpus))
    # SGD
    # fix learning rate
    optimizer_params = [{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                         'weight_decay': weight_decay},
                        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                         'weight_decay': 0.0}]
    optimizer = tfm.AdamW(optimizer_params, lr=learning_rate)
    scheduler = tfm.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100,
                                                    num_training_steps=epochs * epoch_iter)
    res_train_data = pd.DataFrame(columns=['id', 'pred_label', 'pred_logit', 'gt_label', 'prob', 'losses'])
    losses = []
    # logits: List[Tuple[Tensor, Tensor]] = []
    # gpu = GPUtil.getGPUs()[0]
    # n = max(len(dataset) // 9000, 1)
    # print(batch_size)
    if from_checkpoint:
        epoch, mini_epoch, model_state, optimizer_state, scheduler_state, loss = load_checkpoint(
            save_path + 'checkpoint.pt')
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(scheduler_state)
        if continue_train:
            epochs = epochs - epoch + 1
    model = nn.DataParallel(model)
    model.train()
    for e in range(epochs):
        data_loader = DataLoader(dataset, shuffle=False, batch_size=batch_size, collate_fn=lambda x: x)
        epoch_start = time.time()
        loss_value, loss, train_data = train_one_epoch(data_loader, model, optimizer, scheduler,
                                                       data_process_func=data_func, tok=tokenizer)
        losses.extend(loss_value)
        res_train_data = res_train_data.append(train_data)
        # print(res_train_data)
        if save_path is not None and (all_config is None or all_config[ce.save_checkpoint]):
            get_module_from_parallel(model).save_pretrained(save_path)
            if tokenizer is not None:
                tokenizer.save_pretrained(save_path)
            check_point = {
                'model': model,
                'epoch': e,
                'optimizer': optimizer,
                'scheduler': scheduler,
                'loss': loss
            }
            save_checkpoint(save_path + 'checkpoint.pt', check_point)
            log_info(loss_logger, 'saved models for in epoch {}'.format(e))
        loss_seg = losses[e * epoch_iter:]
        log_info(train_logger, '-' * 50)
        log_info(train_logger, 'Epoch {}, Mean Loss {}, Min Loss {}'.format(e, np.mean(loss_seg), np.min(loss_seg)))
        time_diff = time.time() - epoch_start
        log_info(train_logger,
                 'Time {}, Epoch Time {}, Avg Iter Time {}'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                                                                   time_diff, time_diff / epoch_iter))
    # print('all_logits', len(logits))
    return model, np.array(losses), res_train_data
