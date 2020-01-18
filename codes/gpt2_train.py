import time
from datetime import datetime

import numpy as np
import torch
import transformers as tfm
from torch import nn
from torch.utils.data import DataLoader

from loggers import log_info, cuda_mem_in_mb
from util import get_tensor_batch, get_re_data, get_model_output


def train_normal(model, dataset, batch_size, epochs, epoch_iter, learning_rate=1e-2, weight_decay=1e-4,
                 loggers=(None, None, None), n_gpus=1, device=None):
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
    data_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=lambda x: x)
    model = nn.DataParallel(model, device_ids)
    model.train()
    for e in range(epochs):
        epoch_start = time.time()
        for step, raw in enumerate(data_loader):
            batch, labels, attn_mask = get_tensor_batch(raw)
            batch.to(device)
            labels.to(device)
            attn_mask.to(device)
            log_info(loggers[0], 'Allocated batches {}, {}, {}, {}'.format(cuda_mem_in_mb(), batch.shape, labels.shape,
                                                                           attn_mask.shape))
            # log_info(cuda_logger,
            #          'GPU Free {} Used {} Total {}'.format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryTotal))
            # optimizer.zero_grad()
            outputs = model(batch, labels=labels, attention_mask=attn_mask)
            # print(outputs[0].shape)
            loss = outputs[0].mean()

            log_info(loggers[1], '{} Iter Loss {}'.format(step, loss.item()))
            loss_value = loss.item()
            if len(losses) > 0 and abs(loss_value - losses[-1]) > 0.5:
                log_info(loggers[2], 'Huge Loss Change Detected {}\n{}'.format(loss_value - losses[-1], raw))
                # continue
            loss.backward()
            # if (step + 1) % 10 == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            losses.append(loss_value)
            if step >= epoch_iter:
                break
        log_info(loggers[1], '----------------------------------------------------')
        # print(e * epoch_iter, (e + 1) * epoch_iter, losses[e])
        log_info(loggers[1],
                 'Epoch {}, Mean Loss {}, Min Loss {}'.format(e, np.mean(losses[e:e + epoch_iter]),
                                                              np.min(losses[e: e + epoch_iter])))
        time_diff = time.time() - epoch_start
        log_info(loggers[1],
                 'Time {}, Epoch Time {}, Avg Iter Time {}'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                                                                   time_diff, time_diff / epoch_iter))
    return model, torch.tensor(losses)


def train_re(model, tokenizer, entities, idx, sents, batch_size, epochs, epoch_iter, learning_rate=1e-2,
             weight_decay=1e-4, loggers=(None, None, None), n_gpus=1, device=None, max_len=200):
    no_decay = ['bias', 'LayerNorm.weight']
    device_ids = list(range(n_gpus))
    # SGD
    # fix learning rate
    #
    sents = open(sents, 'r')
    optimizer_params = [{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                         'weight_decay': weight_decay},
                        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                         'weight_decay': 0.0}]
    optimizer = tfm.AdamW(optimizer_params, lr=learning_rate)
    scheduler = tfm.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100,
                                                    num_training_steps=epochs * epoch_iter)
    losses = []
    # gpu = GPUtil.getGPUs()[0]
    data_loader = DataLoader(idx, batch_size=batch_size, collate_fn=lambda x: x)
    model = nn.DataParallel(model, device_ids)
    model.train()

    for e in range(epochs):
        epoch_start = time.time()
        sents.seek(idx.pos)
        sent_i = 0
        stored_sent = sents.readline()
        for step, raw in enumerate(data_loader):
            # print(raw)

            data, sent_i, stored_sent = get_re_data(raw, sents, entities, max_len, sent_i, stored_sent, tokenizer)
            # print(list(map(lambda x: x.shape, data)))
            loss = get_model_output(model, data)

            log_info(loggers[1], '{} Iter Loss {}'.format(step, loss.item()))
            loss_value = loss.item()
            if len(losses) > 0 and abs(loss_value - losses[-1]) > 0.5:
                log_info(loggers[2], 'Huge Loss Change Detected {}\n{}'.format(loss_value - losses[-1], raw))
                # continue
            loss.backward()
            print('backward achieved')
            # if (step + 1) % 10 == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            losses.append(loss_value)
        log_info(loggers[1], '----------------------------------------------------')
        # print(e * epoch_iter, (e + 1) * epoch_iter, losses[e])
        log_info(loggers[1],
                 'Epoch {}, Mean Loss {}, Min Loss {}'.format(e, np.mean(losses[e:e + epoch_iter]),
                                                              np.min(losses[e: e + epoch_iter])))
        time_diff = time.time() - epoch_start
        log_info(loggers[1],
                 'Time {}, Epoch Time {}, Avg Iter Time {}'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                                                                   time_diff, time_diff / epoch_iter))
    sents.close()
    return model, torch.tensor(losses)
