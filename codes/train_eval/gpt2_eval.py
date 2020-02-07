import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from libs import get_model_output, in_tensor, encode, sample_sequence
from libs import log_info, cuda_mem_in_mb
from libs import loggers


def eval_one_epoch(dataloader, model, eval_loss, eval_steps, data_process_func):
    # print(len(dataloader))
    losses, perplexities = [], []
    cuda_logger, eval_logger = loggers.cuda_logger, loggers.validation_logger
    for step, raw in enumerate(dataloader):
        step_time = time.time()
        data = data_process_func(raw)
        log_info(cuda_logger,
                 'Allocated batches {}, {}'.format(cuda_mem_in_mb(), {k: v.shape for k, v in data.items()}))
        with torch.no_grad():
            loss = get_model_output(model, data)[0].mean()
            loss_value = loss.item()
        eval_loss += loss_value
        eval_steps += 1
        perplex_value = torch.exp(torch.tensor(eval_loss / eval_steps)).item()
        perplexities.append(perplex_value)
        losses.append(loss_value)
        log_info(eval_logger, '{} Iter Loss {} Perplexity {} Time {}'.format(step, loss_value, perplex_value,
                                                                             time.time() - step_time))
    return losses, perplexities, eval_loss, eval_steps


def sampling_one_epoch(dataloader, model, length, num_samples, data_process_func, tokenizer=None):
    sample_logger, cuda_logger = loggers.sample_logger, loggers.cuda_logger
    ratios = []
    ent_set = set()
    for step, raw in enumerate(dataloader):
        step_time = time.time()
        data = data_process_func(raw[0])
        idx = data['idx']
        if idx not in ent_set and idx[::-1] not in ent_set:
            e1, e2 = data['e1'], data['e2']
            e1b = encode(tokenizer, e1) if isinstance(e1, str) else e1
            e2b = encode(tokenizer, e2) if isinstance(e2, str) else e2
            sents = sample_sequence(model, length, e1b, e2b, num_samples=num_samples)
            in_sent = 0
            for i in range(sents.shape[0]):
                if in_tensor(sents[i], e1) and in_tensor(sents[i], e2):
                    in_sent += 1
            ratio = in_sent / int(sents.shape[0])
            ratios.append((e1, e2, ratio))
            log_info(sample_logger, 'e1 {} e2 {} ratio {} time {}'.format(e1, e2, ratio, time.time() - step_time))
            ent_set.add((e1, e2))
    return ratios


def eval_sequences(model, dataset, num_samples, max_length, data_func=lambda x: x, tokenizer=None):
    sample_logger = loggers.sample_logger
    data_loader = DataLoader(dataset, shuffle=False, batch_size=1, collate_fn=lambda x: x)
    ratios = sampling_one_epoch(data_loader, model, max_length, num_samples, data_func, tokenizer=tokenizer)
    log_info(sample_logger, 'Total ratio {}'.format(np.mean(tuple(x[-1] for x in ratios))))
    return ratios


def evaluate(model, dataset, batch_size, epochs, data_func=lambda x: x):
    validation_logger = loggers.validation_logger
    eval_loss, eval_steps = 0, 0
    losses, perplexities = [], []
    model.eval()
    for e in range(epochs):
        data_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=lambda x: x)
        epoch_iter = len(data_loader)
        loss, perp, eval_loss, eval_steps = eval_one_epoch(data_loader, model, eval_loss, eval_steps, data_func)
        # print(len(losses))
        losses.extend(loss)
        perplexities.extend(perp)
        loss_seg = losses[e * epoch_iter:]
        # print(len(loss), len(losses), e * epoch_iter)
        log_info(validation_logger, '----------------------------------------------------')
        log_info(validation_logger,
                 'Epoch {}, Mean Loss {}, Min Loss {}, Accum Loss {}'.format(e, np.mean(loss_seg), np.min(loss_seg),
                                                                             eval_loss / eval_steps))
    eval_loss /= eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))
    log_info(validation_logger, 'Final perplexity {}'.format(perplexity))
    return perplexity, torch.tensor(perplexities), torch.tensor(losses)
