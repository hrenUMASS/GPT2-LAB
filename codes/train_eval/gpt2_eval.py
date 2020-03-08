import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from libs import get_model_output, in_tensor, encode, process_re_data
from libs import log_info, cuda_mem_in_mb
from libs import loggers
from .sequence_sampling import sample_sequence_entity, get_seq_prob


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
            sents = sample_sequence_entity(model, length, e1b, e2b, num_samples=num_samples, top_k=5)
            in_sent = 0
            for i in range(sents.shape[0]):
                if in_tensor(sents[i], e1) and in_tensor(sents[i], e2):
                    in_sent += 1
            ratio = in_sent / int(sents.shape[0])
            ratios.append((e1, e2, ratio))
            log_info(sample_logger, 'e1 {} e2 {} ratio {} time {}'.format(e1, e2, ratio, time.time() - step_time))
            ent_set.add((e1, e2))
    return ratios


def eval_sequences(model, dataset, num_samples, max_len, data_func=lambda x: x, tokenizer=None):
    sample_logger = loggers.sample_logger
    data_loader = DataLoader(dataset, shuffle=False, batch_size=1, collate_fn=lambda x: x)
    ratios = sampling_one_epoch(data_loader, model, max_len, num_samples, data_func, tokenizer=tokenizer)
    log_info(sample_logger, 'Total ratio {}'.format(np.mean(tuple(x[-1] for x in ratios))))
    return ratios


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


def gpt2_eval_one_epoch(dataloader, gpt2, model, data_func):
    sample_logger = loggers.sample_logger
    ratio_prod, ratio_avg = [], []
    gpt2_prod, gpt2_avg = [], []
    for step, raw in enumerate(dataloader):
        step_time = time.time()
        # print(raw)
        data = data_func(raw)
        # print(data)
        probs = get_seq_prob(model, data, data_func=process_re_data)
        # print(probs)
        for idx, prob in probs.items():
            # ep = np.concatenate((prob[0], prob[1]))
            ep = prob[1]
            prob_avg = np.log(np.mean(ep))
            prob_prod = np.sum(np.log(ep))
            # print(prob_avg, prob_prod, type(prob_avg), np.array(prob_avg), np.array(idx), idx, type(idx))
            ratio_avg.append(np.append(np.array(idx), prob_avg))
            ratio_prod.append(np.append(np.array(idx), prob_prod))
        # print(probs)
        probs = get_seq_prob(gpt2, data, data_func=process_re_data)
        # print(probs)

        for idx, prob in probs.items():
            # ep = np.concatenate((prob[0], prob[1]))
            ep = prob[1]
            prob_avg = np.log(np.mean(ep))
            prob_prod = np.sum(np.log(ep))
            gpt2_avg.append(np.append(np.array(idx), prob_avg))
            gpt2_prod.append(np.append(np.array(idx), prob_prod))
        dl = len(probs)
        # log_info(sample_logger, 'RE Sample {}, ratio prod {}, {}, ratio mean {}, {}, time {}'.format(
        #         #     list(data['idx']), ratio_prod[-dl:][-1], gpt2_prod[-dl:][-1], ratio_avg[-dl:][-1], gpt2_avg[-dl:][-1],
        #         #     time.time() - step_time))
        log_info(sample_logger, 'RE Sample {} ratio prod {}, {}, ratio mean {}, {}'.format(
            dl, [x[-1] for x in ratio_prod[-dl:]], [x[-1] for x in gpt2_prod[-dl:]],
            [x[-1] for x in ratio_avg[-dl:]], [x[-1] for x in gpt2_avg[-dl:]]
        ))
    return np.array(ratio_prod), np.array(ratio_avg), np.array(gpt2_prod), np.array(gpt2_avg)


def gpt2_eval(gpt2, model, dataset, batch_size=32, data_func=lambda x: x):
    sample_logger = loggers.sample_logger
    data_loader = DataLoader(dataset, shuffle=False, batch_size=batch_size, collate_fn=lambda x: x)
    re_prod, re_avg, gpt2_prod, gpt2_avg = gpt2_eval_one_epoch(data_loader, gpt2, model, data_func)
    result = {'re_prod': re_prod, 're_avg': re_avg, 'gpt2_prod': gpt2_prod, 'gpt2_avg': gpt2_avg}
    log_info(sample_logger, 'Total ratio {}'.format(result))
    return result
