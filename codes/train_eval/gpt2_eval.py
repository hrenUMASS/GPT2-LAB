import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from libs import get_column
from libs import get_model_output, process_re_data, encode
from libs import log_info, cuda_mem_in_mb
from libs import loggers
from .sequence_sampling import sample_sequence_entity, get_seq_prob

import transformers as tfm

_tok = tfm.GPT2Tokenizer.from_pretrained('gpt2')


def eval_prob_one_epoch(dataloader, gpt2, model, length, num_samples, data_process_func, tokenizer=None):
    from global_constants import main_device
    result = pd.DataFrame(columns=['e1', 'e2', 'sent', 'log_prod_prob', 'sample_sent'])
    sample_logger = loggers.sample_logger
    max_sample = 32
    divs = num_samples // max_sample
    saps = [max_sample] * divs
    if sum(saps) < num_samples:
        saps.append(num_samples - divs * max_sample)
    for step, raw in enumerate(dataloader):
        data = data_process_func(raw)
        if data is None:
            continue
        for i in range(len(data['e1'])):
            step_time = time.time()
            # print(data)
            e1l, e2l = data['e1'][i], data['e2'][i]
            # print(e1l, e2l)
            # print(tokenizer)
            # print(encode(tokenizer, e1l), encode(tokenizer, e2l))
            # e1, e2 = encode(tokenizer, e1l).to(main_device), encode(tokenizer, e2l).to(main_device)
            e1, e2 = e2l.to(main_device), e1l.to(main_device)
            # e1l, e2l = e2l, e1l
            e1l, e2l = tokenizer.decode(e1.tolist()), tokenizer.decode(e2.tolist())
            sents = []
            sent = []
            gen_time = time.time()
            print('sampling {}, {}'.format(e1l, e2l))
            for ns in saps:
                # print(length, ns)
                sent_temp = sample_sequence_entity(model, length, e1, e2, num_samples=ns, top_k=5)
                if sent_temp is None:
                    continue
                sent_temp = sent_temp.cpu()
                sent.append(sent_temp)
            print('gen_time: {}'.format(time.time() - gen_time))
            # print(sent)
            eval_time = time.time()
            for s in sent:
                for l in range(s.shape[0]):
                    sl = tokenizer.decode(s[l].tolist())
                    if e1l in sl and e2l in sl:
                        sents.append(s[l])
            e1 = e1.cpu()
            e2 = e2.cpu()
            sl = len(sents)
            idx = data['idx'][i]
            res_data = {'e1': [idx[0]] * sl, 'e2': [idx[1]] * sl, 'sent': sents,
                        'log_prod_prob': [], 'sample_sent': [idx[2]] * sl}
            if sl > 0:
                divs = sl // max_sample
                paps = [max_sample] * divs
                if sum(paps) < sl:
                    paps.append(sl - divs * max_sample)
                for j, pap in enumerate(paps):
                    temp_data = {'e1': [e1] * pap, 'e2': [e2] * pap,
                                 'sent': sents[j * max_sample: j * max_sample + pap], 'idx': [idx] * pap}
                    probs = get_seq_prob(gpt2, temp_data, data_func=process_re_data)
                    res_data['log_prod_prob'].extend(get_column(probs, 1))
                    # res_data['loss'].extend(get_column(probs, 2))

            result = pd.concat([result, pd.DataFrame(res_data)])
            print('eval_time: {}'.format(time.time() - eval_time))
            log_info(sample_logger, 'Sampled {} sents for e1 {}, e2 {}'.format(len(sents),
                                                                               tokenizer.decode(e1.tolist()),
                                                                               tokenizer.decode(e2.tolist())))
            print('tot time: {}, avg: {}'.format(time.time() - step_time, (time.time() - step_time) / num_samples))
    return result


def eval_sequences(gpt2, model, dataset, num_samples, max_len, data_func=lambda x: x, tokenizer=None):
    # sample_logger = loggers.sample_logger
    data_loader = DataLoader(dataset, shuffle=False, batch_size=1, collate_fn=lambda x: x)
    ratios = eval_prob_one_epoch(data_loader, gpt2, model, max_len, num_samples, data_func, tokenizer=tokenizer)
    # log_info(sample_logger, 'Total ratio {}'.format(np.mean(tuple(x[-1] for x in ratios))))
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
    # ratio_prod, ratio_avg = [], []
    # gpt2_prod, gpt2_avg = [], []
    ratio_prob = {'idx': [], 'e1': [], 'e2': [], 'prob': []}
    gpt2_prob = {'idx': [], 'e1': [], 'e2': [], 'prob': []}
    for step, raw in enumerate(dataloader):
        # step_time = time.time()
        # print(raw)
        # print(raw)
        for i in range(len(raw)):
            raw[i] = [raw[i][0], raw[i][1],
                      encode(_tok, "a " + _tok.decode(raw[i][0]) + " is a " + _tok.decode(raw[i][1])),
                      raw[i][2]]
        # print(raw)
        # raw[i][-1] = raw[i][2]
        # raw[i][2] = raw[i][0] + " is a " + raw[i][1]
        # print(raw)
        data = data_func(raw)
        # print(data)
        # print(data)
        # print(data)
        probs = get_seq_prob(model, data, data_func=process_re_data)
        # print(probs)
        for i in range(len(probs)):
            # ep = np.concatenate((prob[0], prob[1]))
            idx = data['idx'][i]
            e2 = probs[i][1]
            # prob_avg = np.log(np.mean(ep)).item()
            # prob_prod = np.mean(np.log(ep)).item()
            # print(prob_avg, prob_prod, type(prob_avg), np.array(prob_avg), np.array(idx), idx, type(idx))
            # ratio_avg.append(np.append(np.array(idx), prob_avg))
            # ratio_prod.append(np.append(np.array(idx), prob_prod))
            ratio_prob['idx'].append(np.array(idx))
            ratio_prob['prob'].append(e2)
            ratio_prob['e1'].append(_tok.decode(raw[i][0]))
            ratio_prob['e2'].append(_tok.decode(raw[i][1]))
        # print(probs)
        probs = get_seq_prob(gpt2, data, data_func=process_re_data)
        # print(probs)

        for i in range(len(probs)):
            # ep = np.concatenate((prob[0], prob[1]))
            idx = data['idx'][i]
            e2 = probs[i][1]
            # prob_avg = np.log(np.mean(ep))
            # prob_prod = np.sum(np.log(ep))
            # gpt2_avg.append(np.append(np.array(idx), prob_avg))
            # gpt2_prod.append(np.append(np.array(idx), prob_prod))
            gpt2_prob['idx'].append(np.array(idx))
            gpt2_prob['prob'].append(e2)
            gpt2_prob['e1'].append(_tok.decode(raw[i][0]))
            gpt2_prob['e2'].append(_tok.decode(raw[i][1]))
        dl = len(probs)
        log_info(sample_logger, 'RE Sample {}, {}'.format(dl, raw))
    return pd.DataFrame(ratio_prob), pd.DataFrame(gpt2_prob)


def gpt2_eval(gpt2, model, dataset, batch_size=32, data_func=lambda x: x):
    sample_logger = loggers.sample_logger
    data_loader = DataLoader(dataset, shuffle=False, batch_size=batch_size, collate_fn=lambda x: x)
    re_prob, gpt2_prob = gpt2_eval_one_epoch(data_loader, gpt2, model, data_func)
    result = {'re_prob': re_prob, 'gpt2_prob': gpt2_prob}
    log_info(sample_logger, 'Total ratio {}'.format(result))
    return result
