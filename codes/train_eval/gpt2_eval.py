import copy
import time

import numpy as np
import pandas as pd
import torch
import transformers as tfm
from torch import nn
from torch.utils.data.dataloader import DataLoader

from libs import get_model_output, process_re_data, get_between
from libs import log_info, cuda_mem_in_mb
from libs import loggers
from .sequence_sampling import sample_sequence_entity, get_seq_prob, sample_classifier_sequence

_tok = tfm.GPT2Tokenizer.from_pretrained('gpt2')


def eval_prob_one_epoch(dataloader, gpt2, model, length, num_samples, data_process_func, tokenizer=None):
    from global_constants import main_device
    result = pd.DataFrame(columns=['e1', 'e2', 'sent', 'log_prod_prob', 'sample_sent'])
    result_non = pd.DataFrame(columns=['e1', 'e2', 'sent', 'sample_sent'])
    sample_logger = loggers.sample_logger
    max_sample = 16
    divs = num_samples // max_sample
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
            sents_non = []
            gen_time = time.time()
            print('sampling {}, {}, {}, {}, {}'.format(e1l, e2l, length, num_samples, max_sample))
            sent = sample_sequence_entity(model, length, e1, e2, num_samples=num_samples,
                                          max_size=max_sample, top_k=10, temperature=1).cpu()

            print('gen_time: {}'.format(time.time() - gen_time))
            # print(sent.shape)
            eval_time = time.time()
            for s in sent:
                sl = tokenizer.decode(s)
                if e1l in sl and e2l in sl:
                    sents.append(s)
                else:
                    sents_non.append(s)

            e1 = e1.cpu()
            e2 = e2.cpu()
            sl = len(sents)
            sl_non = len(sents_non)
            idx = data['idx'][i]

            def construct_res(leng, sentences):
                return {'e1': [idx[1]] * leng, 'e2': [idx[2]] * leng, 'sent': sentences,
                        'log_prod_prob': [], 'sample_sent': [idx[3]] * leng}

            res_data = construct_res(sl, sents)
            res_non_data = construct_res(sl_non, sents_non)
            del res_non_data['log_prod_prob']

            if sl > 0:
                divs = sl // max_sample
                paps = [max_sample] * divs
                if sum(paps) < sl:
                    paps.append(sl - divs * max_sample)
                for j, pap in enumerate(paps):
                    temp_data = {'e1': [e1] * pap, 'e2': [e2] * pap,
                                 'sent': sents[j * max_sample: j * max_sample + pap], 'idx': [idx] * pap}
                    probs = get_seq_prob(gpt2, temp_data, data_func=process_re_data, mode='e1,e2,loss,sent,max')
                    # res_data['log_prod_prob'].extend(get_column(probs, 1))
                    # res_data['loss'].extend(get_column(probs, 2))
                    res_data['log_prod_prob'].extend(probs)

            result = pd.concat([result, pd.DataFrame(res_data)])
            result_non = pd.concat([result_non, pd.DataFrame(res_non_data)])
            print('eval_time: {}'.format(time.time() - eval_time))
            log_info(sample_logger, 'Sampled {} sents for e1 {}, e2 {}'.format(len(sents),
                                                                               tokenizer.decode(e1.tolist()),
                                                                               tokenizer.decode(e2.tolist())))
            print('tot time: {}, avg: {}'.format(time.time() - step_time, (time.time() - step_time) / num_samples))
    return {'result': result.reset_index(drop=True), 'result_non': result_non.reset_index(drop=True)}


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
    ratio_prob = {'idx': [], 'e1': [], 'e2': [], 'full_sent': [], 'text_bt': [],
                  'prob_sent': [], 'prob_bt': [],
                  'text_max_prob': [], 'loss': []}
    gpt2_prob = copy.deepcopy(ratio_prob)
    for step, raw in enumerate(dataloader):
        # step_time = time.time()
        # print(raw)
        # print(raw)
        # for i in range(len(raw)):
        #     raw[i] = [raw[i][0], raw[i][1],
        #               encode(_tok, "a " + _tok.decode(raw[i][0]) + " is a " + _tok.decode(raw[i][1])),
        #               raw[i][2]]
        # print(raw)
        # raw[i][-1] = raw[i][2]
        # raw[i][2] = raw[i][0] + " is a " + raw[i][1]
        # print(raw)
        data = data_func(raw)

        # print('data: ', data)
        # print('raw: ', raw)

        def get_prob(f_model, f_raw, f_data, f_data_func, f_table):
            probs = get_seq_prob(f_model, f_data, data_func=f_data_func, mode='loss,sent,max')

            for i in range(len(probs)):
                p = probs[i]
                # ep = np.concatenate((prob[0], prob[1]))
                idx = data['idx'][i]
                # e2 = probs[i][1]
                loss, sent_p, max_s = p['loss'], p['sent'], p['max']
                # prob_avg = np.log(np.mean(ep)).item()
                # prob_prod = np.mean(np.log(ep)).item()
                # print(prob_avg, prob_prod, type(prob_avg), np.array(prob_avg), np.array(idx), idx, type(idx))
                # ratio_avg.append(np.append(np.array(idx), prob_avg))
                # ratio_prod.append(np.append(np.array(idx), prob_prod))
                e1, e2, sent = f_raw[i][0], f_raw[i][1], f_raw[i][2]
                a, b = get_between(e1, e2, sent, inclusive=False)
                f_table['idx'].append(np.array(idx))
                f_table['e1'].append(_tok.decode(raw[i][0]))
                f_table['e2'].append(_tok.decode(raw[i][1]))
                f_table['full_sent'].append(_tok.decode(sent))
                f_table['text_bt'].append(_tok.decode(sent[a:b]))
                f_table['prob_sent'].append(sent_p)
                f_table['prob_bt'].append(sent_p[a:b])
                f_table['text_max_prob'].append(_tok.decode(max_s))
                f_table['loss'].append(loss)

        get_prob(model, raw, data, process_re_data, ratio_prob)
        get_prob(gpt2, raw, data, process_re_data, gpt2_prob)

        dl = len(raw)
        log_info(sample_logger, 'RE Sample {}'.format(dl))
    return {'re': pd.DataFrame(ratio_prob), 'gpt2': pd.DataFrame(gpt2_prob)}


def gpt2_eval(gpt2, model, dataset, batch_size=32, data_func=lambda x: x):
    sample_logger = loggers.sample_logger
    data_loader = DataLoader(dataset, shuffle=False, batch_size=batch_size, collate_fn=lambda x: x)
    result = gpt2_eval_one_epoch(data_loader, gpt2, model, data_func)
    # result = {'re_prob': re_prob, 'gpt2_prob': gpt2_prob}
    log_info(sample_logger, 'Total ratio {}'.format(result))
    return result


def classifier_eval(model: nn.Module, dataset: torch.utils.data.Dataset, num_samples: int, data_func=lambda x: x,
                    tokenizer: tfm.PreTrainedTokenizer = None):
    # sampler_logger = loggers.sample_logger
    data_loader = DataLoader(dataset, shuffle=False, batch_size=1, collate_fn=lambda x: x)
    result = classifier_eval_one_epoch(data_loader, model, num_samples, data_func, tokenizer)
    return result


def classifier_eval_one_epoch(dataloader: DataLoader, model: nn.Module, num_samples: int, data_process_func,
                              tokenizer: tfm.PreTrainedTokenizer = None):
    import global_constants
    main_device = global_constants.main_device
    result = pd.DataFrame(columns=['e1', 'e2', 'full_sent', 'text_bt', 'pred_label', 'prob', 'e1i', 'e2i'])
    sample_logger = loggers.sample_logger

    res_data = {'e1': [], 'e2': [], 'full_sent': [], 'text_bt': [],
                'pred_label': [], 'prob': [],
                'e1i': [], 'e2i': []}

    for step, raw in enumerate(dataloader):
        data = data_process_func(raw)
        if data is None:
            continue

        for i in range(len(data['input_ids'])):
            step_time = time.time()
            gen_time = time.time()
            sent = data['input_ids'][i]
            attn_mask = data['attention_mask'][i]
            pos_ids = data['position_ids'][i]
            e1, e2 = raw[i][0], raw[i][1]
            sent = sent.to(main_device)
            if sent.shape[-1] == 0:
                continue
            input_data = {'input_ids': sent.unsqueeze(0), 'attention_mask': attn_mask.unsqueeze(0),
                          'position_ids': pos_ids.unsqueeze(0)}

            e1l, e2l = tokenizer.decode(e1.tolist()), tokenizer.decode(e2.tolist())
            sentl = tokenizer.decode(sent.tolist())

            print('sampling {}, {}'.format(e1l, e2l))

            label = sample_classifier_sequence(model, input_data)
            print('gen_time: {}'.format(time.time() - gen_time))
            # print(sent)
            eval_time = time.time()

            idx = raw[i][-1]

            a, b = get_between(e1, e2, sent, inclusive=False)

            res_data['e1'].append(e1l)
            res_data['e2'].append(e2l)
            res_data['full_sent'].append(sentl)
            res_data['text_bt'].append(_tok.decode(sent[a:b]))
            res_data['pred_label'].append(label[0][0])
            res_data['prob'].append(label[0][1])
            res_data['e1i'].append(idx[1])
            res_data['e2i'].append(idx[2])

            print('eval_time: {}'.format(time.time() - eval_time))
            log_info(sample_logger, 'Classified e1 {}, e2 {}, sent {}'.format(e1l, e2l, sentl))
            print('tot time: {}, avg: {}'.format(time.time() - step_time, (time.time() - step_time) / num_samples))
    return result.append(pd.DataFrame(res_data))
