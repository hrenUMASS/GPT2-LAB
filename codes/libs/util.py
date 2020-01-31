import time
import traceback
from typing import Sequence

import numpy as np
import torch.nn.functional as F
from torch import nn

from . import loggers
from .global_constants import *
from .loggers import log_info, cuda_mem_in_mb


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, length, e1, e2, num_samples=1, temperature=1, top_k=0, top_p=0.0, repetition_penalty=1.0):
    e1, e2 = torch.tensor(e1, dtype=torch.long).cuda(), torch.tensor(e2, dtype=torch.long).cuda()
    e1, e2 = e1.unsqueeze(0).repeat(num_samples, 1), e2.unsqueeze(0).repeat(num_samples, 1)
    generated = torch.zeros(num_samples, 0).long().cuda()
    data = {'e1': e1, 'e2': e2, 'sent': generated}
    with torch.no_grad():
        for _ in range(length):

            inputs = process_re_data(data)
            outputs = get_model_output(model, inputs)
            next_token_logits = outputs[1][:, -1, :] / (temperature if temperature > 0 else 1.)

            # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
            for i in range(num_samples):
                for _ in set(generated[i].tolist()):
                    next_token_logits[i, _] /= repetition_penalty

            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0:  # greedy sampling:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

            generated = torch.cat((generated, next_token), dim=1)
    return generated


def get_index(ori, cmp):
    index = -1
    for i in range(ori.shape[0]):
        if ori[i] == cmp[0]:
            index = i
            for j in range(cmp.shape[0]):
                if ori[index + j] != cmp[j]:
                    index = -1
                    break
            if index != -1:
                return index
    return index


def in_tensor(ori, dst):
    ori = ori.tolist()
    dst = dst.tolist()
    n = len(dst)
    return any(dst == ori[i:i + n] for i in range(len(ori) - n + 1))


def encode(tokenizer, elem, add_eos=False, **kwargs):
    if isinstance(elem, str):
        result = tokenizer.encode(elem, return_tensors='pt', **kwargs).long()
        if len(result.shape) > 1:
            result = result[0]
        if add_eos:
            result = torch.cat((result, torch.tensor([tokenizer.eos_token_id], dtype=torch.long)))
        return result
    elif isinstance(elem, Sequence) or isinstance(elem, int):
        return torch.tensor(tokenizer.convert_tokens_to_ids(elem))


def get_tensor_batch(batch, batch_size=32, max_len=512):
    if all([x.shape[0] == 0 for x in batch]):
        batch = eos_id + torch.zeros(batch_size, 1, dtype=torch.long)
        labels = ignore_index * torch.ones(*batch.shape, dtype=torch.long)
        attn_mask = torch.zeros(*batch.shape, dtype=torch.long)
        return batch, labels, attn_mask

    # print(batch)
    # print('batch1:{}'.format(batch), [x.shape for x in batch])
    max_len = min(max(batch, key=lambda x: x.shape[-1]).shape[-1], max_len)
    attn_mask = torch.ones(batch_size, max_len, dtype=torch.float16)
    labels = torch.zeros(batch_size, max_len, dtype=torch.long)
    batch = [x[0] if len(x.shape) > 1 else x for x in batch]
    # print('batch2:{}'.format(batch), [x.shape for x in batch])
    for i in range(batch_size):
        if i >= len(batch):
            batch.append(eos_id + torch.zeros(1, dtype=torch.long))
        sent = batch[i]
        attn_mask[i, len(sent):max_len] = 0
        # print('sent:{}'.format(sent), sent.shape)
        batch[i] = torch.cat((sent, torch.zeros(max_len - sent.shape[0], dtype=torch.long) + eos_id), dim=0)
        labels[i] = torch.cat((sent, torch.ones(max_len - sent.shape[0], dtype=torch.long) * ignore_index), dim=0)
    return torch.stack(batch), labels, attn_mask


def cat_tensors(tensor_list, padding=0):
    result = list(tensor_list)
    max_len = max(result, key=lambda x: x.shape[0]).shape[0]
    device = result[0].device
    dtype = result[0].dtype
    for i in range(len(result)):
        e = result[i]
        req_len = max_len - e.shape[0]
        req_shape = list(e.shape)
        req_shape[0] = req_len
        # print(e.shape, req_shape)
        result[i] = torch.cat((e, torch.zeros(*req_shape, dtype=dtype, device=device) + padding))
    # print(result)
    return torch.stack(result)


def get_module_from_parallel(module):
    if isinstance(module, nn.DataParallel):
        return get_module_from_parallel(module.module)
    return module


def type_ids(length, index=0, dtype=torch.long):
    return torch.zeros(length, dtype=dtype) + index


def pos_id(length):
    return torch.arange(0, length, dtype=torch.long)


def process_re_data(data):
    e1_ids, e2_ids = data['e1'], data['e2']
    input_ids = data.get('sent', None)

    result_ids = []
    tokens = []
    attns = []
    poses = []
    labels = []
    for i in range(len(e1_ids)):
        e1, e2 = e1_ids[i], e2_ids[i]
        e1l, e2l = e1.shape[0], e2.shape[0]
        e1p, e2p = pos_id(e1l), pos_id(e2l)
        e1a, e2a = type_ids(e1l, index=1, dtype=torch.float), type_ids(e2l, index=1, dtype=torch.float)
        ids = torch.cat((e1, e2))
        token = torch.cat((type_ids(e1l), type_ids(e2l, index=1)))
        pos = torch.cat((e1p, e2p))
        attn = torch.cat((e1a, e2a))
        lab = torch.cat((e1, e2))
        if input_ids is not None and input_ids.shape[0] > 0:
            in_ids = input_ids[i]
            inl = in_ids.shape[0]
            inp = pos_id(inl)
            ina = type_ids(inl, index=1, dtype=torch.float)
            ids = torch.cat((ids, in_ids))
            token = torch.cat((token, type_ids(in_ids.shape[0], index=2)))
            pos = torch.cat((pos, inp))
            attn = torch.cat((attn, ina))
            lab = torch.cat((lab, in_ids))
        result_ids.append(ids)
        tokens.append(token)
        attns.append(attn)
        poses.append(pos)
        labels.append(lab)
    result_ids = cat_tensors(result_ids, padding=eos_id)
    poses = cat_tensors(poses)
    tokens = cat_tensors(tokens, padding=2)
    attns = cat_tensors(attns)
    labels = cat_tensors(labels, padding=ignore_index)
    return {'input_ids': result_ids, 'attention_mask': attns, 'token_type_ids': tokens, 'position_ids': poses,
            'labels': labels}


def get_re_data(data, max_len=np.inf, train=True, batch_size=32):
    empty = torch.zeros(0, dtype=torch.long)
    e1_data, e2_data = [], []
    sent_data = []
    for x in data:
        if len(data[0]) > 2:
            sent = x[2]
            # print('Testing entities in sentence')
            # print(x[0], x[1], x[2].shape)
            failed = False
            if not (in_tensor(sent, x[0]) and in_tensor(sent, x[1])) and train:
                print('Entity not in sentence\ne1={}\ne2={}\nsent={}\nidx={}'.format(x[0], x[1], sent, x[3]))
                failed = True
            failed |= (x[0].shape[0] < 1 or x[1].shape[0] < 1)
            if failed:
                sent_data.append(empty)
                e1_data.append(empty)
                e2_data.append(empty)
                continue
            adj_max_len = max_len - x[0].shape[0] - x[1].shape[0]
            if sent.shape[0] > adj_max_len:
                ei = max(get_index(sent, x[0]) + x[0].shape[0], get_index(sent, x[1]) + x[1].shape[0])
                print('ei index {}'.format(ei))

                if ei > adj_max_len:
                    sent_data.append(empty)
                    e1_data.append(empty)
                    e2_data.append(empty)
                    continue
                sent = sent[:adj_max_len]
            sent_data.append(sent)
        e1_data.append(x[0])
        e2_data.append(x[1])
    data_len = len(e1_data)
    rm_len = batch_size - data_len
    if rm_len > 0:
        e1_data.extend([empty] * rm_len)
        e2_data.extend([empty] * rm_len)
        sent_data.extend([empty] * rm_len)
    result = {'e1': e1_data, 'e2': e2_data}
    print(len(sent_data), len(e1_data))
    if len(sent_data) == len(e1_data):
        result.update({'sent': sent_data})
    return result


def get_model_output(model, data):
    # device = torch.device('cuda:0')
    for i in data:
        data[i] = data[i].to(main_device)
    # print([x.device for x in data.values()])
    # print(data)
    try:
        output = model(**data)
        return output
    except Exception as e:
        print('data: ', data)
        print([x.device for x in data.values()])
        print(traceback.print_exc())
        exit()


def train_one_epoch(dataloader, model, optimizer, scheduler, data_process_func):
    losses = []
    # print(len(dataloader))
    cuda_logger, train_logger = loggers.cuda_logger, loggers.train_logger
    for step, raw in enumerate(dataloader):
        step_time = time.time()
        data = data_process_func(raw)
        log_info(cuda_logger,
                 'Allocated batches {}, {}'.format(cuda_mem_in_mb(), {k: v.shape for k, v in data.items()}))
        loss = get_model_output(model, data)[0].mean()
        # loss = torch.tensor([0], requires_grad=True, dtype=torch.float) * torch.tensor([1], requires_grad=True,
        # dtype = torch.float)
        loss_value = loss.item()
        # if len(losses) > 0 and abs(loss_value - losses[-1]) > 0.5:
        #     log_info(loggers[2], 'Huge Loss Change Detected {}\n{}'.format(loss_value - losses[-1], raw))
        # continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        losses.append(loss_value)
        log_info(train_logger, '{} Iter Loss {} Time {}'.format(step, loss_value, time.time() - step_time))

    return losses, loss


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
            ratio = in_sent / sents.shape[0]
            ratios.append((e1, e2, ratio))
            log_info(sample_logger, 'e1 {} e2 {} ratio {} time {}'.format(e1, e2, ratio, time.time() - step_time))
            ent_set.add((e1, e2))


def save_checkpoint(save_path, model, epoch, mini_epoch, optimizer, scheduler, loss):
    check_point = {'epoch': epoch, 'mini_epoch': mini_epoch, 'model_state': model.state_dict(),
                   'optimizer_state': optimizer.state_dict(), 'scheduler_state': scheduler.state_dict(), 'loss': loss}
    torch.save(check_point, save_path)


def load_checkpoint(save_path, model_cls=None, optim_cls=None, sche_cls=None, model_param=None, optim_param=None,
                    sche_param=None):
    check_point = torch.load(save_path)
    model = check_point['model_state']
    optimizer = check_point['optimizer_state']
    scheduler = check_point['scheduler_state']
    loss = check_point['loss']
    epoch = check_point['epoch']
    mini_epoch = check_point['mini_epoch']
    if model_cls is not None:
        model = model_cls(**(model_param or {})).load_state_dict(model)
    if optim_cls is not None:
        optimizer = optim_cls(**(optim_param or {})).load_state_dict(optimizer)
    if sche_cls is not None:
        scheduler = sche_cls(**(sche_param or {})).load_state_dict(scheduler)
    return epoch, mini_epoch, model, optimizer, scheduler, loss


def get_dict(d, item, default):
    prepare_logger = loggers.prepare_logger
    if item not in d:
        log_info(prepare_logger, 'item {} not in config, using default {}'.format(item, default))
    return d.get(item, default)
