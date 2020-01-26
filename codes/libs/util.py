import time
import traceback
from typing import Sequence

import numpy as np
from torch import nn

from .global_constants import *
from .loggers import *


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


def get_tensor_batch(batch):
    if all([x.shape[0] == 0 for x in batch]):
        batch = torch.zeros(len(batch), 1, dtype=torch.long)
        labels = ignore_index * torch.ones(*batch.shape, dtype=torch.long)
        attn_mask = torch.zeros(*batch.shape, dtype=torch.long)
        return batch, labels, attn_mask

    # print(batch)
    # print('batch1:{}'.format(batch), [x.shape for x in batch])
    max_len = max(batch, key=lambda x: x.shape[-1]).shape[-1]
    attn_mask = torch.ones(len(batch), max_len, dtype=torch.float16)
    labels = torch.zeros(len(batch), max_len, dtype=torch.long)
    batch = [x[0] if len(x.shape) > 1 else x for x in batch]
    # print('batch2:{}'.format(batch), [x.shape for x in batch])
    for i in range(len(batch)):
        sent = batch[i]
        attn_mask[i, len(sent):max_len] = 0
        # print('sent:{}'.format(sent), sent.shape)
        batch[i] = torch.cat((sent, torch.zeros(max_len - sent.shape[0], dtype=torch.long) + eos_token), dim=0)
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
    input_ids = data.get('input', None)

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
        if input_ids is not None:
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
    result_ids = cat_tensors(result_ids, padding=eos_token)
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
    if len(sent_data) == len(e1_data):
        result.update({'input': sent_data})
    return result


def get_model_output(model, data):
    # device = torch.device('cuda:0')
    for i in data:
        data[i] = data[i].cuda()
    # print([x.device for x in data.values()])
    # print(data)
    try:
        output = model(**data)
        return output[0].mean()
    except Exception as e:
        print('data: ', data)
        print([x.device for x in data.values()])
        print(traceback.print_exc())
        exit()


def train_one_epoch(dataloader, model, optimizer, scheduler, data_process_func):
    losses = []
    # print(len(dataloader))
    for step, raw in enumerate(dataloader):
        step_time = time.time()
        data = data_process_func(raw)
        log_info(cuda_logger,
                 'Allocated batches {}, {}'.format(cuda_mem_in_mb(), {k: v.shape for k, v in data.items()}))
        loss = get_model_output(model, data)

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
    for step, raw in enumerate(dataloader):
        step_time = time.time()
        data = data_process_func(raw)
        log_info(cuda_logger,
                 'Allocated batches {}, {}'.format(cuda_mem_in_mb(), {k: v.shape for k, v in data.items()}))
        with torch.no_grad():
            loss = get_model_output(model, data)
            loss_value = loss.item()
        eval_loss += loss_value
        eval_steps += 1
        perplex_value = torch.exp(torch.tensor(eval_loss / eval_steps)).item()
        perplexities.append(perplex_value)
        losses.append(loss_value)
        log_info(train_logger, '{} Iter Loss {} Perplexity {} Time {}'.format(step, loss_value, perplex_value,
                                                                              time.time() - step_time))
    return losses, perplexities, eval_loss, eval_steps


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
