import inspect
import traceback
from typing import Sequence, Tuple, Union, AnyStr
import time
from datetime import datetime

import numpy as np
import torch
from torch import nn, LongTensor
import transformers as tfm
from transformers.modeling_utils import ModelOutput
from transformers.models.gpt2.modeling_gpt2 import (
    BaseModelOutputWithPastAndCrossAttentions, SequenceClassifierOutputWithPast
)

from . import loggers
from .loggers import log_info

_StrOrLT = Union[AnyStr, LongTensor]

tokenizer = tfm.models.gpt2.GPT2Tokenizer.from_pretrained('gpt2')


def get_column(arr, index):
    return [item[index] for item in arr]


def get_index(ori: _StrOrLT, comp: _StrOrLT):
    if isinstance(ori, str):
        return ori.index(comp)
    # print('index', ori, comp)
    if ori.shape[0] < comp.shape[0]:
        return -1
    index = -1
    for i in range(ori.shape[0] - comp.shape[0] + 1):
        if ori[i] == comp[0]:
            index = i
            for j in range(comp.shape[0]):
                if ori[i + j] != comp[j]:
                    index = -1
                    break
            if index != -1:
                return index
    return index


def parse_epoch_save(epoch_save):
    if epoch_save is None:
        return None
    result = []
    for e in epoch_save:
        if isinstance(e, Sequence):
            e[1] += 1
            if len(e) in (2, 3):
                result.extend(list(range(*e)))
            else:
                log_info(loggers.prepare_logger, 'Cannot parse epoch save range {}'.format(e))
        elif isinstance(e, int):
            result.append(e)
        else:
            log_info(loggers.prepare_logger, 'Cannot parse epoch save range {}'.format(e))
    return result


def safe_sql(cursor, command, arguments=None, fetch='all', size=1):
    prepare_logger = loggers.prepare_logger
    start_time = time.time()
    dtn = datetime.now()
    while True:
        try:
            if arguments is None:
                c = cursor.execute(command)
            else:
                c = cursor.execute(command, arguments)

            if fetch == 'all':
                res = c.fetchall()
                # print(cursor.execute('explain ' + command, arguments).fetchall())
                # if 'idx' not in command:
                #     print(command, arguments)
                #     command2 = command[:command.index('WHERE id ')] + 'WHERE id=?'
                #     print(command2)
                #     print(cursor.execute(command, arguments).fetchall(),
                #           repr(cursor.execute(command2, arguments[0]).fetchall()))
                # print('res', res)
            elif fetch == 'one':
                res = c.fetchone()
            else:
                res = c.fetchmany(size=size)
            return res
        except Exception as e:
            print(e)
            if time.time() - start_time > 60:
                log_info(prepare_logger, 'locked from {} interval {}'.format(dtn.strftime("%d/%m/%Y %H:%M:%S"),
                                                                             time.time() - start_time))
                print(e)
                if 'lock' not in str(e):
                    break
        continue


def split_array(arr, block=1000):
    arr = list(arr)
    seg = len(arr) // block
    if seg * block != len(arr):
        seg += 1
    res = []
    for i in range(seg):
        res.append(arr[block * i: block * (i + 1)])
    return res


def in_tensor(ori, dst):
    ori = ori.tolist()
    dst = dst.tolist()
    n = len(dst)
    return any(dst == ori[i:i + n] for i in range(len(ori) - n + 1))


def encode(tokenizer, elem, add_eos=False, **kwargs):
    if isinstance(elem, str):
        result = tokenizer.encode(text=elem, return_tensors='pt', **kwargs).long()
        # print(result)
        if len(result.shape) > 1:
            result = result[0]
        if add_eos:
            result = torch.cat((result, torch.tensor([tokenizer.eos_token_id], dtype=torch.long)))
        return result
    elif isinstance(elem, Sequence) or isinstance(elem, int):
        return torch.tensor(tokenizer.convert_tokens_to_ids(elem))


def get_tensor_batch(batch, batch_size=32, max_len=512):
    if isinstance(batch, dict):
        batch = batch['sent']
    from global_constants import eos_id, ignore_index
    device = batch.device
    if len(batch.shape) == 1:
        batch = batch.unsequeeze(0)
    if all([x.shape[0] == 0 for x in batch]):
        batch = eos_id + torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        labels = ignore_index * torch.ones(*batch.shape, dtype=torch.long, device=device)
        attn_mask = torch.zeros(*batch.shape, dtype=torch.long, device=device)
        return {'input_ids': batch, 'labels': labels, 'attention_mask': attn_mask}

    # print(batch)
    # print('batch1:{}'.format(batch), [x.shape for x in batch])
    max_len = min(max(batch, key=lambda x: x.shape[-1]).shape[-1], max_len)
    attn_mask = torch.ones(batch_size, max_len, dtype=torch.float16, device=device)
    labels = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    batch = [x[0] if len(x.shape) > 1 else x for x in batch]
    # print('batch2:{}'.format(batch), [x.shape for x in batch])
    for i in range(batch_size):
        if i >= len(batch):
            batch.append(eos_id + torch.zeros(1, dtype=torch.long, device=device))
        sent = batch[i]
        attn_mask[i, len(sent):max_len] = 0
        # print('sent:{}'.format(sent), sent.shape)
        batch[i] = torch.cat(
            (sent,
             torch.zeros(max_len - sent.shape[0], dtype=torch.long, device=device) + eos_id), dim=0)
        labels[i] = torch.cat(
            (sent,
             torch.ones(max_len - sent.shape[0], dtype=torch.long, device=device) * ignore_index), dim=0)
    return {'input_ids': torch.stack(batch), 'labels': labels, 'attention_mask': attn_mask}


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


def type_ids(length, index=0, dtype=torch.long, device='cpu'):
    return torch.zeros(length, dtype=dtype, device=device) + index


def pos_id(length, device='cpu'):
    return torch.arange(0, length, dtype=torch.long, device=device)


def process_re_data(data, between=False, inclusive=True, output_ent=True, e1t=1, e2t=1, st=2):
    from global_constants import eos_id, ignore_index
    e1_ids, e2_ids = data.get('e1', None), data.get('e2', None)
    input_ids = data.get('sent', None)
    if e1_ids is not None and len(e1_ids) > 0:
        device = e1_ids[0].device
    elif input_ids is not None and len(input_ids) > 0:
        device = input_ids[0].device
    else:
        return None
    length = 0
    if e1_ids is not None:
        length = len(e1_ids)
    if input_ids is not None and len(input_ids) > length:
        length = len(input_ids)
    result_ids = []
    tokens = []
    attns = []
    poses = []
    labels = []

    has_ents = e1_ids is not None and e2_ids is not None and len(e1_ids) == len(e2_ids) and len(e1_ids) > 0 and len(
        e2_ids) > 0
    has_sents = input_ids is not None and len(input_ids) > 0

    for i in range(length):
        ids = torch.zeros(0, dtype=torch.long, device=device)
        token = torch.zeros(0, dtype=torch.long, device=device)
        pos = torch.zeros(0, dtype=torch.long, device=device)
        attn = torch.zeros(0, dtype=torch.float, device=device)
        lab = torch.zeros(0, dtype=torch.long, device=device)
        if has_ents:
            e1, e2 = e1_ids[i], e2_ids[i]
            e1l, e2l = e1.shape[0], e2.shape[0]
            e1p, e2p = pos_id(e1l), pos_id(e2l)
            e1a, e2a = type_ids(e1l, index=1, dtype=torch.float), type_ids(e2l, index=1, dtype=torch.float)
            ids = torch.cat((e1, e2))
            token = torch.cat((type_ids(e1l, index=e1t), type_ids(e2l, index=e2t)))
            pos = torch.cat((e1p, e2p))
            attn = torch.cat((e1a, e2a))
            lab = torch.cat((e1, e2))
        if has_sents:
            in_ids = input_ids[i]
            inl = in_ids.shape[0]
            if between and has_ents:
                e1 = tokenizer.decode(e1)
                e2 = tokenizer.decode(e2)
                sent = tokenizer.decode(in_ids)
                a, b = get_between(e1, e2, sent, inclusive=inclusive)
                # print(sent)
                # print(sent[a:b])
                in_ids = encode(tokenizer, sent[a:b], add_eos=True,
                                add_prefix_space=True)
                inl = in_ids.shape[0]
                # print(in_ids)
                ina = type_ids(inl, index=1, dtype=torch.float)
                inp = pos_id(inl)
                # lab = ids
            else:
                inp = pos_id(inl)
                ina = type_ids(inl, index=1, dtype=torch.float)
            tok = type_ids(inl, index=st)
            if output_ent:
                ids = torch.cat((ids, in_ids))
                token = torch.cat((token, tok))
                pos = torch.cat((pos, inp))
                attn = torch.cat((attn, ina))
                lab = torch.cat((lab, in_ids))
            else:
                ids = in_ids
                token = tok
                pos = inp
                attn = ina
                lab = in_ids
        result_ids.append(ids)
        tokens.append(token)
        attns.append(attn)
        poses.append(pos)
        labels.append(lab)
    result_ids = cat_tensors(result_ids, padding=eos_id)
    poses = cat_tensors(poses)
    tokens = cat_tensors(tokens, padding=st)
    attns = cat_tensors(attns)
    labels = cat_tensors(labels, padding=ignore_index)
    if result_ids.shape[1] == 0:
        return None
    result = {'input_ids': result_ids, 'attention_mask': attns, 'token_type_ids': tokens, 'position_ids': poses,
              'labels': labels}
    # print(result)
    return result


def process_cls_data(data):
    from global_constants import eos_id, ignore_index
    result_ids = [x[0] for x in data]

    device = result_ids[0].device

    for x in result_ids:
        x.to(device)

    attns = []
    # tokens = []
    # poses = []
    labels = []
    # print(data)
    for i in range(len(data)):
        s = result_ids[i]
        l = len(s)
        attns.append(type_ids(l, index=1, dtype=torch.float))
        # tokens.append(type_ids(l, index=0))
        # poses.append(pos_id(l))
        labels.append(torch.zeros(1, dtype=torch.long))
        labels[-1][0] = data[i][1]

    result_ids = cat_tensors(result_ids, padding=eos_id)
    # poses = cat_tensors(poses)
    # tokens = cat_tensors(tokens)
    attns = cat_tensors(attns)
    labels = cat_tensors(labels, padding=ignore_index)
    labels = labels[:, 0]
    result = {'input_ids': result_ids, 'attention_mask': attns, 'labels': labels}
    return result


def get_re_data(data, max_len=np.inf, batch_size=32):
    empty = torch.zeros(0, dtype=torch.long)
    e1_data, e2_data = [], []
    sent_data = []
    # print('idx', [x[-1] for x in data])
    for x in data:
        if len(data[0]) > 3:
            sent = x[2]
            # failed = False
            # if not (in_tensor(sent, x[0]) and in_tensor(sent, x[1])):
            #     print('Entity not in sentence\ne1={}\ne2={}\nsent={}\nidx={}'.format(x[0], x[1], sent, x[-1]))
            #     failed = True
            failed = (x[0].shape[0] < 1 or x[1].shape[0] < 1)
            if failed:
                sent_data.append(empty)
                e1_data.append(empty)
                e2_data.append(empty)
                continue
            adj_max_len = max_len - x[0].shape[0] - x[1].shape[0]
            if sent.shape[0] > adj_max_len:
                ei = max(get_index(sent, x[0]) + x[0].shape[0], get_index(sent, x[1]) + x[1].shape[0])
                # print('ei index {}'.format(ei))
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
    result = {'e1': e1_data, 'e2': e2_data, 'idx': [x[-1] for x in data]}
    if len(sent_data) == len(e1_data):
        result.update({'sent': sent_data})
    return result


def get_model_output(model, data) -> Union[
    ModelOutput, Tuple, BaseModelOutputWithPastAndCrossAttentions, SequenceClassifierOutputWithPast]:
    from global_constants import main_device
    # device = torch.device('cuda:0')
    # for k, v in data.items():
    #     print(k, v)
    for i in data:
        if data[i] is not None and isinstance(data[i], torch.Tensor):
            req_grad = data[i].requires_grad
            data[i] = data[i].clone().detach().requires_grad_(req_grad).to(main_device)
    # print([x.device for x in data.values()])
    # print(data)
    try:
        output = model(**data)
        return output
    except Exception as e:
        print('data: ', data)
        for k, v in data.items():
            print(k, v.shape)
        print([x.device for x in data.values() if x is not None])
        print(traceback.print_exc())
        exit()


def save_checkpoint(save_path, check_point):
    check_point = {k: v.state_dict() if hasattr(v, 'state_dict') else v for k, v in check_point.items()}
    torch.save(check_point, save_path)


def load_checkpoint(save_path, model_cls=None, optim_cls=None, sche_cls=None, model_param=None, optim_param=None,
                    sche_param=None):
    check_point = torch.load(save_path)
    model = check_point['model_state']
    optimizer = check_point['optimizer_state']
    scheduler = check_point['scheduler_state']
    loss = check_point['loss']
    epoch = check_point['epoch']
    if model_cls is not None:
        model = model_cls(**(model_param or {})).load_state_dict(model)
    if optim_cls is not None:
        optimizer = optim_cls(**(optim_param or {})).load_state_dict(optimizer)
    if sche_cls is not None:
        scheduler = sche_cls(**(sche_param or {})).load_state_dict(scheduler)
    return epoch, model, optimizer, scheduler, loss


def get_params(config, func):
    return {k: v for k, v in config.items() if
            k in inspect.signature(func).parameters.keys()}


def get_between(e1: _StrOrLT, e2: _StrOrLT, sent: _StrOrLT, inclusive=False) -> Tuple[int, int]:
    if type(e1) != type(e2) or type(e2) != type(sent):
        raise TypeError('Type of e1, e2, sent not match')
    try:
        e1i, e2i = get_index(sent, e1), get_index(sent, e2)
        if e1i > e2i:
            e1i, e2i = e2i, e1i
            e1, e2 = e2, e1
        if inclusive:
            return e1i, e2i + len(e2)
        else:
            return e1i + len(e1), e2i
    except Exception as e:
        print('Cannot find substring/subtensor')
        print(e1, e2, sent)
        print(traceback.print_stack(e))
        exit(1)


def get_dict(d, item, default):
    prepare_logger = loggers.prepare_logger
    if item not in d:
        log_info(prepare_logger, 'item {} not in config, using default {}'.format(item, default))
    return d.get(item, default)


def get_config(config, item):
    from global_constants import default_values
    prepare_logger = loggers.prepare_logger
    if item not in config:
        if item not in default_values:
            log_info(prepare_logger, 'item {} not in config and default, using None'.format(item))
            return None
        default = default_values[item]
        log_info(prepare_logger, 'item {} not in config, using default {}'.format(item, default))
        return default
    return config[item]


def del_key(d, k):
    if k in d:
        del d[k]
