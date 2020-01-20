from typing import Sequence

import numpy as np
import torch


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
    ignore_index = -1
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
        batch[i] = torch.cat((sent, torch.zeros(max_len - sent.shape[0], dtype=torch.long) + 50256), dim=0)
        labels[i] = torch.cat((sent, torch.ones(max_len - sent.shape[0], dtype=torch.long) * ignore_index), dim=0)
    return torch.stack(batch), labels, attn_mask


def cat_tensors(tensor_list, padding=0):
    result = list(tensor_list)
    max_len = max(result, key=lambda x: x.shape[-1]).shape[-1]
    dtype = result[0].dtype
    for i in range(len(result)):
        e = result[i]
        result[i] = torch.cat((e, torch.zeros(max_len - e.shape[-1], dtype=dtype) + padding))
    return torch.stack(result)


def get_re_data(data, max_len=np.inf):
    # print(data[0])
    empty = torch.zeros(0, dtype=torch.long)
    # sent_data = [x[2] for x in data]
    e1_data, e2_data = [], []
    sent_data = []
    for x in data:
        if len(data[0]) > 2:
            sent = x[2]
            # print('Testing entities in sentence')
            # print(x[0], x[1], x[2].shape)
            failed = False
            if not (in_tensor(sent, x[0]) and in_tensor(sent, x[1])):
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

    e1b, e1l, e1m = get_tensor_batch(e1_data)
    e2b, e2l, e2m = get_tensor_batch(e2_data)
    result = {'e1_ids': e1b, 'e1_mask': e1m, 'e1_labels': e1l, 'e2_ids': e2b, 'e2_mask': e2m, 'e2_labels': e2l}
    if len(sent_data) > 0:
        batch, labels, attn_mask = get_tensor_batch(sent_data)
        result.update({'input_ids': batch, 'attention_mask': attn_mask, 'labels': labels})
    return result


def get_model_output(model, data):
    # device = torch.device('cuda:0')
    for i in data:
        data[i].cuda()
    # print(data)
    try:
        output = model(**data)
        return output[0].mean()
    except Exception:
        print('data: ', data)
        print([x.device for x in data.values()])
        exit()
