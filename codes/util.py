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


def get_re_data(raw, entities, max_len, sent_batch=None):
    # print(len(e1))
    sents, ent1s, ent2s = [], [], []

    def long_tensors(*x):
        return [torch.tensor(item, dtype=torch.long) for item in x]

    # print(raw)
    for i in range(len(raw)):
        x = raw[i]
        sent_id = x[2]
        # while sent_id != sent_i:
        # print(sent_id, sent_i)
        # stored_sent = sents.readline()
        # sent_i += 1
        # sent_tensor = tokenizer.encode(stored_sent, return_tensors='pt', add_prefix_space=True)[0]

        if sent_batch is not None:
            sent = long_tensors(sent_batch[sent_id])[0]

        ent1, ent2 = entities[x[0]], entities[x[1]],
        ent1, ent2 = long_tensors(ent1, ent2)
        if ent1.shape[0] > 0 and ent2.shape[0] > 0:
            if sent_batch is not None and sent.shape[0] > max_len:
                ent_index = get_index(sent, ent2) + ent2.shape[0]
                if ent_index > max_len:
                    continue
                else:
                    sent = sent[:max_len]
                sents.append(sent)
            ent1s.append(ent1)
            ent2s.append(ent2)
    # sent = [sents[x[2]] for x in raw]
    # print(e1, e2, sent)
    # print('empties', len([x for x in e1 if x.shape[0] == 0]))
    e1b, e1l, e1m = get_tensor_batch(ent1s)
    e2b, e2l, e2m = get_tensor_batch(ent2s)
    if sent_batch is not None:
        batch, labels, attn_mask = get_tensor_batch(sents)
        return {'e1_ids': e1b, 'e1_mask': e1m, 'e1_labels': e1l, 'e2_ids': e2b, 'e2_mask': e2m, 'e2_labels': e2l,
                'input_ids': batch, 'attention_mask': attn_mask, 'labels': labels}
    return {'e1_ids': e1b, 'e1_mask': e1m, 'e1_labels': e1l, 'e2_ids': e2b, 'e2_mask': e2m, 'e2_labels': e2l}


def get_model_output(model, data):
    for i in data:
        data[i] = data[i].cuda()
    # print(data)
    output = model(**data)
    return output[0].mean()
