import inspect
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from libs import process_re_data, get_model_output, get_tensor_batch, get_index, del_key
from .gpt2_modified import GPT2LMREModel


def top_k_top_p_filtering(logits, top_k=0, filter_value=-np.inf):
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    return logits


def _get_next_logits(model, inputs, generated, num_samples=1, temperature=1, top_k=0, repetition_penalty=1.0):
    # from global_constants import eos_id
    del_key(inputs, 'labels')
    del_key(inputs, 'attention_mask')
    outputs = get_model_output(model, inputs)
    past = outputs[1]
    next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)
    # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
    for i in range(num_samples):
        for _ in set(generated[i].tolist()):
            next_token_logits[i, _] /= repetition_penalty
    filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k)
    # if temperature == 0:  # greedy sampling:
    #     return filtered_logits
    return F.softmax(filtered_logits, dim=-1), past


def get_seq_prob(model, data, data_func=lambda x: get_tensor_batch(x, batch_size=1, max_len=np.inf),
                 mode="e1,e2,loss,sent,max"):
    with torch.no_grad():
        model.eval()
        # print(data, data_func)
        # print(data)
        inputs = data_func(data)
        # print(inputs)
        probs = []
        del_key(inputs, 'labels')

        mode = mode.split(',')

        e1m = "e1" in mode
        e2m = "e2" in mode
        lsm = "loss" in mode
        sem = "sent" in mode
        mam = "max" in mode

        layers = get_model_output(model, inputs)
        output = layers[0]
        # output = F.softmax(layers[0], dim=1)
        keys = ['e1', 'e2', 'sent', 'idx']
        for i, (e1, e2, sent, idx) in enumerate(zip(*(data[k] for k in keys))):
            if sent.shape[0] == 0:
                continue
            # print('output shape: {}, {}'.format(output[i].shape, output[i][:, inputs['input_ids'][i]].diagonal().shape))
            result = {}
            if e1m or e2m:
                e1l, e2l = e1.shape[0], e2.shape[0]
                pre_len = e1l + e2l
            if e1m:
                index_e1 = get_index(sent, e1) + pre_len
                prob_e1 = output[i][index_e1:index_e1 + e1l].cpu().numpy()
                result['prob_e1'] = prob_e1
            if e2m:
                index_e2 = get_index(sent, e2) + pre_len
                prob_e2 = output[i][index_e2:index_e2 + e2l].cpu().numpy()
                result['prob_e2'] = prob_e2
            if lsm:
                loss = layers[0].mean().item()
                result['loss'] = loss
            if sem:
                result['sent'] = output[i][:, inputs['input_ids'][i]].diagonal().cpu().numpy()
            if mam:
                result['max'] = output[i].argmax(-1).cpu().numpy()
            probs.append(result)
    return probs


def _sample_sequence(model, length, data, generated, num_samples=1, temperature=1, top_k=0, repetition_penalty=1.0,
                     data_func=lambda x: process_re_data(x), max_size=16, drop_head=0):
    from global_constants import eos_id
    model.eval()
    next_input = data_func(data)
    # print(next_input)
    pos_id = 0
    if next_input is None:
        return None
    # rstr = ''
    # for k, v in next_input.items():
    #     rstr += '{}: {}\n'.format(k, v)
    # print(rstr)
    model: GPT2LMREModel

    # with torch.no_grad():
    #     for _ in range(length):
    # 
    #         next_logits, past = _get_next_logits(model, next_input, generated, num_samples=num_samples,
    #                                              temperature=temperature, top_k=top_k,
    #                                              repetition_penalty=repetition_penalty)
    #         if temperature == 0:  # greedy sampling:
    #             next_token = torch.argmax(next_logits, dim=-1).unsqueeze(-1)
    #         else:
    #             next_token = torch.multinomial(next_logits, num_samples=1)
    #         next_input = {'input_ids': next_token,
    #                       'token_type_ids': torch.zeros(num_samples, 1, dtype=torch.long),
    #                       'position_ids': torch.zeros(num_samples, 1, dtype=torch.long) + pos_id,
    #                       'past': past}
    #         # rstr = ''
    #         # for k in ['input_ids', 'token_type_ids', 'position_ids']:
    #         #     rstr += '{}: {}\n'.format(k, next_input[k].view(1, -1))
    #         # print(rstr)
    #         pos_id += 1
    #         generated = torch.cat((generated, next_token), dim=1)

    for k, v in next_input.items():
        next_input[k] = v.cpu()[0]
    # print(next_input)
    if max_size < num_samples:
        generated = []
        segments = num_samples // max_size
        if segments * max_size != num_samples:
            segments += 1
        for i in range(segments):
            if (i + 1) * max_size > num_samples:
                next_size = num_samples - i * max_size
            else:
                next_size = max_size
            # print(next_size, length, next_input['input_ids'])
            generated.append(model.generate(
                input_ids=next_input['input_ids'][None, :],
                max_length=length,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                num_return_sequences=next_size,
                do_sample=True,
                token_type_ids=next_input['token_type_ids'][None, :],
                position_ids=next_input['position_ids'][None, :],
                first=None,
                eos_token_id=eos_id,
                pad_token_id=eos_id,
                use_cache=True
            ).cpu()[:, drop_head:])
            # print(generated[-1].shape)
        # generated = cat_tensors(generated, padding=eos_id)
        generated = torch.cat(generated)
    else:
        generated = model.generate(
            input_ids=next_input['input_ids'],
            max_length=length,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_samples,
            do_sample=True,
            token_type_ids=next_input['token_type_ids'],
            position_ids=next_input['position_ids'],
            first=None,
            eos_token_id=eos_id,
            pad_token_id=eos_id
        ).cpu()[:, drop_head:]

    return generated


def sample_sequence_entity(model, length, e1, e2, num_samples=1, max_size=16, temperature=1, top_k=5,
                           repetition_penalty=1.0):
    from global_constants import main_device
    device = main_device
    if isinstance(e1, torch.Tensor):
        device = e1.device
    generated = torch.zeros(num_samples, 0).long().to(device)
    if len(e1) + len(e2) > length:
        return generated
    data = {'e1': e1, 'e2': e2}
    if not isinstance(e1, Sequence):
        data['e1'] = [e1]
        data['e2'] = [e2]
    # print(data)
    generated = _sample_sequence(model, length, data, generated, num_samples=num_samples, temperature=temperature,
                                 top_k=top_k, repetition_penalty=repetition_penalty, data_func=process_re_data,
                                 max_size=max_size, drop_head=len(e1) + len(e2))
    return generated


def sample_sequence(model, length, data, num_samples=1, temperature=1, top_k=0, repetition_penalty=1.0):
    from global_constants import main_device
    device = main_device
    if isinstance(data['e1'], torch.Tensor):
        device = data['e1'].device
    context = torch.zeros((num_samples, 0), dtype=torch.long).to(device)
    params = inspect.signature(model.__call__)
    for k in data:
        if k not in params:
            del_key(data, k)
    generated = _sample_sequence(model, length, data, context, num_samples=num_samples, temperature=temperature,
                                 top_k=top_k, repetition_penalty=repetition_penalty,
                                 data_func=lambda x: get_tensor_batch(x, batch_size=1, max_len=np.inf))
    return generated


def sample_classifier_sequence(model, data):
    # print(data.shape)

    generated = get_model_output(model, data)
    # print(generated[0])
    generated = generated[0].softmax(-1)
    result = []
    for i in range(generated.shape[0]):
        # class possibility
        item = generated[i]
        result.append((item.argmax().item(), item.max().item()))
    return result
