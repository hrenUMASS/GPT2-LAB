import inspect

import numpy as np
import torch
import torch.nn.functional as F

from libs import process_re_data, get_model_output, get_tensor_batch, get_index, del_key


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
    if temperature == 0:  # greedy sampling:
        return filtered_logits
    return F.softmax(filtered_logits, dim=-1), past


def get_seq_prob(model, data, data_func=lambda x: get_tensor_batch(x, batch_size=1, max_len=np.inf)):
    with torch.no_grad():
        model.eval()
        # print(data, data_func)
        inputs = data_func(data)
        # print(inputs)
        probs = []
        del_key(inputs, 'labels')

        layers = get_model_output(model, inputs)
        output = F.softmax(layers[0][0], dim=1)
        for e1, e2, sent, idx in zip(data['e1'], data['e2'], data['sent'], data['idx']):
            if sent.shape[0] == 0:
                continue
            e1l, e2l = e1.shape[0], e2.shape[0]
            pre_len = e1l + e2l
            # index_e1 = get_index(sent, e1) + pre_len
            index_e2 = get_index(sent, e2) + pre_len

            # sub_output_e1 = output[index_e1:index_e1 + e1l]
            sub_output_e2 = output[index_e2:index_e2 + e2l]
            prob_e1, prob_e2 = np.zeros(e1l), np.zeros(e2l)
            # print(1)
            # print(sub_output_e2.max(), sub_output_e2.min())
            # for e1i in range(e1l):
            #     prob_e1[e1i] = sub_output_e1[e1i][e1[e1i]].item()
            for e2i in range(e2l):
                # try:
                prob_e2[e2i] = sub_output_e2[e2i][e2[e2i]].item()
                # except:
                #     pass
            # loss = layers[0].mean().item()
            probs.append((prob_e1, prob_e2))
    return probs


def _sample_sequence(model, length, data, generated, num_samples=1, temperature=1, top_k=0, repetition_penalty=1.0,
                     data_func=lambda x: process_re_data(x)):
    model.eval()
    next_input = data_func(data)
    pos_id = 0
    if next_input is None:
        return None
    with torch.no_grad():
        for _ in range(length):

            next_logits, past = _get_next_logits(model, next_input, generated, num_samples=num_samples,
                                                 temperature=temperature, top_k=top_k,
                                                 repetition_penalty=repetition_penalty)
            if temperature == 0:  # greedy sampling:
                next_token = torch.argmax(next_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(next_logits, num_samples=1)
            next_input = {'input_ids': next_token,
                          'token_type_ids': torch.zeros(num_samples, 1, dtype=torch.long) + 2,
                          'position_ids': torch.zeros(num_samples, 1, dtype=torch.long) + pos_id,
                          'past': past}
            pos_id += 1
            generated = torch.cat((generated, next_token), dim=1)
    return generated


def sample_sequence_entity(model, length, e1, e2, num_samples=1, temperature=1, top_k=5, repetition_penalty=1.0):
    from global_constants import main_device
    device = main_device
    if isinstance(e1, torch.Tensor):
        device = e1.device
    generated = torch.zeros(num_samples, 0).long().to(device)
    if len(e1) + len(e2) > length:
        return generated
    e1, e2 = torch.tensor(e1, dtype=torch.long).to(device), torch.tensor(e2, dtype=torch.long).to(device)
    e1, e2 = e1.unsqueeze(0).repeat(num_samples, 1), e2.unsqueeze(0).repeat(num_samples, 1)
    data = {'e1': e1, 'e2': e2}
    generated = _sample_sequence(model, length, data, generated, num_samples=num_samples, temperature=temperature,
                                 top_k=top_k, repetition_penalty=repetition_penalty,
                                 data_func=process_re_data)
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
    generated = get_model_output(model, data)
    if hasattr(generated, 'logits'):
        # print(generated.logits.shape)
        # print(generated.logits.softmax(-1))
        # print(generated.logits.softmax(-1).max(-1))
        logit = generated.logits.softmax(-1)
    else:
        logit = generated[0].softmax(-1)
    maxes = logit.max(-1)
    result = list(zip(maxes.values.detach().cpu().tolist(), maxes.indices.detach().cpu().tolist()))
    # print(result)
    return result, logit.detach().cpu().numpy()
