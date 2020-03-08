import numpy as np
import torch
import torch.nn.functional as F

from libs import process_re_data, get_model_output, get_tensor_batch, get_index, del_key


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


def _get_next_logits(model, data, generated, num_samples=1, temperature=1, top_k=0, top_p=0.0, repetition_penalty=1.0,
                     data_func=lambda x: process_re_data(x)):
    inputs = data_func(data)
    # print(inputs, inspect.getsource(data_func))
    if 'labels' in inputs:
        del inputs['labels']
    outputs = get_model_output(model, inputs)
    next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)
    # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
    for i in range(num_samples):
        for _ in set(generated[i].tolist()):
            next_token_logits[i, _] /= repetition_penalty
    filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
    if temperature == 0:  # greedy sampling:
        return filtered_logits
    return F.softmax(filtered_logits, dim=-1)


def get_seq_prob(model, data, data_func=lambda x: get_tensor_batch(x, batch_size=1, max_len=np.inf)):
    with torch.no_grad():
        model.eval()
        # print(data, data_func)
        inputs = data_func(data)
        # print(inputs)
        probs = {}
        del_key(inputs, 'labels')
        output = F.softmax(get_model_output(model, inputs)[0][0], dim=1)
        for e1, e2, sent, idx in zip(data['e1'], data['e2'], data['sent'], data['idx']):
            if sent.shape[0] == 0:
                continue
            e1l, e2l = e1.shape[0], e2.shape[0]
            pre_len = e1l + e2l
            index_e1 = get_index(sent, e1) + pre_len
            index_e2 = get_index(sent, e2) + pre_len

            sub_output_e1 = output[index_e1:index_e1 + e1l]
            sub_output_e2 = output[index_e2:index_e2 + e2l]
            prob_e1, prob_e2 = np.zeros(e1l), np.zeros(e2l)
            for e1i in range(e1l):
                prob_e1[e1i] = sub_output_e1[e1i][e1[e1i]].item()
            for e2i in range(e2l):
                prob_e2[e2i] = sub_output_e2[e2i][e2[e2i]].item()
            probs[tuple(idx)] = (prob_e1, prob_e2)
    return probs


def _sample_sequence(model, length, data, generated, num_samples=1, temperature=1, top_k=0, top_p=0.0,
                     repetition_penalty=1.0, data_func=lambda x: process_re_data(x)):
    model.eval()
    with torch.no_grad():
        for _ in range(length):

            next_logits = _get_next_logits(model, data, generated, num_samples=num_samples, temperature=temperature,
                                           top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
                                           data_func=data_func)
            if temperature == 0:  # greedy sampling:
                next_token = torch.argmax(next_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(next_logits, num_samples=1)

            generated = torch.cat((generated, next_token), dim=1)
            data['sent'] = generated
    return generated


def sample_sequence_entity(model, length, e1, e2, num_samples=1, temperature=1, top_k=0, top_p=0.0,
                           repetition_penalty=1.0, stop_func=lambda x: False):
    generated = torch.zeros(num_samples, 0).long().cuda()
    if len(e1) + len(e2) > length:
        return generated
    e1, e2 = torch.tensor(e1, dtype=torch.long).cuda(), torch.tensor(e2, dtype=torch.long).cuda()
    e1, e2 = e1.unsqueeze(0).repeat(num_samples, 1), e2.unsqueeze(0).repeat(num_samples, 1)
    data = {'e1': e1, 'e2': e2}
    generated = _sample_sequence(model, length, data, generated, num_samples=num_samples, temperature=temperature,
                                 top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
                                 data_func=process_re_data)
    return generated


def sample_sequence(model, length, context, num_samples=1, temperature=1, top_k=0, top_p=0.0, repetition_penalty=1.0):
    context = torch.tensor(context, dtype=torch.long).cuda()
    context = context.unsqueeze(0).repeat(num_samples, 1)
    data = {'sent': context}
    generated = _sample_sequence(model, length, data, context, num_samples=num_samples, temperature=temperature,
                                 top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
                                 data_func=lambda x: get_tensor_batch(x, batch_size=1, max_len=np.inf))
    return generated
