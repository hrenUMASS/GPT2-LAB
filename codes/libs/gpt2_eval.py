import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader

from . import loggers
from .loggers import log_info
from .util import eval_one_epoch


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


def sample_sequence(model, length, context, num_samples=1, temperature=1, top_k=0, top_p=0.0, repetition_penalty=1.0):
    context = torch.tensor(context, dtype=torch.long)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in tqdm.trange(length):

            inputs = {'input_ids': generated}

            outputs = model(**inputs)
            next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)

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


def evaluate(model, dataset, batch_size, epochs, epoch_iter, data_func=lambda x: x):
    validation_logger = loggers.validation_logger
    eval_loss, eval_steps = 0, 0
    losses, perplexities = [], []
    data_loader = DataLoader(dataset, shuffle=False, batch_size=batch_size, collate_fn=lambda x: x)
    model.eval()
    for e in range(epochs):
        loss, perp, eval_loss, eval_steps = eval_one_epoch(data_loader, model, eval_loss, eval_steps, data_func)
        losses.extend(loss)
        perplexities.extend(perp)
        loss_seg = losses[e * epoch_iter:]
        log_info(validation_logger, '----------------------------------------------------')
        log_info(validation_logger,
                 'Epoch {}, Mean Loss {}, Min Loss {}, Accum Loss {}'.format(e, np.mean(loss_seg), np.min(loss_seg),
                                                                             eval_loss / eval_steps))
    eval_loss /= eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))
    log_info(validation_logger, 'Final perplexity {}'.format(perplexity))
    return perplexity, torch.tensor(perplexities), torch.tensor(losses)
