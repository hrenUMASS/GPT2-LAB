import numpy as np
import torch
from torch.utils.data import DataLoader

from . import loggers
from .loggers import log_info
from .util import eval_one_epoch, sampling_one_epoch


def eval_sequences(model, dataset, num_samples, max_length, data_func=lambda x: x, tokenizer=None):
    sample_logger = loggers.sample_logger
    data_loader = DataLoader(dataset, shuffle=False, batch_size=1, collate_fn=lambda x: x)
    ratios = sampling_one_epoch(data_loader, model, max_length, num_samples, data_func, tokenizer=tokenizer)
    log_info(sample_logger, 'Total ratio {}'.format(np.mean((x[-1] for x in ratios))))
    return ratios


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
