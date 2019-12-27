import json
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from itertools import cycle
import torch.nn.functional as F
import tqdm
import transformers as tfm
import logging
import os
import argparse
import GPUtil
from datetime import datetime
import time

logging.getLogger('transformers.tokenization_utils').disabled = True

device = torch.device('cuda:0')


# class StreamData(IterableDataset):
#
#     def __init__(self, file_path, tokenizer: tfm.GPT2Tokenizer, process_func=lambda x: x):
#         self.file_path = file_path
#         self.process_func = process_func
#         self.tokenizer = tokenizer
#
#     def __iter__(self):
#         res = self.get_stream()
#         return res
#
#     def get_stream(self):
#         return cycle(self.parse_file())
#
#     def parse_file(self):
#         with open(self.file_path, 'r') as f:
#             # buff = ''
#             # for line in f:
#             #     if len(line.split()) > 0:
#             #         buff += line
#             #         try_process = self.process_func(buff)
#             #         if len(try_process) > 0:
#             #             yield try_process
#             #             buff = ''
#             for line in f:
#                 if len(line) > 2:
#                     yield line


class BlockTextDataset(Dataset):

    def __init__(self, file_path, tokenizer, total_len, block_size=512, valid_func=lambda x: True,
                 process_func=lambda x: x, max_len=None, truncate_mode='truncate'):
        self.valid_func = valid_func
        self.process_func = process_func
        self.block_size = block_size
        self.take_count = 1
        self.total_len = total_len
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.truncate_mode = truncate_mode
        self.file = open(file_path, 'r')
        self.data = self._read_block()

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        result = self.data[item % self.block_size]
        self.take_count += 1
        if self.take_count > self.block_size:
            self.take_count = 1
            self.data = self._read_block()
        return result

    def _read_block(self):
        result = []
        i = 0
        while i < self.block_size:
            line = self.process_func(self.file.readline())
            line = self.tokenizer.encode(line, return_tensors='pt')[0]
            if self.valid_func(line):
                if self.max_len is None or line.shape[0] <= self.max_len:
                    result.append(line)
                    # print('normal', result[-1].shape)
                    i += 1
                else:
                    if self.truncate_mode == 'truncate':
                        result.append(line[:self.max_len])
                        # print('truncate', result[-1].shape)
                        i += 1
                    elif self.truncate_mode == 'append':
                        k = 0
                        while k < line.shape[0]:
                            result.append(line[k:k + self.max_len])
                            # print('append', result[-1].shape)
                            k += self.max_len
                            i += 1
                    else:
                        raise ValueError('No such truncate mode {}'.format(self.truncate_mode))
        # print(list(map(lambda x: x.shape, result)))
        return result


def log_info(logger, msg):
    print(msg)
    if logger is not None:
        logger.info(msg)


def cuda_mem_in_mb():
    return torch.cuda.memory_allocated() / 2 ** 20


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


def get_tensor_batch(batch):
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
        labels[i] = torch.cat((sent, -torch.ones(max_len - sent.shape[0], dtype=torch.long)), dim=0)
    return torch.stack(batch), labels, attn_mask


def train(model, dataset, batch_size, epochs, epoch_iter, learning_rate=1e-2, weight_decay=1e-4,
          loggers=(None, None, None), n_gpus=1):
    no_decay = ['bias', 'LayerNorm.weight']
    device_ids = list(range(n_gpus))
    optimizer_params = [{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                         'weight_decay': weight_decay},
                        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                         'weight_decay': 0.0}]
    optimizer = tfm.AdamW(optimizer_params, lr=learning_rate)
    scheduler = tfm.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100,
                                                    num_training_steps=epochs * epoch_iter)
    losses = []
    # gpu = GPUtil.getGPUs()[0]
    data_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=lambda x: x)
    model = nn.DataParallel(model, device_ids)
    model.train()
    for e in range(epochs):
        epoch_start = time.time()
        for step, raw in enumerate(data_loader):
            batch, labels, attn_mask = get_tensor_batch(raw)
            batch.to(device)
            labels.to(device)
            attn_mask.to(device)
            log_info(loggers[0], 'Allocated batches {}, {}'.format(cuda_mem_in_mb(), batch.shape))
            # log_info(cuda_logger,
            #          'GPU Free {} Used {} Total {}'.format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryTotal))
            # optimizer.zero_grad()
            outputs = model(batch, labels=labels, attention_mask=attn_mask)
            loss = outputs[0].mean()

            log_info(loggers[1], '{} Iter Loss {}'.format(step, loss.item()))
            loss_value = loss.item()
            if len(losses) > 0 and abs(loss_value - losses[-1]) > 0.5:
                log_info(loggers[2], 'Huge Loss Change Detected {}\n{}'.format(loss_value - losses[-1], raw))
                # continue
            loss.backward()
            # if (step + 1) % 10 == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            del batch
            del labels
            del attn_mask
            del outputs
            del loss
            losses.append(loss_value)
            if step >= epoch_iter:
                break
        log_info(loggers[1], '----------------------------------------------------')
        # print(e * epoch_iter, (e + 1) * epoch_iter, losses[e])
        log_info(loggers[1],
                 'Epoch {}, Mean Loss {}, Min Loss {}'.format(e, np.mean(losses[e:e + epoch_iter]),
                                                              np.min(losses[e: e + epoch_iter])))
        time_diff = time.time() - epoch_start
        log_info(loggers[1],
                 'Time {}, Epoch Time {}, Avg Iter Time {}'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                                                                   time_diff, time_diff / epoch_iter))
    return model, torch.tensor(losses)


def evaluate(model, dataset, batch_size, epochs, epoch_iter, logger=None, n_gpus=1):
    eval_loss, eval_steps = 0, 0
    losses, perplexities = [], []
    data_loader = DataLoader(dataset, shuffle=False, batch_size=batch_size, collate_fn=lambda x: x)
    # model = nn.DataParallel(model)
    model.eval()
    # model.to(device)
    for e in range(epochs):
        step = 0
        for raw in data_loader:
            batch, labels, attn_mask = get_tensor_batch(raw)
            batch.to(device)
            labels.to(device)
            attn_mask.to(device)
            with torch.no_grad():
                outputs = model(batch, labels=labels, attention_mask=attn_mask)
                loss = outputs[0]
                loss_value = loss.mean().item()
                eval_loss += loss_value
                eval_steps += 1
                perplex_value = torch.exp(torch.tensor(eval_loss / eval_steps)).item()
                perplexities.append(perplex_value)
                log_info(logger,
                         'Loss {}, perplexity {}'.format(loss_value, perplex_value))
                losses.append(loss_value)
            step += 1
            if step >= epoch_iter:
                break
        log_info(logger, '----------------------------------------------------')
        log_info(logger,
                 'Epoch {}, Mean Loss {}, Min Loss {}, Accum Loss {}'.format(e, np.mean(losses[e:e + epoch_iter]),
                                                                             np.min(losses[e: e + epoch_iter]),
                                                                             eval_loss / eval_steps))
    eval_loss /= eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))
    return perplexity, torch.tensor(perplexities), torch.tensor(losses)


def main(config_file='model_config.json'):
    os.chdir('/'.join(os.path.abspath(__file__).split('/')[:-1]))
    with open(config_file, 'r') as f:
        config = json.load(f) if os.path.exists(config_file) and os.path.isfile(config_file) else {}
    # project_path = '/iesl/canvas/hren/gpt2_wiki_lab/v1'
    data_path = config.get('data_path',
                           '/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/data/raw/wiki2016_both.txt')
    load_path = config.get('load_path', 'gpt2-medium')
    save_path = config.get('save_path', None)

    def get_logger(name, log_file, level=logging.INFO):
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)
        return logger

    prepare_logger, train_logger, validation_logger = None, None, None
    final_logger, cuda_logger, loss_logger = None, None, None

    if save_path is not None:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        log_path = list(os.path.split(config['save_path']))
        log_path.append('log/')
        log_path = '/'.join(log_path)
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        prepare_logger = get_logger('gpt2 prepare logger', log_path + 'pre.log')
        train_logger = get_logger('gpt2 train logger', log_path + 'train.log')
        validation_logger = get_logger('gpt2 validation logger', log_path + 'val.log')
        final_logger = get_logger('gpt2 final logger', log_path + 'fin.log')
        cuda_logger = get_logger('cuda logger', log_path + 'cuda.log')
        loss_logger = get_logger('loss logger', log_path + 'los.log')

    json_encoder = json.JSONEncoder(ensure_ascii=False, indent=2)
    log_info(prepare_logger, 'config loaded:\n' + json_encoder.encode(config))

    log_info(prepare_logger, 'loading models: ' + load_path)

    model = tfm.GPT2LMHeadModel.from_pretrained(load_path)
    tokenizer = tfm.GPT2Tokenizer.from_pretrained(load_path)
    log_info(prepare_logger, 'model loaded')

    epochs = config.get('epochs', 20)
    epoch_iter = config.get('epoch_iter', 100)
    batch_size = config.get('batch_size', 16)
    learning_rate = float(config.get('learning_rate', 1e-2))
    weight_decay = float(config.get('weight_decay', 1e-4))
    n_gpus = config.get('n_gpus', 4)
    max_len = config.get('max_len', 512)
    truncate_mode = config.get('truncate_mode', 'truncate')

    data = BlockTextDataset(data_path, tokenizer, epochs * epoch_iter * batch_size, max_len=max_len,
                            truncate_mode=truncate_mode)
    log_info(cuda_logger, 'avaliable cudas {}'.format(torch.cuda.device_count()))
    log_info(prepare_logger, 'start training:\n\tepochs: {}\n\tepoch_iter: {}\n\tbatch_size: {}'.format(
        epochs, epoch_iter, batch_size))
    gpu = GPUtil.getGPUs()[0]
    log_info(cuda_logger, 'GPU Free {} Used {} Total {}'.format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryTotal))
    log_info(cuda_logger, 'Start cuda memory {}'.format(cuda_mem_in_mb()))
    model = model.to(device)
    log_info(cuda_logger, 'Allocatd model {}'.format(cuda_mem_in_mb()))
    new_model, train_losses = train(model, data, batch_size, epochs, epoch_iter,
                                    learning_rate=learning_rate, weight_decay=weight_decay,
                                    loggers=(cuda_logger, train_logger, loss_logger), n_gpus=n_gpus)
    perplexity, perplexities, eval_losses = evaluate(new_model, data, batch_size, epochs, epoch_iter,
                                                     logger=validation_logger, n_gpus=n_gpus)

    if save_path is not None:
        log_info(final_logger, 'saving trained models: ' + save_path)
        tfm.GPT2LMHeadModel.save_pretrained(new_model, save_path)
        tfm.GPT2Tokenizer.save_pretrained(tokenizer, save_path)
        log_info(final_logger, 'saving training losses')
        torch.save(train_losses, log_path + 'train_losses.pt')
        log_info(final_logger, 'saving evaluation losses')
        torch.save(eval_losses, log_path + 'eval_losses.pt')
        torch.save(torch.tensor(perplexity), log_path + 'perplexity.pt')
        torch.save(perplexities, log_path + 'perplexities.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    main(args.config)
