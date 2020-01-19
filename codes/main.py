import argparse
import json
import logging
import os

import GPUtil
import torch
import transformers as tfm
from torch import nn

from dataset import TextDataset, IdxDataset
from gpt2_eval import evaluate_re, evaluate_normal
from gpt2_modified import GPT2LMREModel
from gpt2_train import train_normal, train_re
from loggers import get_logger, log_info, cuda_mem_in_mb

logging.getLogger('transformers.tokenization_utils').disabled = True

device = torch.device('cuda:0')


def get_module_from_parallel(module):
    if isinstance(module, nn.DataParallel):
        return get_module_from_parallel(module.module)
    return module


def single_train(config):
    # project_path = '/iesl/canvas/hren/gpt2_wiki_lab/v1'
    print(config)
    lab_data_path = '/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/data/raw/'
    data_path = config.get('data_path', lab_data_path + 'wiki2016_both.txt')
    load_path = config.get('load_path', 'gpt2-medium')
    save_path = config.get('save_path', None)
    save_model = config.get('save_model', False)

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
    model_select = config.get('model_select', 0)

    log_info(cuda_logger, 'avaliable cudas {}'.format(torch.cuda.device_count()))
    log_info(prepare_logger, 'start training:\n\tepochs: {}\n\tepoch_iter: {}\n\tbatch_size: {}'.format(
        epochs, epoch_iter, batch_size))
    gpu = GPUtil.getGPUs()[0]
    log_info(cuda_logger, 'GPU Free {} Used {} Total {}'.format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryTotal))
    log_info(cuda_logger, 'Start cuda memory {}'.format(cuda_mem_in_mb()))
    log_info(cuda_logger, 'Allocated model {}'.format(cuda_mem_in_mb()))
    if model_select == 1:
        log_info(prepare_logger, 'Selected GPT2RE')
        model = GPT2LMREModel.from_pretrained(load_path)
        model.to(device)
        lab_ent_data = lab_data_path + 'wiki2016_nchunk_entity_agg/'
        ent_path = config.get('ent_path', lab_ent_data + 'wiki2016_ent')
        sent_path = config.get('sent_path', '/iesl/canvas/hren/gpt2_wiki_lab/data/wiki2016_sents_mapped')
        idx_path = config.get('idx_path', lab_ent_data + 'wiki2016_idx')
        idx_file = open(idx_path, 'r')
        ent_file = open(ent_path, 'r')
        sent_file = open(sent_path, 'r')
        idx_data = IdxDataset(idx_file, epoch_iter * batch_size)

        def get_max_ent_sent(idx_data):
            max_ent, max_sent = 0, 0
            for x in idx_data.data:
                if x[0] > max_ent:
                    max_ent = x[0]
                if x[1] > max_ent:
                    max_ent = x[1]
                if x[2] > max_sent:
                    max_sent = x[2]
            return max_ent, max_sent

        max_ent_i, max_sent_i = get_max_ent_sent(idx_data)
        log_info(prepare_logger, 'preparing entities and sentences')
        ent_data = TextDataset(ent_file, tokenizer, max_ent_i)
        sent_data = TextDataset(sent_file, tokenizer, max_sent_i, max_len=max_len)

        # with open(ent_path, 'r') as f:
        #     for _ in range(max_ent_i + 1):
        #         l = f.readline()
        #         enc = tokenizer.encode(l[:-1], return_tensors='pt', add_prefix_space=True)[0]
        #         ent_data.append(enc)
        log_info(prepare_logger, 'entities and sentences loaded, with length {}, {}'.format(max_ent_i, max_sent_i))
        # idx_filter = np.array(idx_data.data)
        # sent_max_len = idx_data[-1][-1]
        # sent_data = []
        # log_info(prepare_logger, 'preparing sentences')

        # log_info(prepare_logger, 'sentences loaded')
        log_info(cuda_logger, "Allocated data {}".format(cuda_mem_in_mb()))
        new_model, train_losses = train_re(model, tokenizer, ent_data, idx_data, sent_data, batch_size, epochs,
                                           epoch_iter, learning_rate=learning_rate, weight_decay=weight_decay,
                                           loggers=(cuda_logger, train_logger, loss_logger), n_gpus=n_gpus,
                                           max_len=max_len)
        idx_data = IdxDataset(idx_file, epoch_iter * batch_size)

        max_ent_2, max_sent_2 = get_max_ent_sent(idx_data)
        ent_data = TextDataset(ent_file, tokenizer, max_ent_2 - max_ent_i)
        sent_data = TextDataset(sent_file, tokenizer, max_sent_2 - max_sent_i, max_len=max_len)

        perplexity, perplexities, eval_losses = evaluate_re(model, tokenizer, ent_data, idx_data, sent_data, batch_size,
                                                            epochs, epoch_iter, logger=validation_logger, n_gpus=n_gpus)
        idx_file.close()
        ent_file.close()
        sent_file.close()
    else:
        model = tfm.GPT2LMHeadModel.from_pretrained(load_path)
        model = model.to(device)
        data = TextDataset(data_path, tokenizer, epoch_iter * batch_size, max_len=max_len,
                           valid_func=lambda x: x.shape[0] > 2, truncate_mode=truncate_mode)
        new_model, train_losses = train_normal(model, data, batch_size, epochs, epoch_iter, learning_rate=learning_rate,
                                               weight_decay=weight_decay,
                                               loggers=(cuda_logger, train_logger, loss_logger), n_gpus=n_gpus)
        perplexity, perplexities, eval_losses = evaluate_normal(new_model, data, batch_size, epochs, epoch_iter,
                                                                logger=validation_logger, n_gpus=n_gpus)

    if save_path is not None:
        if save_model:
            new_model = get_module_from_parallel(new_model)
            tokenizer = get_module_from_parallel(tokenizer)
            log_info(final_logger, 'saving trained models: ' + save_path)
            new_model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
        log_info(final_logger, 'saving training losses')
        torch.save(train_losses, log_path + 'train_losses.pt')
        log_info(final_logger, 'saving evaluation losses')
        torch.save(eval_losses, log_path + 'eval_losses.pt')
        torch.save(torch.tensor(perplexity), log_path + 'perplexity.pt')
        torch.save(perplexities, log_path + 'perplexities.pt')


def main(config_file='model_config.json'):
    os.chdir('/'.join(os.path.abspath(__file__).split('/')[:-1]))
    with open(config_file, 'r') as f:
        config = json.load(f) if os.path.exists(config_file) and os.path.isfile(config_file) else {}
    models = None
    for k, v in config.items():
        if isinstance(v, list):
            if models is None:
                models = len(v)
            elif models != len(v):
                raise ValueError('Config field {} has wrong length'.format(k))
    models = models if models is not None else 1
    for i in range(models):
        new_config = {}
        for k, v in config.items():
            if isinstance(v, list):
                new_config[k] = v[i]
            else:
                new_config[k] = v
        single_train(new_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    main(args.config)
