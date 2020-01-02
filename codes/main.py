import argparse
import json
import logging
import os

import GPUtil
import torch
import transformers as tfm

from dataset import TextDataset
from gpt2_eval import evaluate
from gpt2_train import train
from loggers import get_logger, log_info, cuda_mem_in_mb

logging.getLogger('transformers.tokenization_utils').disabled = True

device = torch.device('cuda:0')


def single_train(config):
    # project_path = '/iesl/canvas/hren/gpt2_wiki_lab/v1'
    data_path = config.get('data_path',
                           '/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/data/raw/wiki2016_both.txt')
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

    data = TextDataset(data_path, tokenizer, epochs * epoch_iter * batch_size, max_len=max_len,
                       valid_func=lambda x: x.shape[0] > 2, truncate_mode=truncate_mode)
    log_info(cuda_logger, 'avaliable cudas {}'.format(torch.cuda.device_count()))
    log_info(prepare_logger, 'start training:\n\tepochs: {}\n\tepoch_iter: {}\n\tbatch_size: {}'.format(
        epochs, epoch_iter, batch_size))
    gpu = GPUtil.getGPUs()[0]
    log_info(cuda_logger, 'GPU Free {} Used {} Total {}'.format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryTotal))
    log_info(cuda_logger, 'Start cuda memory {}'.format(cuda_mem_in_mb()))
    model = model.to(device)
    log_info(cuda_logger, 'Allocatd model {}'.format(cuda_mem_in_mb()))
    new_model, train_losses = train(model, data, batch_size, epochs, epoch_iter, learning_rate=learning_rate,
                                    weight_decay=weight_decay, loggers=(cuda_logger, train_logger, loss_logger),
                                    n_gpus=n_gpus)
    perplexity, perplexities, eval_losses = evaluate(new_model, data, batch_size, epochs, epoch_iter,
                                                     logger=validation_logger, n_gpus=n_gpus)

    if save_path is not None and save_model:
        log_info(final_logger, 'saving trained models: ' + save_path)
        tfm.GPT2LMHeadModel.save_pretrained(new_model, save_path)
        tfm.GPT2Tokenizer.save_pretrained(tokenizer, save_path)
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
    config_fields = ['data_path', 'load_path', 'save_path', 'epochs', 'epoch_iter', 'batch_size', 'learning_rate',
                     'weight_decay', 'n_gpus', 'max_len', 'truncate_mode']
    models = None
    for field in config_fields:
        value = config.get(field, None)
        if value is not None and isinstance(value, list):
            if models is None:
                models = len(value)
            elif models != len(value):
                raise ValueError('Config field {} has wrong length'.format(field))
    for i in range(models):
        new_config = {}
        for field in config_fields:
            value = config.get(field, None)
            if value is not None:
                if isinstance(value, list):
                    new_config[field] = value[i]
                else:
                    new_config[field] = value
        single_train(new_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    main(args.config)
