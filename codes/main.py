import argparse
import json
import logging
import os

import GPUtil
import torch
import transformers as tfm

from codes.dataset import BlockTextDataset
from codes.gpt2_eval import evaluate
from codes.gpt2_train import train

logging.getLogger('transformers.tokenization_utils').disabled = True

device = torch.device('cuda:0')


def log_info(logger, msg):
    print(msg)
    if logger is not None:
        logger.info(msg)


def cuda_mem_in_mb():
    return torch.cuda.memory_allocated() / 2 ** 20


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
    new_model, train_losses = train(model, data, batch_size, epochs, epoch_iter, learning_rate=learning_rate,
                                    weight_decay=weight_decay, loggers=(cuda_logger, train_logger, loss_logger),
                                    n_gpus=n_gpus)
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
