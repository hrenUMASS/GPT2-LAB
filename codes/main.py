import argparse
import json
import logging
import os

import GPUtil
import torch
import transformers as tfm

from libs import TextDataset, IdxDataset, GPT2LMREModel
from libs import evaluate, train
from libs import get_module_from_parallel, get_re_data, get_tensor_batch, process_re_data
from libs import log_info, initial_loggers, cuda_mem_in_mb
from libs import loggers

logging.getLogger('transformers.tokenization_utils').disabled = True

device = torch.device('cuda:0')


def single_train(config):
    # project_path = '/iesl/canvas/hren/gpt2_wiki_lab/v1'
    print(config)
    lab_data_path = '/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/data/raw/'
    data_path = config.get('data_path', lab_data_path + 'wiki2016_both.txt')
    load_path = config.get('load_path', 'gpt2-medium')
    save_path = config.get('save_path', None)
    save_model = config.get('save_model', False)

    if save_path is not None:
        log_path = list(os.path.split(save_path))
        save_path = '/'.join(log_path) + '/'
        log_path.append('log/')
        log_path = '/'.join(log_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        initial_loggers(log_path)

    prepare_logger, cuda_logger, final_logger = loggers.prepare_logger, loggers.cuda_logger, loggers.final_logger

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
    continue_train = config.get('continue', False)
    from_checkpoint = config.get('from_checkpoint', continue_train)
    eval_size = config.get('eval_size', min(batch_size * epoch_iter // 2, 1000))

    log_info(cuda_logger, 'avaliable cudas {}'.format(torch.cuda.device_count()))
    log_info(prepare_logger, 'start training:\n\tepochs: {}\n\tepoch_iter: {}\n\tbatch_size: {}'.format(
        epochs, epoch_iter, batch_size))
    gpu = GPUtil.getGPUs()[0]
    log_info(cuda_logger, 'GPU Free {} Used {} Total {}'.format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryTotal))
    log_info(cuda_logger, 'Start cuda memory {}'.format(cuda_mem_in_mb()))
    log_info(cuda_logger, 'Allocated model {}'.format(cuda_mem_in_mb()))

    if 'idx_path' in config:
        lab_ent_data = lab_data_path + 'wiki2016_nchunk_entity_agg/'
        ent_path = config.get('ent_path', lab_ent_data + 'wiki2016_ent')
        sent_path = config.get('sent_path', '/iesl/canvas/hren/gpt2_wiki_lab/data/wiki2016_sents_mapped')
        idx_path = config.get('idx_path', lab_ent_data + 'wiki2016_idx')
        idx_file = open(idx_path, 'r')
        ent_file = open(ent_path, 'r')
        sent_file = open(sent_path, 'r')
        dataset = IdxDataset(tokenizer, idx_file, ent_file, sent_file, epoch_iter * batch_size, eval_size=eval_size)
        idx_file.close()
        ent_file.close()
        sent_file.close()
        data_func = lambda x: process_re_data(get_re_data(x, max_len=max_len))
        ent_i, sent_i = dataset.get_total_ent_sent()
        log_info(prepare_logger, 'Load entities {}, sentences {}'.format(ent_i, sent_i))
        log_info(cuda_logger, "Allocated data {}".format(cuda_mem_in_mb()))
        log_info(cuda_logger, 'GPU Free {} Used {} Total {}'.format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryTotal))
    else:
        dataset = TextDataset(data_path, tokenizer, epoch_iter * batch_size, max_len=max_len,
                              valid_func=lambda x: x.shape[0] > 2, truncate_mode=truncate_mode, eval_size=eval_size)
        sent_i = len(dataset)
        log_info(prepare_logger, 'Load sentences {}'.format(sent_i))
        log_info(cuda_logger, "Allocated data {}".format(cuda_mem_in_mb()))
        log_info(cuda_logger, 'GPU Free {} Used {} Total {}'.format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryTotal))

        def data_func(x):
            batch, labels, attn_mask = get_tensor_batch(x)
            return {'input_ids': batch, 'labels': labels, 'attention_mask': attn_mask}

    if model_select == 1:
        if not isinstance(dataset, IdxDataset):
            log_info(prepare_logger, 'Not IdxDataset! Relation Model must use the IdxDataset!')
            exit()
        log_info(prepare_logger, 'Selected GPT2RE')
        model = GPT2LMREModel.from_pretrained(load_path)
    else:
        log_info(prepare_logger, 'Selected GPT2LMHeadModel')
        model = tfm.GPT2LMHeadModel.from_pretrained(load_path)
    model = model.to(device)
    new_model, train_losses = train(model, dataset, batch_size, epochs, epoch_iter, learning_rate=learning_rate,
                                    weight_decay=weight_decay, n_gpus=n_gpus,
                                    save_path=save_path if save_model else None,
                                    tokenizer=tokenizer, continue_train=continue_train,
                                    from_checkpoint=from_checkpoint, data_func=data_func)
    dataset.change_mode()
    perplexity, perplexities, eval_losses = evaluate(new_model, dataset, batch_size, epochs,
                                                     data_func=data_func)

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
        torch.save(perplexity, log_path + 'perplexity.pt')
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
