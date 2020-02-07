import json
import math
import os

import GPUtil
import numpy as np
import torch
import transformers as tfm

from data_handler import DataIndexer
from libs import loggers
from libs.loggers import initial_loggers, log_info, cuda_mem_in_mb
from libs.util import get_config, get_module_from_parallel, get_params
from train_eval import train, evaluate


def start_func(config):
    from global_constants import mode_fields, model_templates, dataset_templates, \
        dataset_fields
    from global_constants import model_modes_func, data_process_func
    from global_constants import ModelEnums, IdxDataEnums, TrainModesEnums, ConfigEnums
    me, ide, tme, ce = ModelEnums, IdxDataEnums, TrainModesEnums, ConfigEnums
    config = {ce[k]: v for k, v in config.items()}
    mode = tme[get_config(config, ce.mode)]
    fields = mode_fields[mode]
    con = {k: get_config(config, k) for k in fields}
    model_type = me[con[ce.model]]
    load_path = get_config(con, ce.load_path)
    save_path = get_config(con, ce.save_path)

    if save_path is not None:
        if save_path[-1] != '/':
            save_path += '/'
        log_path = list(os.path.split(save_path)[:-1])
        log_path.append('log/')
        log_path = '/'.join(log_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        initial_loggers(log_path)

    prepare_logger, cuda_logger, final_logger = loggers.prepare_logger, loggers.cuda_logger, loggers.final_logger
    json_encoder = json.JSONEncoder(ensure_ascii=False, indent=2)
    log_info(prepare_logger, 'config loaded:\n' + json_encoder.encode(con))

    log_info(prepare_logger, 'loading models: ' + load_path)

    tok = tfm.GPT2Tokenizer.from_pretrained(load_path)
    log_info(prepare_logger, 'model loaded')
    log_info(cuda_logger, 'avaliable cudas {}'.format(torch.cuda.device_count()))
    log_info(prepare_logger, 'start training:\n\tepochs: {}\n\tepoch_iter: {}\n\tbatch_size: {}'.format(
        con[ce.epoch], con[ce.epoch_iter], con[ce.batch_size]))

    gpu = GPUtil.getGPUs()[0]
    log_info(cuda_logger, 'GPU Free {} Used {} Total {}'.format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryTotal))
    log_info(cuda_logger, 'Start cuda memory {}'.format(cuda_mem_in_mb()))
    log_info(cuda_logger, 'Allocated model {}'.format(cuda_mem_in_mb()))
    model = model_templates[model_type].from_pretrained(load_path)
    dataset_type = dataset_templates[ide[con[ce.dataset_type]]]
    dataset_parameters = {k.name: con[k] for k in dataset_fields['idx'][dataset_type]}
    data_indexer = DataIndexer(**dataset_parameters)
    con[ce.data_process_func] = data_process_func[mode][model_type]['idx'][dataset_type]
    con[ce.dataset_type] = dataset_type
    con[ce.tokenizer] = tok
    con[ce.model] = model
    con[ce.prev_eval_loss] = np.inf
    if ce.batch_len not in fields:
        con[ce.batch_len] = math.ceil((con[ce.epoch_iter] * con[ce.batch_size]) / 3200)

    method = model_modes_func[mode]
    eval_params = get_params(con, DataIndexer.get_eval)
    con[ce.evalset] = data_indexer.get_eval(**eval_params)
    eval_len = con[ce.eval_len]
    batch_len = con[ce.batch_len]
    temp = batch_len // (eval_len * 10)
    batch_lens = [eval_len * 10 for _ in range(temp)]
    batch_lens += [batch_len - temp]
    for i, bl in enumerate(batch_lens):
        new_con = dict(con)
        new_con[ce.dataset] = data_indexer.get_dataset(i, tokenizer=tok, dataset_type=dataset_type, batch_len=bl)
        new_con[ce.epoch_iter] = len(new_con[ce.dataset]) // new_con[ce.batch_size]
        new_model = method(new_con, i)
        con[ce.model] = new_model


def single_train(config, index):
    from global_constants import ConfigEnums, main_device
    ce = ConfigEnums
    save_path = config[ce.save_path]
    save_model = config[ce.save_model]

    config[ce.save_path] = config[ce.save_path] if config[ce.save_model] else None
    config[ce.model] = config[ce.model].to(main_device)

    final_logger = loggers.final_logger
    model_state = config[ce.model].state_dict()
    train_params = get_params(config, train)
    new_model, train_losses = train(**train_params)
    config[ce.dataset] = config[ce.evalset]
    eval_params = get_params(config, evaluate)
    perplexity, perplexities, eval_losses = evaluate(**eval_params)
    refuse = False
    if torch.mean(eval_losses) < config[ce.prev_eval_loss]:
        new_model.load_state_dict(model_state)
        refuse = True
        log_info(final_logger, 'loss {} is high, refused'.format(index))
    if save_path is not None:
        if save_model and not refuse:
            new_model = get_module_from_parallel(new_model)
            tokenizer = get_module_from_parallel(config[ce.tokenizer])
            log_info(final_logger, 'saving trained models: ' + save_path)
            new_model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
        log_path = list(os.path.split(save_path)[:-1])
        log_path.append('log/')
        log_path.append(str(index) + '/')
        log_path = '/'.join(log_path)
        log_info(final_logger, 'saving training losses')
        torch.save(train_losses, log_path + 'train_losses.pt')
        log_info(final_logger, 'saving evaluation losses')
        torch.save(eval_losses, log_path + 'eval_losses.pt')
        torch.save(perplexity, log_path + 'perplexity.pt')
        torch.save(perplexities, log_path + 'perplexities.pt')
        log_info(final_logger, 'mean eval losses {}'.format(torch.mean(eval_losses)))
        log_info(final_logger, 'All saved')
    return new_model


def single_sequence_generation(config):
    pass
    # data = {}
    # if ent_data is not None:
    #     data['ent'] = torch.load(ent_data)
    #
    # idx_file = open(idx_path, 'r')
    # ent_file = open(ent_path, 'r')
    # dataset = IdxEntityDataset(tokenizer, idx_file=idx_file, ent_file=ent_file, total_len=num_idx, data=data)
    # idx_file.close()
    # ent_file.close()
    # log_info(prepare_logger, 'Load idxes {} entities {}'.format(*dataset.get_loaded_length()))
    #
    # log_info(cuda_logger, "Allocated data {}".format(cuda_mem_in_mb()))
    # log_info(cuda_logger, 'GPU Free {} Used {} Total {}'.format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryTotal))
    #
    # log_info(prepare_logger, 'Selected GPT2LMHeadModel')
    # model = GPT2LMREModel.from_pretrained(load_path)
    # model = model.to(main_device)
    # data_func = lambda x: {'e1': x[0], 'e2': x[1], 'idx': x[2]}
    # ratios = eval_sequences(model, dataset, num_samples, max_len, data_func=data_func, tokenizer=tokenizer)
    #
    # if save_path is not None:
    #     log_info(final_logger, 'saving ratios')
    #     torch.save(ratios, log_path + 'ratios.pt')
    #     log_info(final_logger, 'All saved')
