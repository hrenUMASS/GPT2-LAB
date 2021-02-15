import json
import os
from pathlib import Path

# import GPUtil
import numpy as np
import torch
import transformers as tfm
from transformers.modeling_utils import PreTrainedModel

from libs import get_config, get_module_from_parallel, get_params, parse_epoch_save
from libs import initial_loggers, log_info, cuda_mem_in_mb
from libs import loggers
from train_eval import train, evaluate, gpt2_eval


class Config(dict):

    def __init__(self, config, default_config):
        super().__init__(self)
        self.update(config)
        self.default = default_config
        self.logger = loggers.prepare_logger

    def __getitem__(self, item):
        if item in self:
            return dict.__getitem__(self, item)
        elif item in self.default:
            default = self.default[item]
            log_info(self.logger, 'item {} not in config, using default {}'.format(item, default))
            return default
        log_info(self.logger, 'item {} not in config and default, using None'.format(item))
        return None


def save_torch(obj, file_path):
    file_path, file_name = os.path.split(file_path)
    file_name, file_ext = os.path.splitext(file_name)
    i = 1
    if os.path.isfile(file_path + '/' + file_name + file_ext):
        while os.path.isfile(file_path + '/' + file_name + str(i) + file_ext):
            i += 1
    print(file_path + '/' + file_name + str(i) + file_ext)
    torch.save(obj, file_path + '/' + file_name + str(i) + file_ext)


def start_func(config):
    from global_constants import data_process_func, default_values, ModelParam, main_methods
    from global_constants import ModelEnums, DatasetEnums, TrainModesEnums, ConfigEnums, DataIndexerEnums
    me, de, tme, ce, di = ModelEnums, DatasetEnums, TrainModesEnums, ConfigEnums, DataIndexerEnums
    config = Config(config, default_config=default_values)
    mode = tme[config[ce.mode]]
    model_warp_kv = me[config[ce.model]]
    model_warp: ModelParam = model_warp_kv.value
    model_type = model_warp.model
    model_config_type = model_warp.config

    load_path = config[ce.load_path]
    save_path = config[ce.save_path]

    if save_path is not None:
        if save_path[-1] != '/':
            save_path += '/'
        log_path = list(os.path.split(save_path)[:-1])
        log_path.append('log/')
        log_path = '/'.join(log_path)
        if not os.path.exists(save_path):
            # os.mkdir(save_path)
            Path(save_path).mkdir(parents=True, exist_ok=True)
        initial_loggers(log_path)

    prepare_logger, cuda_logger, final_logger = loggers.prepare_logger, loggers.cuda_logger, loggers.final_logger
    json_encoder = json.JSONEncoder(ensure_ascii=False, indent=2)
    log_info(prepare_logger, 'config loaded:\n' + json_encoder.encode(config))

    log_info(prepare_logger, 'loading models: ' + load_path)

    tok = tfm.GPT2Tokenizer.from_pretrained(load_path)
    log_info(prepare_logger, 'model loaded')
    log_info(cuda_logger, 'avaliable cudas {}'.format(torch.cuda.device_count()))
    log_info(cuda_logger, 'Start cuda memory {}'.format(cuda_mem_in_mb()))
    log_info(cuda_logger, 'Allocated model {}'.format(cuda_mem_in_mb()))

    model_config = model_config_type.from_pretrained(load_path)

    config_model_config = config[ce.config]

    for k, v in config_model_config.items():
        setattr(model_config, k, v)
    # print(model_config)
    model = model_type.from_pretrained(load_path, config=model_config)

    dataset_type = de[config[ce.dataset_type]]
    indexer_type = di[config[ce.indexer_type]].value
    # print(indexer_type, type(indexer_type), repr(indexer_type))
    dataset_class = dataset_type.value.class_type
    config[ce.data_func] = data_process_func[mode][model_warp_kv] \
        [dataset_type](max_len=config[ce.max_len], batch_size=config[ce.batch_size] if ce.batch_size in config else 1)
    config[ce.dataset_type] = dataset_class
    config[ce.tokenizer] = tok
    config[ce.model] = model
    if ce.gpt2 in config:
        config[ce.gpt2] = tfm.GPT2LMHeadModel.from_pretrained(config[ce.gpt2])

    method = mode.value.func

    config[ce.method] = main_methods[mode][model_warp_kv]

    # dataset_parameters = {k.name: config[k] for k in dataset_type.value.fields}
    dataset_parameters = get_params(config, indexer_type)

    data_indexer = indexer_type(**dataset_parameters)
    config[ce.data_indexer] = data_indexer
    save_type = config[ce.save_type]
    config[ce.all_config] = config
    config[ce.save_epoch] = parse_epoch_save(config[ce.save_epoch])
    print(config[ce.save_epoch])
    if ce.eval_len in config:
        config[ce.prev_eval_loss] = np.inf
        # con[ce.eval_len] = max(con[ce.eval_len], 1)
        eval_params = get_params(config, indexer_type.get_eval)
        config[ce.evalset] = data_indexer.get_eval(**eval_params)
    if save_type == 'segments':
        for i in range(config[ce.loaders]):
            new_con = dict(config)
            new_con[ce.dataset] = data_indexer.get_dataset(i, tokenizer=tok, dataset_type=dataset_class,
                                                           batch_len=config[ce.batch_len])
            if new_con[ce.dataset] is None:
                break
            # ds = new_con[ce.dataset]
            new_con[ce.epoch_iter] = len(new_con[ce.dataset]) // (
                new_con[ce.batch_size] if ce.batch_size in new_con else 1)
            new_model, loss = method(new_con, i)
            config[ce.model] = new_model
            config[ce.prev_eval_loss] = loss
    elif save_type == 'epochs':
        epochs = config[ce.epochs]
        loaders = config[ce.loaders]
        config[ce.epochs] = 1
        for e in range(epochs):
            new_con = dict(config)
            for i in range(loaders):
                new_con[ce.save_model] = True
                new_con[ce.dataset] = data_indexer.get_dataset(i, tokenizer=tok, dataset_type=dataset_class,
                                                               batch_len=config[ce.batch_len])
                if new_con[ce.dataset] is not None:
                    print(len(new_con[ce.dataset].data))
                if new_con[ce.dataset_type] is None:
                    break
                new_con[ce.epoch_iter] = len(new_con[ce.dataset]) // (
                    new_con[ce.batch_size] if ce.batch_size in new_con else 1)
                # if i != epochs - 1:
                #     new_con[ce.save_model] = False
                new_model, loss = method(new_con, e)
                config[ce.model] = new_model
                config[ce.prev_eval_loss] = loss
    else:
        raise Exception('No such save type {}'.format(config[ce.save_type]))


def single_train(config, index):
    print(index)
    from global_constants import ConfigEnums, main_device
    ce = ConfigEnums
    save_path = config[ce.save_path]
    save_model = config[ce.save_model]

    config[ce.save_path] = config[ce.save_path] if config[ce.save_model] else None
    config[ce.model] = config[ce.model].to(main_device)

    final_logger = loggers.final_logger
    model_state = config[ce.model].state_dict()
    # print(list(model_state.keys()))
    method = config[ce.method]
    train_params = get_params(config, method)
    new_model, train_losses, logits = method(**train_params)
    loss = train_losses.mean()
    new_model = get_module_from_parallel(new_model)
    if config[ce.evalset] is not None:
        config[ce.dataset] = config[ce.evalset]
        eval_params = get_params(config, evaluate)
        perplexity, perplexities, eval_losses = evaluate(**eval_params)
        loss = eval_losses.mean()
        # print('index i', index)
        log_info(final_logger, 'final mean loss {}'.format(loss))
    if (save_path is not None and
            (config[ce.save_epoch] is None or
             index + 1 in config[ce.save_epoch] or
             index + 1 == config[ce.epochs])):
        if save_model:
            new_model = get_module_from_parallel(new_model)
            tokenizer = get_module_from_parallel(config[ce.tokenizer])
            log_info(final_logger, 'saving trained models: ' + save_path)
            if config[ce.save_type] == 'segments':
                log_info(final_logger, 'saving trained models: ' + save_path)
                new_model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
            elif config[ce.save_type] == 'epochs':
                save_path2 = save_path + '/' + str(index)
                if not os.path.isdir(save_path2):
                    # os.mkdir(save_path2)
                    Path(save_path2).mkdir(parents=True, exist_ok=True)
                new_model.save_pretrained(save_path2)
                tokenizer.save_pretrained(save_path2)

    # save log
    if save_path is not None:
        log_path = list(os.path.split(save_path))
        log_path.append('log')
        log_path.append(str(index) + '/')
        log_path = '/'.join(log_path)
        if not os.path.exists(log_path):
            Path(log_path).mkdir(parents=True, exist_ok=True)
            # os.mkdir(log_path)
        log_info(final_logger, 'saving training losses')
        save_torch(train_losses, log_path + 'train_losses.pt')
        print(len(logits))
        if len(logits) > 0:
            log_info(loggers.final_logger, 'saving logits {}'.format(len(logits)))
            save_torch(logits, log_path + 'logits.pt')
        log_info(final_logger, 'saving evaluation losses')
        if config[ce.evalset] is not None:
            save_torch(eval_losses, log_path + 'eval_losses.pt')
            save_torch(perplexity, log_path + 'perplexity.pt')
            save_torch(perplexities, log_path + 'perplexities.pt')
            log_info(final_logger, 'mean eval losses {}'.format(eval_losses.mean()))
        log_info(final_logger, 'All saved')
    return new_model, loss


def single_sequence_generation(config, index):
    print(index)
    from global_constants import ConfigEnums, main_device
    ce = ConfigEnums
    save_path = config[ce.save_path]
    config[ce.model] = config[ce.model].to(main_device)
    config[ce.gpt2] = config[ce.gpt2].to(main_device)
    final_logger = loggers.final_logger
    method = config[ce.method]
    eval_params = get_params(config, method)
    ratios = method(**eval_params)
    if save_path is not None:
        log_path = list(os.path.split(save_path))
        log_path.append('log')
        log_path.append(str(index) + '/')
        log_path = '/'.join(log_path)
        if not os.path.exists(log_path):
            # os.mkdir(log_path)\
            Path(log_path).mkdir(parents=True, exist_ok=True)
        log_info(final_logger, 'saving ratios')
        save_torch(ratios, log_path + 'ratios.pt')
        log_info(final_logger, 'All saved')
    return config[ce.model], -1


def gpt2_model_eval(config, index):
    from global_constants import ConfigEnums, main_device
    ce = ConfigEnums
    save_path = config[ce.save_path]
    config[ce.model] = config[ce.model].to(main_device)
    config[ce.gpt2] = config[ce.gpt2].to(main_device)
    final_logger = loggers.final_logger
    eval_params = get_params(config, gpt2_eval)
    ratios = gpt2_eval(**eval_params)
    if save_path is not None:
        log_path = list(os.path.split(save_path))
        log_path.append('log')
        log_path.append(str(index) + '/')
        log_path = '/'.join(log_path)
        if not os.path.exists(log_path):
            # os.mkdir(log_path)
            Path(log_path).mkdir(parents=True, exist_ok=True)
        log_info(final_logger, 'saving ratios')
        save_torch(ratios, log_path + 'gpt2_ratios.pt')
        log_info(final_logger, 'All saved')
    return config[ce.model], -1
