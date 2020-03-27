import json
import os

# import GPUtil
import numpy as np
import torch
import transformers as tfm

from libs import get_config, get_module_from_parallel, get_params
from libs import initial_loggers, log_info, cuda_mem_in_mb
from libs import loggers
from train_eval import train, evaluate, eval_sequences, gpt2_eval


def start_func(config):
    from global_constants import data_process_func
    from global_constants import ModelEnums, DatasetEnums, TrainModesEnums, ConfigEnums, DataIndexerEnums
    me, de, tme, ce, di = ModelEnums, DatasetEnums, TrainModesEnums, ConfigEnums, DataIndexerEnums
    config = {ce[k]: v for k, v in config.items() if k in ce.__members__}
    # print(config)
    mode = tme[get_config(config, ce.mode)]
    fields = mode.value.fields
    con = {k: get_config(config, k) for k in fields}
    # print(con)
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
    log_info(prepare_logger, 'config loaded:\n' + json_encoder.encode({k.name: v for k, v in con.items()}))

    log_info(prepare_logger, 'loading models: ' + load_path)

    tok = tfm.GPT2Tokenizer.from_pretrained(load_path)
    log_info(prepare_logger, 'model loaded')
    log_info(cuda_logger, 'avaliable cudas {}'.format(torch.cuda.device_count()))
    # log_info(prepare_logger, 'start training:\n\tepochs: {}\n\tbatch_len: {}\n\tbatch_size: {}'.format(
    #     con[ce.epochs], con[ce.batch_len], con[ce.batch_size]))

    # gpu = GPUtil.getGPUs()[0]
    # log_info(cuda_logger, 'GPU Free {} Used {} Total {}'.format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryTotal))
    log_info(cuda_logger, 'Start cuda memory {}'.format(cuda_mem_in_mb()))
    log_info(cuda_logger, 'Allocated model {}'.format(cuda_mem_in_mb()))
    model = model_type.value.from_pretrained(load_path)

    dataset_type = de[con[ce.dataset_type]]
    indexer_type = di[con[ce.indexer_type]].value
    # print(indexer_type, type(indexer_type), repr(indexer_type))
    dataset_class = dataset_type.value.class_type
    con[ce.data_func] = data_process_func[mode][model_type] \
        [dataset_type](max_len=con[ce.max_len], batch_size=con[ce.batch_size] if ce.batch_size in con else 1)
    con[ce.dataset_type] = dataset_class
    con[ce.tokenizer] = tok
    con[ce.model] = model
    if ce.gpt2 in con:
        con[ce.gpt2] = tfm.GPT2LMHeadModel.from_pretrained(con[ce.gpt2])
    method = mode.value.func

    dataset_parameters = {k.name: con[k] for k in dataset_type.value.fields}

    data_indexer = indexer_type(**dataset_parameters)
    con[ce.data_indexer] = data_indexer
    if ce.eval_len in con:
        con[ce.prev_eval_loss] = np.inf
        # con[ce.eval_len] = max(con[ce.eval_len], 1)
        eval_params = get_params(con, indexer_type.get_eval)
        # print(eval_params)
        con[ce.evalset] = data_indexer.get_eval(**eval_params)

    for i in range(con[ce.loaders]):
        new_con = dict(con)
        new_con[ce.dataset] = data_indexer.get_dataset(i, tokenizer=tok, dataset_type=dataset_class,
                                                       batch_len=con[ce.batch_len])
        # ds = new_con[ce.dataset]
        new_con[ce.epoch_iter] = len(new_con[ce.dataset]) // (new_con[ce.batch_size] if ce.batch_size in new_con else 1)
        new_model, loss = method(new_con, i)
        con[ce.model] = new_model
        con[ce.prev_eval_loss] = loss


def single_train(config, index):
    from global_constants import ConfigEnums, main_device
    ce = ConfigEnums
    save_path = config[ce.save_path]
    save_model = config[ce.save_model]

    config[ce.save_path] = config[ce.save_path] if config[ce.save_model] else None
    config[ce.model] = config[ce.model].to(main_device)

    final_logger = loggers.final_logger
    model_state = config[ce.model].state_dict()
    # print(list(model_state.keys()))
    train_params = get_params(config, train)
    new_model, train_losses = train(**train_params)
    new_model = get_module_from_parallel(new_model)
    config[ce.dataset] = config[ce.evalset]
    eval_params = get_params(config, evaluate)
    perplexity, perplexities, eval_losses = evaluate(**eval_params)
    refuse = False
    loss = torch.mean(eval_losses)
    # print('index i', index)
    log_info(final_logger, 'final mean loss {}'.format(loss))
    # if loss > config[ce.prev_eval_loss]:
    #     new_model.load_state_dict(model_state)
    #     refuse = True
    #     log_info(final_logger, 'loss {} is high, refused'.format(index))
    #     loss = config[ce.prev_eval_loss]
    # else:
    #     config[ce.prev_eval_loss] = loss
    if save_path is not None:
        if save_model and not refuse:
            new_model = get_module_from_parallel(new_model)
            tokenizer = get_module_from_parallel(config[ce.tokenizer])
            log_info(final_logger, 'saving trained models: ' + save_path)
            new_model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
        log_path = list(os.path.split(save_path)[:-1])
        log_path.append('log')
        log_path.append(str(index) + '/')
        log_path = '/'.join(log_path)
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        log_info(final_logger, 'saving training losses')
        torch.save(train_losses, log_path + 'train_losses.pt')
        log_info(final_logger, 'saving evaluation losses')
        torch.save(eval_losses, log_path + 'eval_losses.pt')
        torch.save(perplexity, log_path + 'perplexity.pt')
        torch.save(perplexities, log_path + 'perplexities.pt')
        log_info(final_logger, 'mean eval losses {}'.format(torch.mean(eval_losses)))
        log_info(final_logger, 'All saved')
    return new_model, loss


def single_sequence_generation(config, index):
    from global_constants import ConfigEnums, main_device
    ce = ConfigEnums
    save_path = config[ce.save_path]
    config[ce.model] = config[ce.model].to(main_device)
    config[ce.gpt2] = config[ce.gpt2].to(main_device)
    final_logger = loggers.final_logger
    eval_params = get_params(config, eval_sequences)
    ratios = eval_sequences(**eval_params)
    if save_path is not None:
        log_path = list(os.path.split(save_path)[:-1])
        log_path.append('log')
        log_path.append(str(index) + '/')
        log_path = '/'.join(log_path)
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        log_info(final_logger, 'saving ratios')
        torch.save(ratios, log_path + 'ratios.pt')
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
        log_path = list(os.path.split(save_path)[:-1])
        log_path.append('log')
        log_path.append(str(index) + '/')
        log_path = '/'.join(log_path)
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        log_info(final_logger, 'saving ratios')
        torch.save(ratios, log_path + 'gpt2_ratios.pt')
        log_info(final_logger, 'All saved')
    return config[ce.model], -1
