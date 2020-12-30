import json
import os

# import GPUtil
import numpy as np
import torch
import transformers as tfm

from libs import get_config, get_module_from_parallel, get_params, parse_epoch_save
from libs import initial_loggers, log_info, cuda_mem_in_mb
from libs import loggers
from train_eval import train, evaluate, gpt2_eval


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
    from global_constants import data_process_func, main_methods, main_device
    from global_constants import ModelEnums, DatasetEnums, TrainModesEnums, ConfigEnums, DataIndexerEnums
    me, de, tme, ce, di = ModelEnums, DatasetEnums, TrainModesEnums, ConfigEnums, DataIndexerEnums
    config = {ce[k]: v for k, v in config.items() if k in ce.__members__}
    # print(config)
    mode = tme[get_config(config, ce.mode)]
    fields = mode.value.fields
    con = {k: get_config(config, k) for k in fields}
    # print(con)
    model_warp = me[con[ce.model]]
    model_type = model_warp.value.model
    model_config_type = model_warp.value.config
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
    model_config, model_kwargs = model_config_type.from_pretrained(
        load_path,
        return_unused_kwargs=True,
    )

    config_model_config = con[ce.config]

    for k, v in config_model_config.items():
        setattr(model_config, k, v)
    model = model_type.from_pretrained(load_path, config=model_config, **model_kwargs)

    if not (ce.random_init in con and con[ce.random_init]):
        log_info(prepare_logger, 'Model weight initialized')
        model.init_weights()

    dataset_type = de[con[ce.dataset_type]]
    indexer_type = di[con[ce.indexer_type]].value
    # print(indexer_type, type(indexer_type), repr(indexer_type))
    dataset_class = dataset_type.value.class_type
    con[ce.data_func] = data_process_func[mode][model_warp] \
        [dataset_type](max_len=con[ce.max_len], batch_size=con[ce.batch_size] if ce.batch_size in con else 1)
    con[ce.method] = main_methods[mode][model_warp]

    con[ce.dataset_type] = dataset_class
    con[ce.tokenizer] = tok
    con[ce.model] = model
    if ce.gpt2 in con:
        con[ce.gpt2] = tfm.GPT2LMHeadModel.from_pretrained(con[ce.gpt2])
    method = mode.value.func

    dataset_parameters = {k.name: con[k] for k in dataset_type.value.fields}

    model = model.to(main_device)
    print(model.device, main_device)

    data_indexer = indexer_type(**dataset_parameters)
    con[ce.data_indexer] = data_indexer
    save_type = con[ce.save_type]
    # print(con[ce.save_epoch], parse_epoch_save(con[ce.save_epoch]))
    con[ce.save_epoch] = parse_epoch_save(con[ce.save_epoch])
    if ce.eval_len in con:
        con[ce.prev_eval_loss] = np.inf
        # con[ce.eval_len] = max(con[ce.eval_len], 1)
        eval_params = get_params(con, indexer_type.get_eval)
        # print(eval_params)
        con[ce.evalset] = data_indexer.get_eval(**eval_params)
    if save_type == 'segments':
        for i in range(con[ce.loaders]):
            new_con = dict(con)
            new_con[ce.dataset] = data_indexer.get_dataset(i, tokenizer=tok, dataset_type=dataset_class,
                                                           batch_len=con[ce.batch_len])
            if new_con[ce.dataset] is None:
                break
            # ds = new_con[ce.dataset]
            new_con[ce.epoch_iter] = len(new_con[ce.dataset]) // (
                new_con[ce.batch_size] if ce.batch_size in new_con else 1)
            new_model, loss = method(new_con, i)
            con[ce.model] = new_model
            con[ce.prev_eval_loss] = loss
    elif save_type == 'epochs':
        epochs = con[ce.epochs]
        loaders = con[ce.loaders]
        con[ce.epochs] = 1
        for e in range(epochs):
            new_con = dict(con)
            for i in range(loaders):
                new_con[ce.save_model] = True
                new_con[ce.dataset] = data_indexer.get_dataset(i, tokenizer=tok, dataset_type=dataset_class,
                                                               batch_len=con[ce.batch_len])
                if new_con[ce.dataset_type] is None:
                    break
                new_con[ce.epoch_iter] = len(new_con[ce.dataset]) // (
                    new_con[ce.batch_size] if ce.batch_size in new_con else 1)
                # if i != loaders - 1:
                #     new_con[ce.save_model] = False
                new_model, loss = method(new_con, e)
                con[ce.model] = new_model
                con[ce.prev_eval_loss] = loss
    else:
        raise Exception('No such save type {}'.format(con[ce.save_type]))


def single_train(config, index):
    log_info(loggers.sample_logger, index)
    # for k, v in config.items():
    #     log_info(loggers.sample_logger, '{}:{}'.format(k, v))
    from global_constants import ConfigEnums, main_device
    ce = ConfigEnums
    save_path = config[ce.save_path]
    save_model = config[ce.save_model]

    config[ce.save_path] = config[ce.save_path] if config[ce.save_model] else None
    config[ce.model] = config[ce.model].to(main_device)

    final_logger = loggers.final_logger
    # model_state = config[ce.model].state_dict()
    # print(list(model_state.keys()))
    train_params = get_params(config, train)
    train_params['config'] = config
    new_model, train_losses = train(**train_params)
    new_model = get_module_from_parallel(new_model)
    if config[ce.evalset] is not None:
        config[ce.dataset] = config[ce.evalset]
        eval_params = get_params(config, evaluate)
        perplexity, perplexities, eval_losses = evaluate(**eval_params)
        loss = torch.mean(eval_losses)
        # print('index i', index)
        log_info(final_logger, 'final mean eval loss {}'.format(loss))
    # if loss > config[ce.prev_eval_loss]:
    #     new_model.load_state_dict(model_state)
    #     refuse = True
    #     log_info(final_logger, 'loss {} is high, refused'.format(index))
    #     loss = config[ce.prev_eval_loss]
    # else:
    #     config[ce.prev_eval_loss] = loss
    print(index + 1, config[ce.save_epoch], save_model)
    if save_path is not None and (config[ce.save_epoch] is None or index + 1 in config[ce.save_epoch]):
        if save_model:
            new_model = get_module_from_parallel(new_model)
            tokenizer = get_module_from_parallel(config[ce.tokenizer])
            log_info(final_logger, 'saving trained models: ' + save_path)
            if config[ce.save_type] == 'segments':
                new_model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
            elif config[ce.save_type] == 'epochs':
                save_path2 = save_path + '/' + str(index)
                if not os.path.isdir(save_path2):
                    os.mkdir(save_path2)
                new_model.save_pretrained(save_path2)
                tokenizer.save_pretrained(save_path2)

        log_path = list(os.path.split(save_path)[:-1])
        log_path.append('log')
        log_path.append(str(index) + '/')
        log_path = '/'.join(log_path)
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        log_info(final_logger, 'saving training losses')
        save_torch(train_losses, log_path + 'train_losses.pt')
        log_info(final_logger, 'saving evaluation losses')
        if config[ce.evalset] is not None:
            save_torch(eval_losses, log_path + 'eval_losses.pt')
            save_torch(perplexity, log_path + 'perplexity.pt')
            save_torch(perplexities, log_path + 'perplexities.pt')
            log_info(final_logger, 'mean eval losses {}'.format(torch.mean(eval_losses)))
        log_info(final_logger, 'All saved')
    if config[ce.evalset] is not None:
        return new_model, loss
    else:
        return new_model, torch.mean(train_losses)


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
        log_path = list(os.path.split(save_path)[:-1])
        log_path.append('log')
        log_path.append(str(index) + '/')
        log_path = '/'.join(log_path)
        if not os.path.exists(log_path):
            os.mkdir(log_path)
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
        log_path = list(os.path.split(save_path)[:-1])
        log_path.append('log')
        log_path.append(str(index) + '/')
        log_path = '/'.join(log_path)
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        log_info(final_logger, 'saving ratios')
        save_torch(ratios, log_path + 'gpt2_ratios.pt')
        log_info(final_logger, 'All saved')
    return config[ce.model], -1
