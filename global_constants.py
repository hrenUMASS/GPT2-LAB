from enum import Enum, EnumMeta

import numpy as np
import torch
import transformers as tfm
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2LMHeadModel, GPT2ForSequenceClassification, GPT2Config
)

import data_handler as dh
import libs
import train_eval as te
from start_module import single_train, single_sequence_generation, gpt2_model_eval

ignore_index = -1
eos_id = 50256
main_device = torch.device('cuda:0')


class DatasetParam:

    def __init__(self, class_type):
        self.class_type = class_type


class ModeParam:
    def __init__(self, func):
        self.func = func


class ModelParam:
    def __init__(self, model, config):
        self.model = model
        self.config = config


class ConfigEnums:
    mode = 'mode'
    load_path = 'load_path'
    save_path = 'save_path'
    save_model = 'save_model'
    epochs = 'epochs'
    epoch_iter = 'epoch_iter'
    batch_size = 'batch_size'
    batch_len = 'batch_len'
    loaders = 'loaders'
    learning_rate = 'learning_rate'
    weight_decay = 'weight_decay'
    max_len = 'max_len'
    model = 'model'
    gpt2 = 'gpt2'
    dataset_type = 'dataset_type'
    ent_path = 'ent_path'
    sent_path = 'sent_path'
    idx_path = 'idx_path'
    idx_index_path = 'idx_index_path'
    num_samples = 'num_samples'
    eval_len = 'eval_len'
    tokenizer = 'tokenizer'
    data_func = 'data_func'
    continue_train = 'continue_train'
    from_checkpoint = 'from_checkpoint'
    dataset = 'dataset'
    evalset = 'evalset'
    prev_eval_loss = 'prev_eval_loss'
    data_indexer = 'data_indexer'
    sent_index_path = 'sent_index_path'
    ent_index_path = 'ent_index_path'
    ent_data = 'ent_data'
    data = 'data'
    db_path = 'db_path'
    ids = 'ids'
    indexer_type = 'indexer_type'
    batch_len_size = 'batch_len_size'
    save_type = 'save_type'
    config = 'config'
    save_epoch = 'save_epoch'
    save_checkpoint = 'save_checkpoint'
    all_config = 'all_config'
    method = 'method'
    between = 'between'


_ce = ConfigEnums


class ModelEnums(Enum):
    GPT2LMREModel = ModelParam(te.GPT2LMREModel, GPT2Config)
    GPT2LMHeadModel = ModelParam(GPT2LMHeadModel, GPT2Config)
    GPT2REClsModel = ModelParam(GPT2ForSequenceClassification, GPT2Config)


class DatasetEnums(Enum):
    idxDefault = DatasetParam(dh.IdxDataset)
    idxEnts = DatasetParam(dh.IdxEntityDataset)
    idxSents = DatasetParam(dh.IdxTextDataset)
    idxFull = DatasetParam(dh.IdxFullDataset)
    idxDBFull = DatasetParam(dh.IdxFullDBDataset)
    idxDBEnts = DatasetParam(dh.IdxEpDBDataset)
    idxDBCls = DatasetParam(dh.IdxClsDataset)


class DataIndexerEnums(Enum):
    idxDefaultIndexer = dh.DataIndexer
    idxDBIndexer = dh.DBIndexer


class TrainModesEnums(Enum):
    train_eval = ModeParam(single_train)
    eval_sequences = ModeParam(single_sequence_generation)
    gpt2_model_eval = ModeParam(gpt2_model_eval)


_me = ModelEnums
_de = DatasetEnums
_tme = TrainModesEnums

data_process_func = {
    _tme.train_eval: {
        _me.GPT2LMREModel: {
            _de.idxDefault: lambda max_len=np.inf, batch_size=32:
            (lambda x: libs.process_re_data(libs.get_re_data(x, max_len=max_len, batch_size=batch_size))),
            _de.idxSents: lambda max_len=np.inf, batch_size=32:
            (lambda x: libs.get_tensor_batch(x, max_len=max_len, batch_size=batch_size)),
            _de.idxFull: lambda max_len=np.inf, batch_size=32:
            (lambda x: libs.process_re_data(
                libs.get_re_data(x, max_len=max_len, batch_size=batch_size))),
            _de.idxDBFull: lambda max_len=np.inf, batch_size=32:
            (lambda x: libs.process_re_data(libs.get_re_data(x, max_len=max_len, batch_size=batch_size))),
        },
        _me.GPT2LMHeadModel: {
            _de.idxSents: lambda max_len=np.inf, batch_size=32:
            (lambda x: libs.get_tensor_batch(
                libs.get_column(x, 0), max_len=max_len, batch_size=batch_size)),
            _de.idxFull: lambda max_len=np.inf, batch_size=32:
            (lambda x: libs.get_tensor_batch(
                libs.get_re_data(x, max_len=max_len, batch_size=batch_size)['sent']))
        },
        _me.GPT2REClsModel: {
            _de.idxDBCls: lambda max_len=np.inf, batch_size=32: (
                lambda x: libs.process_cls_data(x)
            )
        }
    },
    _tme.eval_sequences: {
        _me.GPT2LMREModel: {
            _de.idxEnts: lambda max_len=np.inf, batch_size=1: (lambda x: {'e1': x[0], 'e2': x[1], 'idx': x[-1]}),
            _de.idxFull: lambda max_len=np.inf, batch_size=32: (
                lambda x: libs.get_re_data(x, max_len=max_len, batch_size=batch_size)
            ),
            _de.idxDBFull: lambda max_len=np.inf, batch_size=32: (
                lambda x: libs.get_re_data(x, max_len=max_len, batch_size=batch_size)
            ),
            _de.idxDBEnts: lambda max_len=np.inf, batch_size=32: (
                lambda x: libs.get_re_data(x, max_len, batch_size=batch_size)
            )
        },
        _me.GPT2REClsModel: {
            _de.idxDBFull: lambda max_len=np.inf, batch_size=32: (
                lambda x: libs.process_re_data(libs.get_re_data(
                    x, max_len=max_len, batch_size=batch_size
                ), between=True, inclusive=False, output_ent=False, st=0)
            )
        }
    },
    _tme.gpt2_model_eval: {
        _me.GPT2LMREModel: {
            _de.idxDBEnts: lambda max_len=np.inf, batch_size=32: (
                lambda x: libs.get_re_data(x, max_len=max_len, batch_size=batch_size)),
            _de.idxFull: lambda max_len=np.inf, batch_size=32: (
                lambda x: libs.get_re_data(x, max_len=max_len, batch_size=batch_size)),
            _de.idxDBFull: lambda max_len=np.inf, batch_size=32: (
                lambda x: libs.get_re_data(x, max_len=max_len, batch_size=batch_size))
        }
    }
}

main_methods = {
    _tme.train_eval: {
        _me.GPT2LMREModel: te.train,
        _me.GPT2REClsModel: te.train,
        _me.GPT2LMHeadModel: te.train
    },
    _tme.eval_sequences: {
        _me.GPT2LMREModel: te.eval_sequences,
        _me.GPT2REClsModel: te.classifier_eval,
        _me.GPT2LMHeadModel: te.eval_sequences
    },
    _tme.gpt2_model_eval: {
        _me.GPT2LMREModel: te.gpt2_eval,
        _me.GPT2REClsModel: te.gpt2_eval,
        _me.GPT2LMHeadModel: te.gpt2_eval
    }
}

default_values = {
    _ce.mode: 'train_eval',
    _ce.load_path: 'gpt2-medium',
    _ce.gpt2: 'gpt2',
    _ce.save_model: True,
    _ce.epochs: 1,
    # _ce.epoch_iter: 100,
    _ce.batch_size: 32,
    _ce.batch_len: 100,
    _ce.loaders: 10,
    _ce.learning_rate: 0.001,
    _ce.weight_decay: 0.0001,
    _ce.max_len: 512,
    _ce.model: 'GPT2LMHeadModel',
    _ce.dataset_type: 'IdxFullDataset',
    _ce.ent_path: '',
    _ce.sent_path: '',
    _ce.idx_path: '',
    _ce.idx_index_path: '',
    _ce.ent_index_path: '',
    _ce.sent_index_path: '',
    _ce.ent_data: '',
    _ce.num_samples: 20,
    _ce.eval_len: 10,
    _ce.continue_train: False,
    _ce.from_checkpoint: False,
    _ce.db_path: '',
    _ce.ids: None,
    _ce.indexer_type: 'idxDefaultIndexer',
    _ce.save_type: 'epochs',
    _ce.config: {},
    _ce.save_epoch: None,
    _ce.save_checkpoint: False,
    _ce.method: None,
    _ce.between: False
}
