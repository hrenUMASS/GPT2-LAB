from enum import Enum

import numpy as np
import torch
import transformers as tfm

import data_handler as dh
import libs
import train_eval as te
from start_module import single_train, single_sequence_generation

lab_data_path = '/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/data/raw/wiki2016_nchunk_entity_agg/'
self_data_path = '/iesl/canvas/hren/gpt2_wiki_lab/data/'
ignore_index = -1
eos_id = 50256
main_device = torch.device('cuda:0')


class AutoEnum(Enum):
    def _generate_next_value_(name, st, count, last_values):
        return name


class ModelEnums(AutoEnum):
    GPT2LMREModel = 'GPT2LMREModel'
    GPT2LMHeadModel = 'GPT2LMHeadModel'


class IdxDataEnums(AutoEnum):
    default = 'default'
    entities = 'entities'
    sentences = 'sentences'
    full = 'full'


class TrainModesEnums(AutoEnum):
    train_eval = 'train_eval'
    eval_sequences = 'eval_sequences'


class ConfigEnums(AutoEnum):
    mode = 'mode'
    load_path = 'load_path'
    save_path = 'save_path'
    save_model = 'save_model'
    epoch = 'epoch'
    epoch_iter = 'epoch_iter'
    batch_size = 'batch_size'
    batch_len = 'batch_len'
    learning_rate = 'learning_rate'
    weight_decay = 'weight_decay'
    max_len = 'max_len'
    model = 'model'
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


_me = ModelEnums
_ide = IdxDataEnums
_tme = TrainModesEnums
_ce = ConfigEnums

model_templates = {
    _me.GPT2LMHeadModel: tfm.GPT2LMHeadModel,
    _me.GPT2LMREModel: te.GPT2LMREModel
}

model_modes_func = {
    _tme.train_eval: single_train,
    _tme.eval_sequences: single_sequence_generation
}

dataset_templates = {
    _ide.default: dh.IdxDataset,
    _ide.entities: dh.IdxEntityDataset,
    _ide.sentences: dh.IdxTextDataset,
    _ide.full: dh.IdxFullDataset
}

dataset_fields = {
    'idx': {
        _ide.default: {_ce.idx_path},
        _ide.entities: {_ce.idx_path, _ce.ent_path},
        _ide.sentences: {_ce.idx_path, _ce.sent_path},
        _ide.full: {_ce.idx_path, _ce.sent_path, _ce.ent_path}
    }
}

mode_fields = {
    _tme.train_eval: {
        _ce.load_path, _ce.save_model, _ce.epoch, _ce.epoch_iter, _ce.batch_size, _ce.learning_rate,
        _ce.weight_decay, _ce.max_len, _ce.ent_path, _ce.sent_path, _ce.idx_path, _ce.idx_index_path,
        _ce.from_checkpoint, _ce.continue_train, _ce.eval_len
    },
    _tme.eval_sequences: {
        _ce.load_path, _ce.batch_len, _ce.max_len, _ce.ent_path, _ce.idx_path, _ce.num_samples, _ce.from_checkpoint,
        _ce.continue_train
    }
}

data_process_func = {
    _tme.train_eval: {
        _me.GPT2LMREModel: {
            'idx': {
                _ide.entities: lambda x, max_len=np.inf, batch_size=32: libs.process_re_data(
                    libs.get_re_data(x, max_len=max_len, batch_size=batch_size)),
                _ide.sentences: lambda x, max_len=np.inf, batch_size=32: libs.get_tensor_batch(
                    x, max_len=max_len, batch_size=batch_size),
                _ide.full: lambda x, max_len=np.inf, batch_size=32: libs.process_re_data(
                    libs.get_re_data(x, max_len=max_len, batch_size=batch_size))
            }
        },
        _me.GPT2LMHeadModel: {
            'idx': {
                _ide.sentences: lambda x, max_len=np.inf, batch_size=32:
                libs.get_tensor_batch(libs.get_column(x, 0), max_len=max_len, batch_size=batch_size),
                _ide.full: lambda x, max_len=np.inf, batch_size=32:
                libs.get_tensor_batch(libs.get_re_data(x, max_len=max_len, batch_size=batch_size)['sent'])
            }
        }
    },
    _tme.eval_sequences: {
        'idx': {
            _ide.entities: lambda x: {'e1': x[0], 'e2': x[1], 'idx': x[2]}
        }
    }
}

default_values = {
    _ce.mode: 'train_eval',
    _ce.load_path: 'gpt2-medium',
    _ce.save_model: True,
    _ce.epoch: 5,
    _ce.epoch_iter: 100,
    _ce.batch_size: 16,
    _ce.batch_len: 100,
    _ce.learning_rate: 0.001,
    _ce.weight_decay: 0.0001,
    _ce.max_len: 512,
    _ce.model_select: 'GPT2LMHeadModel',
    _ce.dataset_select: 'IdxFullDataset',
    _ce.ent_path: lab_data_path + 'wiki2016_ent',
    _ce.sent_path: self_data_path + 'wiki2016_sents_mapped',
    _ce.idx_path: self_data_path + 'wiki2016_idx',
    _ce.idx_index_path: self_data_path + 'wiki2016_idx_indexes',
    _ce.num_samples: 20,
    _ce.eval_len: 10,
    _ce.continue_train: False,
    _ce.from_checkpoint: False
}
