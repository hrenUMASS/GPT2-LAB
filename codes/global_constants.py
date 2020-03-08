from enum import Enum

import numpy as np
import torch
import transformers as tfm

import data_handler as dh
import libs
import train_eval as te
from start_module import single_train, single_sequence_generation, gpt2_model_eval

lab_data_path = '/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/data/raw/wiki2016_nchunk_entity_agg/'
self_data_path = '/iesl/canvas/hren/gpt2_wiki_lab/data/'
ignore_index = -1
eos_id = 50256
main_device = torch.device('cuda:0')


class DatasetParam:

    def __init__(self, class_type, *fields):
        self.class_type = class_type
        self.fields = set(fields)


class ModeParam:
    def __init__(self, func, *fields):
        self.func = func
        self.fields = set(fields)


class ConfigEnums(Enum):
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


_ce = ConfigEnums


class ModelEnums(Enum):
    GPT2LMREModel = te.GPT2LMREModel
    GPT2LMHeadModel = tfm.GPT2LMHeadModel


class DatasetEnums(Enum):
    idxDefault = DatasetParam(dh.IdxDataset, _ce.idx_index_path, _ce.idx_path)
    idxEnts = DatasetParam(dh.IdxEntityDataset, _ce.idx_index_path, _ce.idx_path, _ce.ent_path)
    idxSents = DatasetParam(dh.IdxTextDataset, _ce.idx_index_path, _ce.idx_path, _ce.sent_path)
    idxFull = DatasetParam(dh.IdxFullDataset, _ce.idx_index_path, _ce.idx_path, _ce.sent_path, _ce.ent_path)
    idxDBFull = DatasetParam(dh.IdxFullDBDataset, _ce.db_path, _ce.ids)


class DataIndexerEnums(Enum):
    idxDefaultIndexer = dh.DataIndexer
    idxDBIndexer = dh.DBIndexer


class TrainModesEnums(Enum):
    train_eval = ModeParam(single_train,
                           _ce.mode, _ce.model, _ce.load_path, _ce.save_path, _ce.save_model, _ce.idx_path,
                           _ce.sent_path, _ce.ent_path, _ce.sent_index_path, _ce.ent_index_path, _ce.idx_index_path,
                           _ce.db_path, _ce.ids, _ce.indexer_type,
                           _ce.dataset_type, _ce.loaders, _ce.batch_len, _ce.eval_len, _ce.epochs, _ce.batch_size,
                           _ce.learning_rate, _ce.weight_decay, _ce.max_len,
                           _ce.from_checkpoint, _ce.continue_train, _ce.data)
    eval_sequences = ModeParam(single_sequence_generation,
                               _ce.mode, _ce.model, _ce.load_path, _ce.save_path, _ce.idx_path, _ce.sent_path,
                               _ce.ent_path, _ce.idx_index_path, _ce.sent_index_path, _ce.ent_index_path, _ce.ent_data,
                               _ce.db_path, _ce.ids, _ce.indexer_type,
                               _ce.dataset_type, _ce.loaders, _ce.batch_len, _ce.max_len, _ce.num_samples,
                               _ce.from_checkpoint, _ce.continue_train, _ce.data)
    gpt2_model_eval = ModeParam(gpt2_model_eval, _ce.mode, _ce.model, _ce.load_path, _ce.save_path, _ce.idx_path,
                                _ce.sent_path, _ce.ent_path, _ce.idx_index_path, _ce.sent_index_path, _ce.db_path,
                                _ce.ent_index_path, _ce.ent_data, _ce.gpt2, _ce.ids, _ce.indexer_type,
                                _ce.dataset_type, _ce.loaders, _ce.batch_len, _ce.max_len,
                                _ce.from_checkpoint, _ce.continue_train, _ce.data)


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
                libs.get_re_data(x, max_len=max_len, batch_size=batch_size)))
        },
        _me.GPT2LMHeadModel: {
            _de.idxSents: lambda max_len=np.inf, batch_size=32:
            (lambda x: libs.get_tensor_batch(
                libs.get_column(x, 0), max_len=max_len, batch_size=batch_size)),
            _de.idxFull: lambda max_len=np.inf, batch_size=32:
            (lambda x: libs.get_tensor_batch(
                libs.get_re_data(x, max_len=max_len, batch_size=batch_size)['sent']))
        }
    },
    _tme.eval_sequences: {
        _me.GPT2LMREModel: {
            _de.idxEnts: lambda max_len=np.inf, batch_size=1: (lambda x: {'e1': x[0], 'e2': x[1], 'idx': x[-1]}),
            _de.idxFull: lambda max_len=np.inf, batch_size=1: (lambda x: {'e1': x[0], 'e2': x[1], 'idx': x[-1]})
        }
    },
    _tme.gpt2_model_eval: {
        _me.GPT2LMREModel: {
            _de.idxFull: lambda max_len=np.inf, batch_size=32: (
                lambda x: libs.get_re_data(x, max_len=max_len, batch_size=batch_size)),
            _de.idxDBFull: lambda max_len=np.inf, batch_size=32: (
                lambda x: libs.get_re_data(x, max_len=max_len, batch_size=batch_size))
        }
    }
}

default_values = {
    _ce.mode: 'train_eval',
    _ce.load_path: 'gpt2-medium',
    _ce.gpt2: 'gpt2',
    _ce.save_model: True,
    _ce.epochs: 5,
    # _ce.epoch_iter: 100,
    _ce.batch_size: 16,
    _ce.batch_len: 100,
    _ce.loaders: 10,
    _ce.learning_rate: 0.001,
    _ce.weight_decay: 0.0001,
    _ce.max_len: 512,
    _ce.model: 'GPT2LMHeadModel',
    _ce.dataset_type: 'IdxFullDataset',
    _ce.ent_path: lab_data_path + 'wiki2016_ent',
    _ce.sent_path: self_data_path + 'wiki2016_sents_mapped',
    _ce.idx_path: self_data_path + 'wiki2016_idx',
    _ce.idx_index_path: self_data_path + 'wiki2016_idx_indexes',
    _ce.ent_index_path: self_data_path + 'wiki2016_ents_indexes',
    _ce.sent_index_path: self_data_path + 'wiki2016_sents_indexes',
    _ce.ent_data: self_data_path + 'entity_vocab.pt',
    _ce.num_samples: 20,
    _ce.eval_len: 10,
    _ce.continue_train: False,
    _ce.from_checkpoint: False,
    _ce.db_path: self_data_path + 'wiki2016.db',
    _ce.ids: self_data_path + 'rexT/filtered.json',
    _ce.indexer_type: 'idxDefaultIndexer'
}
