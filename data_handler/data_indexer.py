# import GPUtil

import libs
from .abstract_indexer import AbstractIndexer


class DataIndexer(AbstractIndexer):

    def __init__(self, idx_index_path, idx_path, sent_path=None, ent_path=None, sent_index_path=None,
                 ent_index_path=None):
        super(DataIndexer, self).__init__()
        self.indexes = []
        self.ent_indexes = []
        self.sent_indexes = []
        with open(idx_index_path, 'r') as f:
            for line in f:
                self.indexes.append(int(line[:-1]))
        if sent_index_path is not None:
            with open(sent_index_path, 'r') as f:
                for line in f:
                    self.sent_indexes.append(int(line[:-1]))
        if ent_index_path is not None:
            with open(ent_index_path, 'r') as f:
                for line in f:
                    self.ent_indexes.append(int(line[:-1]))
        self.idx_file = open(idx_path, 'r')
        self.sent_file = open(sent_path, 'r') if sent_path is not None else None
        self.ent_file = open(ent_path, 'r') if ent_path is not None else None

    def get_dataset_pos(self, nth_loader, batch_len):
        return _get_dataset_pos(nth_loader, batch_len, self.indexes)

    def get_dataset(self, nth_loader, prototype=None, tokenizer=None, dataset_type=None, batch_len=100, data=None):
        index = self.get_dataset_pos(nth_loader, batch_len)
        # print('iid', nth_loader, batch_len, index)
        self.idx_file.seek(index)
        result = _get_dataset(self.idx_file, sent_file=self.sent_file, ent_file=self.ent_file,
                              prototype=prototype, tokenizer=tokenizer, dataset_type=dataset_type, batch_len=batch_len,
                              data=data)
        prepare_logger, cuda_logger = libs.prepare_logger, libs.cuda_logger
        # gpu = GPUtil.getGPUs()[0]
        libs.log_info(prepare_logger, 'Load idxs {} sentences {}'.format(*result.get_loaded_length()))
        libs.log_info(cuda_logger, "Allocated data {}".format(libs.cuda_mem_in_mb()))
        # libs.log_info(cuda_logger,
        #               'GPU Free {} Used {} Total {}'.format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryTotal))
        return result

    def get_eval(self, prototype=None, tokenizer=None, dataset_type=None, eval_len=10, data=None):
        if self.eval is not None:
            return self.eval
        self.idx_file.seek(self.indexes[-eval_len])
        eva = _get_dataset(self.idx_file, sent_file=self.sent_file, ent_file=self.ent_file,
                           prototype=prototype, tokenizer=tokenizer, dataset_type=dataset_type, batch_len=eval_len,
                           data=data)
        self.eval = eva
        return self.eval

    def get_text_pos(self, real_index, index_type):
        if index_type == 'ent':
            return _get_text_pos(real_index, self.ent_indexes)
        return _get_text_pos(real_index, self.sent_indexes)


def _get_dataset_pos(nth_loader, batch_len, indexes):
    return indexes[nth_loader * batch_len]


def _get_text_pos(real_index, indexes):
    return indexes[real_index // 3200]


def _get_dataset(idx_file, sent_file=None, ent_file=None,
                 prototype=None, tokenizer=None, dataset_type=None, batch_len=100, data=None):
    if prototype is not None:
        if data is None:
            data = dict(prototype.get_data())
        tokenizer = tokenizer or prototype.tokenizer
        dataset_type = dataset_type or type(prototype)

    if tokenizer is None:
        raise Exception('tokenizer is none!')
    if dataset_type is None:
        raise Exception('dataset type is none!')

    if data is not None and 'data' in data:
        del data['data']

    if sent_file is not None:
        sent_file.seek(0)
    if ent_file is not None:
        ent_file.seek(0)

    params = {
        'tokenizer': tokenizer,
        'idx_file': idx_file,
        'ent_file': ent_file,
        'sent_file': sent_file,
        'batch_len': batch_len,
        'data': data
    }

    return dataset_type(**params)
