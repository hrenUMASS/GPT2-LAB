import GPUtil

import libs


class DataIndexer:

    def __init__(self, idx_index_path, idx_path, sent_path=None, ent_path=None):
        self.indexes = []
        with open(idx_index_path, 'r') as f:
            for line in f:
                self.indexes.append(int(line[:-1]))
        self.idx_file = open(idx_path, 'r')
        self.sent_file = open(sent_path, 'r') if sent_path is not None else None
        self.ent_file = open(ent_path, 'r') if ent_path is not None else None
        self.eval = None

    def get_dataset_pos(self, nth_loader, batch_len):
        return _get_dataset_pos(nth_loader, batch_len, self.indexes)

    def get_dataset(self, nth_loader, prototype=None, tokenizer=None, dataset_type=None, batch_len=100):
        index = self.get_dataset(nth_loader, batch_len)
        result = _get_dataset(index, self.idx_file, sent_file=self.sent_file, ent_file=self.ent_file,
                              prototype=prototype, tokenizer=tokenizer, dataset_type=dataset_type, batch_len=batch_len)
        prepare_logger, cuda_logger = libs.prepare_logger, libs.cuda_logger
        gpu = GPUtil.getGPUs()[0]
        libs.log_info(prepare_logger, 'Load idxs {} sentences {}'.format(*result.get_loaded_length()))
        libs.log_info(cuda_logger, "Allocated data {}".format(libs.cuda_mem_in_mb()))
        libs.log_info(cuda_logger,
                      'GPU Free {} Used {} Total {}'.format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryTotal))
        return result

    def get_eval(self, prototype=None, tokenizer=None, dataset_type=None, eval_len=10):
        if self.eval is not None:
            return self.eval
        eva = _get_dataset(self.indexes[-eval_len], self.idx_file, sent_file=self.sent_file, ent_file=self.ent_file,
                           prototype=prototype, tokenizer=tokenizer, dataset_type=dataset_type, batch_len=eval_len)
        self.eval = eva
        return self.eval


def _get_dataset_pos(nth_loader, batch_len, indexes):
    return indexes[nth_loader * batch_len]


def _get_dataset(index, idx_file, sent_file=None, ent_file=None,
                 prototype=None, tokenizer=None, dataset_type=None, batch_len=100):
    data = {}
    if prototype is not None:
        data = dict(prototype.get_data())
        tokenizer = tokenizer or prototype.tokenizer
        dataset_type = dataset_type or type(prototype)

    if tokenizer:
        raise Exception('tokenizer is none!')
    if dataset_type is None:
        raise Exception('dataset type is none!')

    if 'data' in data:
        del data['data']
    idx_file.seek(index)
    params = {
        'tokenizer': tokenizer,
        'idx_file': idx_file,
        'ent_file': ent_file,
        'sent_file': sent_file,
        'batch_len': batch_len,
        'data': data
    }
    return dataset_type(**params)
