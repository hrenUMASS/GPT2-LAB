import json
import sqlite3

from .abstract_indexer import AbstractIndexer


class DBIndexer(AbstractIndexer):

    def __init__(self, db_path, ids=None, batch_len_size=3200):
        super(DBIndexer, self).__init__()
        if isinstance(ids, str):
            with open(ids, 'r') as f:
                ids = json.load(f)
        self.ids = ids
        self.db_path = db_path
        self.start_id = 0
        self.batch_size = batch_len_size

    def get_dataset(self, nth_loader, prototype=None, tokenizer=None, dataset_type=None, batch_len=100, data=None):
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

        ids = self.ids
        start_id = self.start_id
        if ids is not None:
            ids = ids[nth_loader * batch_len * self.batch_size + 1:(nth_loader + 1) * batch_len * self.batch_size + 1]
        else:
            self.start_id += self.batch_size * batch_len
        params = {
            'tokenizer': tokenizer,
            'db_path': self.db_path,
            'start_id': start_id,
            'batch_len': batch_len,
            'data': data,
            'ids': ids
        }
        return dataset_type(**params)

    def get_eval(self, prototype=None, tokenizer=None, dataset_type=None, eval_len=10, data=None):
        if eval_len == 0:
            return None
        if self.eval is not None:
            return self.eval
        if self.ids is not None:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            max_data = c.execute('SELECT id FROM idx order by id desc LIMIT 1').fetchall()[0]
            start_id = max_data - 3200 * eval_len
            temp_id = self.start_id
            self.start_id = start_id
            eva = self.get_dataset(0, prototype=prototype, tokenizer=tokenizer, dataset_type=dataset_type,
                                   batch_len=eval_len, data=data)
            self.start_id = temp_id
        else:
            nth_loader = len(self.ids) // self.batch_size - eval_len
            eva = self.get_dataset(nth_loader, prototype=prototype, tokenizer=tokenizer, dataset_type=dataset_type,
                                   batch_len=eval_len, data=data)
        self.eval = eva
        return eva
