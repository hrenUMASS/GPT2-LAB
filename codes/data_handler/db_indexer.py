import json
import sqlite3

from .abstract_indexer import AbstractIndexer


class DBIndexer(AbstractIndexer):

    def __init__(self, db_path, ids=None, batch_size=32):
        super(DBIndexer, self).__init__()
        if isinstance(ids, str):
            with open(ids, 'r') as f:
                ids = json.load(f)
        self.ids = ids
        print(db_path)
        self.cursor = sqlite3.connect(db_path).cursor()
        self.start_id = 0
        self.batch_size = batch_size

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
            ids = ids[nth_loader * batch_len * self.batch_size:(nth_loader + 1) * batch_len * self.batch_size]
        else:
            self.start_id += self.batch_size * batch_len
        params = {
            'tokenizer': tokenizer,
            'cursor': self.cursor,
            'start_id': start_id,
            'batch_len': batch_len,
            'data': data,
            'ids': ids
        }
        # print(params)
        if ids is not None and len(ids) < 1:
            return None
        return dataset_type(**params)

    def get_eval(self, prototype=None, tokenizer=None, dataset_type=None, eval_len=10, data=None):
        from libs import safe_sql
        if eval_len == 0:
            return None
        if self.eval is not None:
            return self.eval
        if self.ids is None:

            max_data = safe_sql(self.cursor, 'SELECT id FROM idx order by id desc LIMIT 1')[0]
            if isinstance(max_data, list) or isinstance(max_data, tuple):
                max_data = max_data[0]
            start_id = max_data - self.batch_size * eval_len
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
