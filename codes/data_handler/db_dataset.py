import json
import sqlite3

import numpy as np
from torch.utils.data import Dataset

from libs import encode


class IdxFullDBDataset(Dataset):

    def __init__(self, tokenizer, db_path=None, start_id=0, batch_len=100, data=None, ids=None, **kwargs):
        # print('idx', total_len, eval_size, data)
        self.tokenizer = tokenizer
        self.db = sqlite3.connect(db_path)
        self.cursor = self.db.cursor()
        # print(self.data)
        if data is not None and 'data' in data:
            self.data = data['data']
        elif ids is not None:
            if isinstance(ids, str):
                with open(ids, 'r') as f:
                    ids = json.load(f)
            self.data = self.cursor.execute('SELECT e1, e2, sent FROM idx WHERE id IN {}'.format(tuple(ids))).fetchall()
        else:
            self.data = self.cursor.execute('SELECT e1, e2, sent FROM idx WHERE id > ? AND id <= ?',
                                            (start_id, start_id + batch_len * 3200)).fetchall()
            # print(self.data, start_id, batch_len * 3200, self.cursor.execute('select MAX(id) from idx').fetchall())
        self.data = np.array(self.data)
        eids = set(self.data[:, [0, 1]].reshape(-1, 1).squeeze(1))
        # print(max(eids))
        # print(eids)
        if data is not None and 'ent' in data:
            self.ent_data = data['ent']
        else:
            self.ent_data = self.cursor.execute('SELECT * FROM entities WHERE id IN {}'.format(tuple(eids))).fetchall()
            self.ent_data = {e[0]: e[1] for e in self.ent_data}
            # print(self.ent_data)
        sids = set(self.data[:, 2])
        # print(max(sids))
        # print(sids)
        if data is not None and 'sent' in data:
            self.sent_data = data['sent']
        else:
            # print(sids)
            self.sent_data = self.cursor.execute(
                'SELECT * FROM sentences WHERE id IN {}'.format(tuple(sids))).fetchall()
            # print(self.sent_data)
            # print(type(self.sent_data))
            # print(self.sent_data[:10])
            self.sent_data = {s[0]: encode(tokenizer, s[1], add_eos=True, add_prefix_space=True) for s in
                              self.sent_data}
            # print(type(self.sent_data), self.sent_data)
            # print(self.sent_data)
        # print('idd', idx_file.tell())
        # print(self.data, len(self.data))

    def get_loaded_length(self):
        return len(self.data)

    def __getitem__(self, item):
        # print(self.ent_data)
        idx = self.data[item]
        e1, e2 = self.ent_data[idx[-3]], self.ent_data[idx[-2]]
        sent = self.sent_data[idx[-1]]
        # print(idx, idx[-1] in self.sent_data, sent)
        e1, e2 = self.unify_entity(e1, sent, idx[-1]), self.unify_entity(e2, sent, idx[-1])
        sent = self.sent_data[idx[-1]]
        return e1, e2, sent, idx

    def __len__(self):
        return len(self.data)

    def get_data(self):
        return {'data': self.data}

    def split(self, partitions, eval_size=0, split_mark=('data',)):
        n = partitions
        l = len(self.data)
        param = dict(vars(self))
        param['eval_size'] = eval_size
        result = []
        data = self.get_data()
        temp_data = dict(data)
        for i in range(n):
            for k in temp_data:
                if k in split_mark:
                    temp_data[k] = data[k][i:l:n]
            param['data'] = temp_data
            result.append(type(self)(**param))
        return result

    def unify_entity(self, ent, sent, sent_idx):
        def in_tensor(ent, sent_tok, idx):
            tot = ''
            changed = False
            for k in range(idx, len(sent_tok)):
                temp = sent_tok[k].replace(space, '')
                if temp[:2] == '.,' and len(tot) == len(ent) - 1:
                    tot += '.'
                else:
                    tot = (tot + temp) if ent.startswith(tot + temp) else (
                        (tot + space + temp) if ent.startswith(tot + space + temp) else None)
                if tot is None:
                    return None, False
                elif tot == ent:
                    if temp[:2] == '.,':
                        sent_tok[k] = '.'
                        sent_tok.insert(k + 1, temp[1:])
                        changed = True
                    return sent_tok[idx: k + 1], changed

        space = 'Ġ'
        sent_tok = self.tokenizer.convert_ids_to_tokens(sent.tolist())
        ent_temp = ''.join(self.tokenizer.tokenize(ent))
        a = 'I am here.'
        b = self.tokenizer.encode(a, return_tensors='pt')[0]
        # print(ent_temp)
        # print(sent, sent_tok, ent)
        for i in range(len(sent_tok)):
            tmp = sent_tok[i].replace(space, '')
            if ent_temp.startswith(tmp):
                ent_tok, changed = in_tensor(ent_temp, sent_tok, i)
                if ent_tok is not None:
                    if changed:
                        self.sent_data[sent_idx] = encode(self.tokenizer, sent_tok)
                    return encode(self.tokenizer, ent_tok)
        return encode(self.tokenizer, ent)
