import json
import sqlite3

import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from libs import encode, safe_sql, split_array, get_between
import libs


class IdxDBDataset(Dataset):

    def __init__(self, cursor=None, start_id=0, batch_len=100, data=None, ids=None, **kwargs):
        self.cursor = cursor
        if data is not None and 'data' in data:
            self.data = data['data']
        elif ids is not None:
            if isinstance(ids, str):
                with open(ids, 'r') as f:
                    ids = json.load(f)
            self.data = []
            ids_seg = split_array(ids)
            cursor.execute('BEGIN TRANSACTION')
            for seg in ids_seg:
                # print(seg)
                self.data.extend(
                    safe_sql(self.cursor, 'SELECT * FROM idx WHERE id IN ({})'
                             .format(','.join('?' * len(seg))), arguments=seg)
                )
            cursor.execute('COMMIT')
        else:
            self.data = safe_sql(self.cursor, 'SELECT * FROM idx WHERE id > ? AND id <= ?',
                                 (start_id, start_id + batch_len))
        self.data = np.array(self.data)
        libs.log_info(libs.prepare_logger, 'loaded data length {}'.format(len(self.data)))

    def get_loaded_length(self):
        return len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        idx = self.data[item]
        return idx

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


class IdxEpDBDataset(IdxDBDataset):
    def __init__(self, tokenizer, cursor=None, start_id=0, batch_len=100, data=None, ids=None, **kwargs):
        super(IdxEpDBDataset, self).__init__(cursor=cursor, start_id=start_id, batch_len=batch_len, data=data,
                                             ids=ids, **kwargs)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        eids = {int(x) for x in self.data[:, [1, 2]].reshape(-1, 1).squeeze(1)}
        if data is not None and 'ent' in data:
            self.ent_data = data['ent']
        else:
            self.ent_data = []
            eids_seg = split_array(eids)
            self.cursor.execute('BEGIN TRANSACTION')
            for seg in eids_seg:
                # print('seg', seg)
                # print(self.cursor.execute('SELECT * FROM entities WHERE id=?', (seg[0],)).fetchall())
                # print(all(isinstance(x, int) for x in seg))
                self.ent_data.extend(
                    safe_sql(self.cursor, 'SELECT id, entity FROM entities WHERE id IN ({})'
                             .format(','.join('?' * len(seg))), arguments=seg)
                )
                # self.ent_data.extend(
                #     self.cursor.execute('SELECT * FROM entities WHERE id IN ({})'
                #                         .format(','.join('?' * len(seg))), seg)
                # )
            self.cursor.execute('COMMIT')
            self.ent_data = {e[0]: e[1] for e in self.ent_data}

    def __getitem__(self, item):
        idx = self.data[item]
        e1, e2 = self.ent_data[idx[1]], self.ent_data[idx[2]]
        # return encode(self.tokenizer, e1), encode(self.tokenizer, e2), idx
        return encode(self.tokenizer, e1), encode(self.tokenizer, e2), idx


class IdxFullDBDataset(IdxEpDBDataset):

    def __init__(self, tokenizer, cursor=None, start_id=0, batch_len=100, data=None, ids=None, between=False, **kwargs):
        super(IdxFullDBDataset, self).__init__(tokenizer=tokenizer, cursor=cursor, start_id=start_id,
                                               batch_len=batch_len, data=data, ids=ids, **kwargs)
        sids = {int(x) for x in self.data[:, 3]}
        self.between = between
        # print(len(sids))
        if data is not None and 'sent' in data:
            self.sent_data = data['sent']
        else:
            self.sent_data = []
            sids_seg = split_array(sids)
            self.cursor.execute('BEGIN TRANSACTION')
            for seg in sids_seg:
                self.sent_data.extend(
                    safe_sql(self.cursor, 'SELECT id, sentence FROM sentences WHERE id IN ({})'
                             .format(','.join('?' * len(seg))), arguments=seg)
                )
                # self.sent_data.extend(
                #     self.cursor.execute('SELECT * FROM sentences WHERE id IN ({})'
                #                         .format(','.join('?' * len(seg))), seg)
                # )
            self.cursor.execute('COMMIT')
            self.sent_data = {s[0]: encode(tokenizer, s[1], add_eos=True, add_prefix_space=True) for s in
                              self.sent_data}
            # print(len(self.sent_data))

            # print(list(self.sent_data.keys()))
            # print(self.data)

    def get_loaded_length(self):
        return len(self.data)

    def __getitem__(self, item):
        # e1, e2, idx = IdxEpDBDataset.__getitem__(self, item)
        idx = self.data[item]
        e1, e2 = self.ent_data[idx[1]], self.ent_data[idx[2]]
        sent = self.sent_data[idx[3]]

        if self.between:
            tok = self.tokenizer
            sent = tok.decode(sent)
            a, b = get_between(tok.decode(e1), tok.decode(e2), sent, inclusive=False)
            sent = encode(tok, sent[a:b], add_eos=True,
                          add_prefix_space=True)

        e1, e2 = self.unify_entity(e1, sent, idx[3]), self.unify_entity(e2, sent, idx[3])
        sent = self.sent_data[idx[3]]
        # print('i', len(self), item)
        return e1, e2, sent, idx

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
        print(ent)

        ent_temp = ''.join(self.tokenizer.tokenize(ent))
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


class IdxClsDataset(IdxDBDataset):

    def __init__(self, tokenizer, cursor, start_id=0, batch_len=100, data=None, ids=None, **kwargs):
        super(IdxClsDataset, self).__init__(cursor=cursor, start_id=start_id, batch_len=batch_len, data=data,
                                            ids=ids, **kwargs)

        self.tokenizer = tokenizer

    def __getitem__(self, item):
        # sent label idx
        # print(self.data[item])
        return (encode(
            self.tokenizer,
            ' ' + self.data[item][2].strip()),
                int(self.data[item][1]),
                int(self.data[item][0]))
