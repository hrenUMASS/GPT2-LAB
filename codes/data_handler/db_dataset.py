import json

import numpy as np
import regex as re
from torch.utils.data import Dataset

import libs
from libs import encode, safe_sql, split_array


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
        self.tokenizer = tokenizer
        self.data = np.array(self.data)
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
                    safe_sql(self.cursor, 'SELECT * FROM entities WHERE id IN ({})'
                             .format(','.join('?' * len(seg))), arguments=seg)
                )
                # self.ent_data.extend(
                #     self.cursor.execute('SELECT * FROM entities WHERE id IN ({})'
                #                         .format(','.join('?' * len(seg))), seg)
                # )
            self.cursor.execute('COMMIT')
            self.ent_data = {e[0]: ' ' + e[1].strip() for e in self.ent_data}

    def __getitem__(self, item):
        idx = self.data[item]
        e1, e2 = self.ent_data[idx[1]], self.ent_data[idx[2]]
        # return encode(self.tokenizer, e1), encode(self.tokenizer, e2), idx
        return encode(self.tokenizer, e1), encode(self.tokenizer, e2), idx


class IdxFullDBDataset(IdxEpDBDataset):

    def __init__(self, tokenizer, cursor=None, start_id=0, batch_len=100, data=None, ids=None, select_between=True,
                 **kwargs):
        super(IdxFullDBDataset, self).__init__(tokenizer=tokenizer, cursor=cursor, start_id=start_id,
                                               batch_len=batch_len, data=data, ids=ids, **kwargs)
        self.select_between = select_between
        self.data = np.array(self.data)
        sids = {int(x) for x in self.data[:, 3]}
        # print(len(sids))
        if data is not None and 'sent' in data:
            self.sent_data = data['sent']
        else:
            self.sent_data = []
            sids_seg = split_array(sids)
            self.cursor.execute('BEGIN TRANSACTION')
            for seg in sids_seg:
                self.sent_data.extend(
                    safe_sql(self.cursor, 'SELECT * FROM sentences WHERE id IN ({})'
                             .format(','.join('?' * len(seg))), arguments=seg)
                )
                # self.sent_data.extend(
                #     self.cursor.execute('SELECT * FROM sentences WHERE id IN ({})'
                #                         .format(','.join('?' * len(seg))), seg)
                # )
            self.cursor.execute('COMMIT')
            self.sent_data = {s[0]: s[1] for s in self.sent_data}
            # self.sent_data = {s[0]: encode(tokenizer, s[1], add_eos=True, add_prefix_space=True) for s in
            #                   self.sent_data}
            # print(len(self.sent_data))

            # print(list(self.sent_data.keys()))
            # print(self.data)

    def get_loaded_length(self):
        return len(self.data)

    def __getitem__(self, item):
        # e1, e2, idx = IdxEpDBDataset.__getitem__(self, item)
        idx = self.data[item]
        e1, e2 = self.ent_data[idx[1]].strip(), self.ent_data[idx[2]].strip()
        sent = self.sent_data[idx[3]].strip()
        # print(e1, e2, sent)
        try:
            if self.select_between:
                e1a, e2a = [], []
                for e1m in re.finditer('\\b' + e1 + '\\b', sent):
                    e1a.append(e1m.start())
                for e2m in re.finditer('\\b' + e2 + '\\b', sent):
                    e2a.append(e2m.start())
                if len(e1a) == 0 or len(e2a) == 0:
                    e1i, e2i = sent.index(e1), sent.index(e2)
                    start = min(e1i, e2i)
                    if start == e1i:
                        end = e2i + len(e2)
                    else:
                        end = e1i + len(e1)
                else:
                    start = min(e1a + e2a)
                    if start in e1a:
                        end = min(e2a) + len(e2)
                    else:
                        end = min(e1a) + len(e1)
                sent = sent[start:end]
        except:
            pass
        # log_info(libs.sample_logger, '{}\t{}'.format(e1, e2))
        # log_info(libs.sample_logger, '{}'.format(sent))
        sent = encode(self.tokenizer, sent, add_eos=False, add_prefix_space=True)
        e1, sent = self.unify_entity(e1, sent, idx[3])
        e2, sent = self.unify_entity(e2, sent, idx[3])
        # print(type(e1), type(e2), type(sent))
        # print(e1, e2, sent, '\n')
        # sent = self.sent_data[idx[3]]
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
            return None, False

        space = 'Ġ'
        sent_tok = self.tokenizer.convert_ids_to_tokens(sent.tolist())
        ent_temp = ''.join(self.tokenizer.tokenize(ent))
        # print(ent_temp)
        # print(sent, sent_tok, ent)
        for i in range(len(sent_tok)):
            tmp = sent_tok[i].replace(space, '')
            if ent_temp.startswith(tmp):
                ent_tok, changed = in_tensor(ent_temp, sent_tok, i)
                if ent_tok is not None:
                    # if changed:
                    #     self.sent_data[sent_idx] = encode(self.tokenizer, sent_tok)
                    return encode(self.tokenizer, ent_tok), encode(self.tokenizer, sent_tok)
        return encode(self.tokenizer, ent), encode(self.tokenizer, sent_tok)


class IdxClsDataset(IdxDBDataset):

    def __init__(self, tokenizer, cursor, start_id=0, batch_len=100, data=None, ids=None, **kwargs):
        super(IdxClsDataset, self).__init__(cursor=cursor, start_id=start_id, batch_len=batch_len, data=data,
                                            ids=ids, **kwargs)

        self.tokenizer = tokenizer

    def __getitem__(self, item):
        # sent label idx
        return encode(self.tokenizer, self.data[item][2]), self.data[item][1], self.data[item][0]
