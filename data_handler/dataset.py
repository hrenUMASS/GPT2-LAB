import numpy as np
import transformers as tfm
from torch.utils.data import Dataset

from libs import encode
from .data_indexer import DataIndexer


def _read_block(fp, total_len, tokenizer: tfm.PreTrainedTokenizer, max_len=None, valid_func=lambda x: True,
                process_func=lambda x: x, truncate_mode='truncate', max_step=np.inf, add_eos=False):
    result = []
    i, k = 0, 0
    raw = fp.readline()
    while raw != '' and i < total_len and k < max_step:
        line = process_func(raw)
        line = encode(tokenizer, line, add_prefix_space=True)
        if add_eos:
            line += [tokenizer.eos_token_id]
        if valid_func(line):
            if max_len is None or len(line) <= max_len:
                result.append(line)
                # print('normal', result[-1].shape)
                i += 1
            else:
                if truncate_mode == 'truncate':
                    result.append(line[:max_len])
                    # print('truncate', result[-1].shape)
                    i += 1
                elif truncate_mode == 'append':
                    k = 0
                    while k < len(line):
                        result.append(line[k:k + max_len])
                        # print('append', result[-1].shape)
                        k += max_len
                        i += 1
                elif truncate_mode == 'discard':
                    pass
                else:
                    raise ValueError('No such truncate mode {}'.format(truncate_mode))
        k += 1
        raw = fp.readline()
    # print(list(map(lambda x: x.shape, result)))
    return result


def get_tokens(tokenizer, ent):
    return tokenizer.encode(ent, add_prefix_space=True, return_tensors='pt')[0], \
           tokenizer.encode(ent, return_tensors='pt')[0]


class IdxDataset(Dataset):

    def __init__(self, tokenizer, idx_file=None, batch_len=100, data=None, data_indexer=None, **kwargs):
        # print('idx', total_len, eval_size, data)
        self.data_indexer: DataIndexer = data_indexer
        self.tokenizer = tokenizer
        self.batch_len = batch_len
        if data is not None and 'idx' in data:
            self.data = data['idx']
        else:
            self.data = self.load_idx(idx_file)
            # print(self.data)
        self.data = np.array(self.data)
        # print('idd', idx_file.tell())
        # print(self.data, len(self.data))

    def load_idx(self, idx_file):
        result = []
        data_len = self.batch_len * 3200
        print('loading idx', idx_file.name, 'len', data_len)
        # print(data_len)
        for i, line in enumerate(idx_file):
            if i >= data_len:
                break
            result.append(tuple(map(lambda x: int(x), line.split())))
        return result

    def get_loaded_length(self):
        return len(self.data)

    def __getitem__(self, item):
        idx = self.data[item]
        return idx[0], idx[1], idx[2]

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


class IdxTextDataset(IdxDataset):

    def __init__(self, tokenizer, idx_file=None, sent_file=None, batch_len=100, data=None, data_indexer=None, **kwargs):
        super(IdxTextDataset, self).__init__(tokenizer, idx_file=idx_file, batch_len=batch_len, data=data,
                                             data_indexer=data_indexer, **kwargs)
        self.sent_data = {}
        if data is not None and 'sent' in data:
            self.sent_data = data['sent']
            all_sents = self.data[:, 2]
            if not all(x in data['sent'] for x in all_sents):
                self.load_sent(sent_file)
        else:
            self.load_sent(sent_file)
        # print(self.sent_data)

    def load_sent(self, sent_file):
        print('loading sentences', sent_file.name)
        if self.data_indexer is None:
            sents = set(self.data[:, 2])
            for i in range(max(sents) + 1):
                sent = sent_file.readline()[:-1]
                if i in sents and i not in self.sent_data:
                    self.sent_data[i] = encode(self.tokenizer, sent, add_eos=True, add_prefix_space=True)
        else:
            sents = np.sort(self.data[:, 2])
            for item in sents:
                if item not in self.sent_data:
                    raw_index = self.data_indexer.get_text_pos(item, 'sent')
                    while raw_index < item:
                        raw_index += 1
                        sent_file.readline()
                    self.sent_data[item] = encode(self.tokenizer, sent_file.readline()[:-1], add_eos=True,
                                                  add_prefix_space=True)
        return self.sent_data

    def get_loaded_length(self):
        return IdxDataset.get_loaded_length(self), len(self.sent_data)

    def __getitem__(self, item):
        _, _, senti = IdxDataset.__getitem__(self, item)
        return self.sent_data[senti], senti

    def get_data(self):
        result = {'sent': self.sent_data}
        result.update(IdxDataset.get_data(self))
        return result


class IdxEntityDataset(IdxDataset):

    def __init__(self, tokenizer, idx_file=None, ent_file=None, batch_len=100, data=None, data_indexer=None, **kwargs):
        super(IdxEntityDataset, self).__init__(tokenizer, idx_file=idx_file, batch_len=batch_len, data=data,
                                               data_indexer=data_indexer, **kwargs)
        self.ent_data = {}
        if data is not None and 'ent' in data:
            self.ent_data = data['ent']
            all_ents = self.data[:, [0, 1]].reshape(1, -1).squeeze()
            if not all(x in data['ent'] for x in all_ents):
                print('Not all entity in idx exists in ent')
                self.load_ent(ent_file)
        else:
            self.load_ent(ent_file)
        # print(self.ent_data)

    def load_ent(self, ent_file):
        print('loading entities', ent_file.name)
        # print('ent seek', ent_file.tell())
        ents = self.data[:, [0, 1]].reshape(1, -1).squeeze()
        if self.data_indexer is None:
            ents = set(ents)
            for i in range(max(ents) + 1):
                ent = ent_file.readline()[:-1]
                if i in ents and i not in self.ent_data:
                    self.ent_data[i] = ent
        else:
            ents = np.sort(ents)
            for item in ents:
                if item not in self.ent_data:
                    raw_index = self.data_indexer.get_text_pos(item, 'ent')
                    while raw_index < item:
                        raw_index += 1
                        ent_file.readline()
                    self.ent_data[item] = ent_file.readline()[:-1]
        return self.ent_data

    def get_loaded_length(self):
        return IdxDataset.get_loaded_length(self), len(self.ent_data)

    def __getitem__(self, item):
        e1i, e2i, _ = IdxDataset.__getitem__(self, item)
        return self.ent_data[e1i], self.ent_data[e2i], (e1i, e2i)

    def get_data(self):
        result = {'ent': self.ent_data}
        result.update(IdxDataset.get_data(self))
        return result


class IdxFullDataset(IdxEntityDataset, IdxTextDataset):

    def __init__(self, tokenizer, idx_file=None, ent_file=None, sent_file=None, batch_len=100, data=None,
                 data_indexer=None, **kwargs):
        super(IdxFullDataset, self).__init__(tokenizer, idx_file=idx_file, ent_file=ent_file, sent_file=sent_file,
                                             batch_len=batch_len, data=data, data_indexer=data_indexer, **kwargs)

    def __getitem__(self, item):
        e1, e2, eidx = IdxEntityDataset.__getitem__(self, item)
        sent, senti = IdxTextDataset.__getitem__(self, item)
        e1, e2 = self.unify_entity(e1, sent, senti), self.unify_entity(e2, sent, senti)
        sent, senti = IdxTextDataset.__getitem__(self, item)
        return e1, e2, sent, (*eidx, senti)

    def get_loaded_length(self):
        # print(vars(self))
        return IdxDataset.get_loaded_length(self), len(self.sent_data), len(self.ent_data)

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
        sent_tok = self.tokenizer.convert_ids_to_tokens(sent)
        ent_temp = ''.join(self.tokenizer.tokenize(ent))
        # print(ent_temp)
        for i in range(len(sent_tok)):
            tmp = sent_tok[i].replace(space, '')
            if ent_temp.startswith(tmp):
                ent_tok, changed = in_tensor(ent_temp, sent_tok, i)
                if ent_tok is not None:
                    if changed:
                        self.sent_data[sent_idx] = encode(self.tokenizer, sent_tok)
                    return encode(self.tokenizer, ent_tok)
        return encode(self.tokenizer, ent)

    def get_data(self):
        result = IdxTextDataset.get_data(self)
        result.update(IdxEntityDataset.get_data(self))
        return result


if __name__ == '__main__':
    pass
    # import transformers as tfm
    # from torch.utils.data import DataLoader
    # from gpt2_train import get_tensor_batch

    # tok = tfm.GPT2Tokenizer.from_pretrained('../gpt2_pretrained')
    # a = TextDataset('../../data/wiki2016_sents', tok, 16, max_len=512)
    # b = DataLoader(a, batch_size=4, collate_fn=lambda x: x)
    # for i in b:
    #     print(i)
    # print(get_tensor_batch(i))
    # print(get_tensor_batch(i))
