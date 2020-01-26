import numpy as np
import transformers as tfm
from torch.utils.data import Dataset

from .util import encode


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


class DatasetWithEval(Dataset):
    def __init__(self, tokenizer, total_len, eval_size=0, **kwargs):
        self.tokenizer = tokenizer
        self.total_len = total_len
        self.eval = False
        self.eval_size = eval_size
        self.data = []

    def change_mode(self, evaluate=True):
        self.eval = evaluate
        return self

    def __len__(self):
        if self.eval:
            return self.eval_size
        return self.total_len

    def __getitem__(self, item):
        if self.eval:
            item += self.total_len
        return self.data[item]

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
        # print(list(param.keys()), list(data.keys()))
        for i in range(n):
            for k in temp_data:
                if k in split_mark:
                    temp_data[k] = data[k][i:l:n]
            param['data'] = temp_data
            result.append(type(self)(**param))
        return result


class TextDataset(DatasetWithEval):

    def __init__(self, fp, tokenizer, total_len, valid_func=lambda x: True,
                 process_func=lambda x: x, max_len=None, truncate_mode='truncate', eval_size=512):
        super(TextDataset, self).__init__(tokenizer, total_len, eval_size=eval_size)
        if isinstance(fp, str):
            fp = open(fp, 'r')
        data_len = total_len + eval_size
        self.data = _read_block(fp, data_len, tokenizer, max_len=max_len, valid_func=valid_func,
                                process_func=process_func, truncate_mode=truncate_mode, add_eos=True)


class IdxDataset(DatasetWithEval):

    def __init__(self, tokenizer, idx_file=None, total_len=512, eval_size=512, data=None, **kwargs):
        super(IdxDataset, self).__init__(tokenizer, total_len, eval_size=eval_size)
        if idx_file is not None and data is None:
            self.idx_file = idx_file
            self.data = np.array(self.load_idx())
        else:
            self.data = data['idx']
            self.total_len = len(self.data) - self.eval_size

    def load_idx(self):
        result = []
        data_len = self.total_len + self.eval_size
        for i, line in enumerate(self.idx_file):
            result.append(tuple(map(lambda x: int(x), line.split())))
            if i >= data_len:
                break
        return result

    def get_loaded_length(self):
        return len(self.data)

    def __getitem__(self, item):
        idx = IdxDataset.__getitem__(self, item)
        return idx[0], idx[1], idx[2]


class IdxTextDataset(IdxDataset):

    def __init__(self, tokenizer, idx_file=None, sent_file=None, total_len=512, data=None, eval_size=0, **kwargs):
        super(IdxTextDataset, self).__init__(tokenizer, idx_file=idx_file, total_len=total_len, eval_size=eval_size,
                                             data=data, **kwargs)
        if None not in (idx_file, sent_file) and data is None:
            self.sent_file = sent_file
            self.sent_data = self.load_sent()
        else:
            self.sent_data = data['sent']

    def load_sent(self):
        result = {}
        sents = set(self.data[:, 2])
        for i in range(max(sents) + 1):
            sent = self.sent_file.readline()[:-1]
            if i in sents:
                result[i] = encode(self.tokenizer, sent, add_eos=True, add_prefix_space=True)
        return result

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

    def __init__(self, tokenizer, idx_file=None, ent_file=None, total_len=512, data=None, eval_size=0, **kwargs):
        super(IdxEntityDataset, self).__init__(tokenizer, idx_file=idx_file, total_len=total_len, eval_size=eval_size,
                                               data=data, **kwargs)
        if None not in (idx_file, ent_file) and data is None:
            self.ent_file = ent_file
            self.ent_data = self.load_ent()
        else:
            self.ent_data = data['ent']

    def load_ent(self):
        result = {}
        ents = set(self.data[:, [0, 1]].reshape(1, -1).squeeze())
        for i in range(max(ents) + 1):
            ent = self.ent_file.readline()[:-1]
            if i in ents:
                result[i] = ent
        return result

    def get_loaded_length(self):
        return IdxDataset.get_loaded_length(self), len(self.ent_data)

    def __getitem__(self, item):
        e1i, e2i, _ = IdxDataset.__getitem__(self, item)
        return self.ent_data[e1i], self.ent_data[e2i], (e1i, e2i)

    def get_data(self):
        result = {'sent': self.ent_data}
        result.update(IdxDataset.get_data(self))
        return result


class IdxFullDataset(IdxEntityDataset, IdxTextDataset):

    def __init__(self, tokenizer, idx_file=None, ent_file=None, sent_file=None, total_len=512, data=None, eval_size=0,
                 **kwargs):
        super(IdxFullDataset, self).__init__(tokenizer, total_len=total_len, eval_size=eval_size, idx_file=idx_file,
                                             ent_file=ent_file, sent_file=sent_file, data=data, **kwargs)
        print(self.data)
        print(self.ent_data)
        print(self.sent_data)
        print(self.get_loaded_length())

    def __getitem__(self, item):
        e1, e2, eidx = IdxEntityDataset.__getitem__(self, item)
        sent, senti = IdxTextDataset.__getitem__(self, item)
        e1, e2 = self.unify_entity(e1, sent, senti), self.unify_entity(e2, sent, senti)
        return e1, e2, sent, (*eidx, senti)

    def get_loaded_length(self):
        return IdxDataset.get_loaded_length(self), len(self.sent_data), len(self.ent_data)

    def unify_entity(self, ent, sent, sent_idx):
        def in_tensor(ent, sent_tok, idx):
            leng = 0
            for k in range(idx, len(sent_tok)):
                temp = sent_tok[k].replace(space, '')
                l = len(temp)
                if temp == ent[leng:leng + l]:
                    leng += l
                elif ent[leng] == space and temp == ent[leng + 1:leng + l + 1]:
                    leng += (1 + l)
                else:
                    if ent[leng] == space:
                        leng += 1
                    i = 0
                    while leng < len(ent):
                        if ent[leng] != temp[i]:
                            return None
                        leng += 1
                        i += 1
                    p1 = temp[:i]
                    p2 = temp[i:]
                    sent_tok[k] = p1
                    sent_tok.insert(k + 1, p2)
                    return sent_tok[idx:k + 2]
                if leng == len(ent):
                    return sent_tok[idx:k + 1]

        space = 'Ġ'
        sent_tok = self.tokenizer.convert_ids_to_tokens(sent)
        ent_temp = ''.join(self.tokenizer.tokenize(ent))
        # print(ent_temp)
        for i in range(len(sent_tok)):
            temp = sent_tok[i].replace(space, '')
            if ent_temp.startswith(temp):
                ent_tok = in_tensor(ent_temp, sent_tok, i)
                if ent_tok is not None:
                    # print(True)
                    # print(ent_tok, sent_tok)
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
