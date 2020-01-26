import numpy as np
import transformers as tfm
from torch.utils.data import Dataset

from codes.util import encode


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
    def __init__(self, tokenizer, total_len, eval_size=0):
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

    def split(self, partitions, eval_size=0):
        n = partitions
        l = len(self.data)
        param = dict(vars(self))
        param['eval_size'] = self.eval_size
        result = []
        for i in range(n):
            temp_data = self.data[i:l:n]
            param.update({'data': temp_data})
            result.append(type(self)(**param))


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

    def __init__(self, tokenizer, idx_file=None, ent_file=None, sent_file=None, total_len=512, data=None,
                 eval_size=512):
        super(IdxDataset, self).__init__(tokenizer, total_len, eval_size=eval_size)
        if None not in (idx_file, ent_file, sent_file):
            self.idx_file = idx_file
            self.ent_file = ent_file
            self.sent_file = sent_file
            i = 0
            line = idx_file.readline()
            data_len = total_len + eval_size
            while line and i < data_len:
                self.data.append(tuple(map(lambda x: int(x), line.split())))
                line = idx_file.readline()
                i += 1
            self.data = np.array(self.data)
            self.sent_data = {}
            self.ent_data = {}
            self._read_block()
        else:
            self.ent_data = data['ent']
            self.sent_data = data['sent']
            self.data = data['idx']
            data_len = len(self.data)
            self.total_len = data_len - eval_size

    def __getitem__(self, item):
        idx = super(IdxDataset, self).__getitem__(item)
        e1i, e2i, senti = idx[0], idx[1], idx[2]
        e1, e2, sent = self.ent_data[e1i], self.ent_data[e2i], self.sent_data[senti]
        e1, e2 = self.unify_entity(e1, sent), self.unify_entity(e2, sent)
        return e1, e2, sent, idx

    def _read_block(self):
        ents = np.sort(np.unique(self.data[:, [0, 1]]))
        sents = np.sort(np.unique(self.data[:, 2]))
        idx = 0
        for i in range(np.max(ents) + 1):
            ent = self.ent_file.readline()[:-1]
            if ents[idx] == i:
                self.ent_data[i] = ent
                idx += 1
        idx = 0
        for i in range(np.max(sents) + 1):
            sent = self.sent_file.readline()[:-1]
            if sents[idx] == i:
                self.sent_data[i] = encode(self.tokenizer, sent, add_eos=True, add_prefix_space=True)

    def get_total_ent_sent(self):
        return len(self.ent_data), len(self.sent_data)

    def unify_entity(self, ent, sent):
        def in_tensor(ent, sent_tok, idx):
            tot = ''
            for k in range(idx, len(sent_tok)):
                temp = sent_tok[k].replace(space, '')
                tot = (tot + temp) if ent.startswith(tot + temp) else (
                    (tot + space + temp) if ent.startswith(tot + space + temp) else None)
                if tot is None:
                    return None
                elif tot == ent:
                    return sent_tok[idx: k + 1]

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
                    return encode(self.tokenizer, ent_tok)
        return encode(self.tokenizer, ent)

    def split(self, partitions, eval_size=0):
        tok = self.tokenizer
        n = partitions
        l = len(self.data)
        return [IdxDataset(tok, data={'idx': self.data[i:l:n], 'ent': self.ent_data, 'sent': self.sent_data},
                           eval_size=eval_size) for i in range(n)]


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
