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


class BlockTextDataset(Dataset):

    def __init__(self, file_path, tokenizer, total_len, block_size=512, valid_func=lambda x: True,
                 process_func=lambda x: x, max_len=None, truncate_mode='truncate'):
        self.valid_func = valid_func
        self.process_func = process_func
        self.block_size = block_size
        self.take_count = 1
        self.total_len = total_len
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.truncate_mode = truncate_mode
        self.file = open(file_path, 'r')
        self.data = _read_block(self.file, self.block_size, self.tokenizer, max_len=self.max_len,
                                valid_func=self.valid_func, process_func=self.process_func,
                                truncate_mode=self.truncate_mode)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        result = self.data[item % self.block_size]
        self.take_count += 1
        if self.take_count > self.block_size:
            self.take_count = 1
            self.data = _read_block(self.file, self.block_size, self.tokenizer, max_len=self.max_len,
                                    valid_func=self.valid_func, process_func=self.process_func,
                                    truncate_mode=self.truncate_mode)
        return result


class TextDataset(Dataset):

    def __init__(self, fp, tokenizer, total_len, valid_func=lambda x: True,
                 process_func=lambda x: x, max_len=None, truncate_mode='truncate', add_eos=False):
        if isinstance(fp, str):
            fp = open(fp, 'r')
        self.data = _read_block(fp, total_len, tokenizer, max_len=max_len, valid_func=valid_func,
                                process_func=process_func, truncate_mode=truncate_mode, add_eos=add_eos)
        # self.data = np.array(self.data, dtype=np.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class IdxDataset(Dataset):

    def __init__(self, tokenizer, idx_file, ent_file, sent_file, total_len):
        self.idx_file = idx_file
        self.ent_file = ent_file
        self.sent_file = sent_file
        self.idx_data = []
        i = 0
        line = idx_file.readline()
        while line and i < total_len * 2:
            self.idx_data.append(tuple(map(lambda x: int(x), line.split())))
            line = idx_file.readline()
            i += 1
        self.idx_data = np.array(self.idx_data)
        self.sent_data = []
        self.ent_data = []
        self.tokenizer = tokenizer
        self.eval = False
        self.total_len = total_len
        # self.space_encoder = IdxDataset._generate_space_vocab(tokenizer.encoder)
        # self.space_decoder = {v: k for k, v in self.space_encoder.items()}

        self._read_block()

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        if self.eval:
            item += self.total_len
        idx = self.idx_data[item]
        # print('idx', idx, item, idx[0] - self.enti, idx[1] - self.enti, idx[2] - self.senti)
        e1i, e2i, senti = idx[0], idx[1], idx[2]
        # print(e1i, e2i, senti)
        # print('e1', e1i, idx[0])
        # print('e2', e2i, idx[1])
        # print('sent', senti, idx[2])
        e1, e2, sent = self.ent_data[e1i], self.ent_data[e2i], self.sent_data[senti]
        e1, e2 = self.unify_entity(e1, sent), self.unify_entity(e2, sent)
        # print(e1, e2, sent)
        return e1, e2, sent, idx

    def _read_block(self):
        max_ent = np.max(self.idx_data[:, [0, 1]])
        max_sent = np.max(self.idx_data[:, 2])
        # print(np.max(self.idx_data[:, [0, 1]]), self.enti, max_ent)
        # print(np.max(self.idx_data[:, 2]), self.senti, max_sent)
        self.ent_data = [self.ent_file.readline()[:-1] for _ in range(max_ent + 1)]
        # print(self.idx_data)
        # print(list(enumerate(self.ent_data)))
        pos = self.sent_file.tell()
        # print(list(enumerate([self.sent_file.readline()[:-1] for _ in range(max_sent + 1)])))
        self.sent_file.seek(pos)
        for _ in range(max_sent + 1):
            self.sent_data.append(
                encode(self.tokenizer, self.sent_file.readline()[:-1], add_eos=True, add_prefix_space=True))

    def get_total_ent_sent(self):
        return len(self.ent_data), len(self.sent_data)

    def unify_entity(self, ent, sent):
        def in_tensor(ent, sent_tok, idx):
            tot = ''
            for k in range(idx, len(sent_tok)):
                temp = sent_tok[k].replace(space, '')
                tot = (tot + temp) if ent.startswith(tot + temp) else (
                    (tot + space + temp) if ent.startswith(tot + space + temp) else None)
                # print(temp, tot)
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

    def change_mode(self, evaluate=True):
        self.eval = evaluate
        return self


if __name__ == '__main__':
    import transformers as tfm
    from torch.utils.data import DataLoader
    from gpt2_train import get_tensor_batch

    tok = tfm.GPT2Tokenizer.from_pretrained('../gpt2_pretrained')
    a = TextDataset('../../data/wiki2016_sents', tok, 16, max_len=512)
    b = DataLoader(a, batch_size=4, collate_fn=lambda x: x)
    for i in b:
        # print(i)
        # print(get_tensor_batch(i))
        print(get_tensor_batch(i))
