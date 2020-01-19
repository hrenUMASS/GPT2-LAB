import numpy as np
import transformers as tfm
from torch.utils.data import Dataset

from util import get_index


def _read_block(fp, total_len, tokenizer: tfm.PreTrainedTokenizer, max_len=None, valid_func=lambda x: True,
                process_func=lambda x: x, truncate_mode='truncate', max_step=np.inf, add_eos=False):
    result = []
    i, k = 0, 0
    raw = fp.readline()
    while raw != '' and i < total_len and k < max_step:
        line = process_func(raw)
        line = tokenizer.encode(line, add_prefix_space=True)
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
    return tokenizer.encode(ent, add_prefix_space=True, return_tensors='pt'), tokenizer.encode(ent, return_tensors='pt')


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

    def __init__(self, tokenizer, idx_file, ent_file, sent_file, total_len, max_len=np.inf, ent_past_index=0,
                 sent_past_index=0):
        self.idx_file = idx_file
        self.ent_file = ent_file
        self.sent_file = sent_file
        self.idx_data = []
        i = 0
        line = idx_file.readline()
        while line and i < total_len:
            self.idx_data.append(tuple(map(lambda x: int(x), line.split())))
            line = idx_file.readline()
            i += 1
        self.idx_data = np.array(self.idx_data)
        self.sent_data = []
        self.ent_data = []
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.enti = ent_past_index
        self.senti = sent_past_index

        self._read_block()

    def __len__(self):
        return len(self.idx_data)

    def __getitem__(self, item):
        idx = self.idx_data[item]
        e1i, e2i, senti = idx[0] - self.enti, idx[1] - self.enti, idx[2] - self.senti
        e1, e2, sent = self.ent_data[e1i], self.ent_data[e2i], self.sent_data[senti]
        e1tp, e1t = get_tokens(self.tokenizer, e1)
        e2tp, e2t = get_tokens(self.tokenizer, e2)
        e1 = e1tp if get_index(sent, e1tp) != -1 else e1t
        e2 = e2tp if get_index(sent, e2tp) != -1 else e2t
        return e1, e2, sent, idx

    def _read_block(self):

        max_ent = np.max(self.idx_data[:, [0, 1]]) - self.enti
        max_sent = np.max(self.idx_data[:, 2]) - self.senti
        self.ent_data = [self.ent_file.readline() for _ in (max_ent + 1)]
        for _ in range(max_sent):
            self.sent_data.append(self.tokenizer.encode(self.sent_file.readline()))

    def get_total_ent_sent(self):
        return len(self.ent_data), len(self.sent_data)


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
