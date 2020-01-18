import numpy as np
import torch
from torch.utils.data import Dataset


def _read_block(fp, total_len, tokenizer, max_len=None, valid_func=lambda x: True, process_func=lambda x: x,
                truncate_mode='truncate', max_step=np.inf):
    result = []
    i, k = 0, 0

    raw = fp.readline()
    while raw != '' and i < total_len and k < max_step:
        line = process_func(raw)
        line = tokenizer.encode(line, return_tensors='pt')[0].clone().detach().type(torch.long)
        if valid_func(line):
            if max_len is None or line.shape[0] <= max_len:
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
                    while k < line.shape[0]:
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

    def __init__(self, file_path, tokenizer, total_len, valid_func=lambda x: True,
                 process_func=lambda x: x, max_len=None, truncate_mode='truncate', with_index=False):
        with open(file_path, 'r') as fp:
            self.data = _read_block(fp, total_len, tokenizer, max_len=max_len, valid_func=valid_func,
                                    process_func=process_func, truncate_mode=truncate_mode)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class IdxDataset(Dataset):

    def __init__(self, idx_file, total_len):
        self.data = []
        i = 0
        self.pos = idx_file.tell()
        line = idx_file.readline()
        while line and i < total_len:
            self.data.append(tuple(map(lambda x: int(x), line.split())))
            line = idx_file.readline()
            i += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


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
