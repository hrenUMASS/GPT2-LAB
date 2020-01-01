from torch.utils.data import Dataset


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
        self.file_pos = 0
        self.data = self._read_block()

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        result = self.data[item % self.block_size]
        self.take_count += 1
        if self.take_count > self.block_size:
            self.take_count = 1
            self.data = self._read_block()
        return result

    def _read_block(self):
        result = []
        i, k = 0, 0
        max_step = 1024
        while i < self.block_size and k < max_step:
            line = self.process_func(self.file.readline())
            line = self.tokenizer.encode(line, return_tensors='pt')[0]
            if self.valid_func(line):
                if self.max_len is None or line.shape[0] <= self.max_len:
                    result.append(line)
                    # print('normal', result[-1].shape)
                    i += 1
                else:
                    if self.truncate_mode == 'truncate':
                        result.append(line[:self.max_len])
                        # print('truncate', result[-1].shape)
                        i += 1
                    elif self.truncate_mode == 'append':
                        k = 0
                        while k < line.shape[0]:
                            result.append(line[k:k + self.max_len])
                            # print('append', result[-1].shape)
                            k += self.max_len
                            i += 1
                    elif self.truncate_mode == 'discard':
                        pass
                    else:
                        raise ValueError('No such truncate mode {}'.format(self.truncate_mode))
            k += 1
        # print(list(map(lambda x: x.shape, result)))
        return result

    def turn(self):
        self.file_pos = self.file.seek(0)
