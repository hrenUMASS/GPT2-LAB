import logging

import torch
import transformers as tfm

from libs import IdxFullDataset

logging.getLogger('transformers.tokenization_utils').disabled = True

samples = 32 * 62500

agg_data = '/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/data/raw/wiki2016_nchunk_entity_agg/'
data_path = '/iesl/canvas/hren/gpt2_wiki_lab/data/'
model_path = '/iesl/canvas/hren/gpt2_wiki_lab/models/gpt2_pretrained/small/'
save_path = data_path + 'entity_vocab.pt'

idx_path = data_path + 'wiki2016_idx'
ent_path = agg_data + 'wiki2016_ent'
sent_path = data_path + 'wiki2016_sents_mapped'

idx_file = open(idx_path, 'r')
ent_file = open(ent_path, 'r')
sent_file = open(sent_path, 'r')

tokenizer = tfm.GPT2Tokenizer.from_pretrained(model_path)
dataset = IdxFullDataset(tokenizer, idx_file=idx_file, ent_file=ent_file, sent_file=sent_file,
                         total_len=samples, eval_size=5000)
idx_file.close()
ent_file.close()
sent_file.close()

result = {}

for data in dataset:
    e1, e2 = data[0], data[1]
    e1i, e2i = data[3][0], data[3][1]
    result[e1i] = e1
    result[e2i] = e2

print(result)
torch.save(result, save_path)
