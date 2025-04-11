import torch
import sentencepiece as spm
import json
import math

from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence

import models

sp = spm.SentencePieceProcessor()
sp.load('data/tokenizer.model')

with open('data/train.jsonl') as f:
    data = [json.loads(line) for line in f]
    data = [(sp.encode_as_ids(json_object["prompt"]),sp.encode_as_ids(json_object["completion"])) for json_object in data]

model = models.RNN(128,256,4,vocab_size=10001)
epochs = 30
batch_size = 128

for epoch in range(epochs):
    idxs = torch.randperm(len(data))
    for batch_idx in range(math.ceil(len(data)/batch_size)):
        batch = idxs[batch_idx*batch_size:min(batch_idx*batch_size+batch_size,len(data))]
        print("data_idx",batch[0])
        print(data[batch[0]])
        batch = [torch.Tensor(data[idx][0] + data[idx][1] + [2]).int() for idx in batch]
        lens = [len(b) for b in batch]
        print(lens[0])
        batch = pad_sequence(batch,batch_first=True,padding_value = 10000)
        batch = pack_padded_sequence(batch,lens,batch_first=True,enforce_sorted=False)
        batch,_ = pad_packed_sequence(batch, batch_first=True)
        print(batch[0,:lens[0]])
        print(sp.decode_ids(batch[0,:lens[0]].tolist()))
        # C dimension is in the middle or need to do some reshaping (N,C)

        break
    break