import time

import torch
import sentencepiece as spm
import json
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
import models

sp = spm.SentencePieceProcessor()
sp.load('data/tokenizer.model')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('data/train.jsonl') as f:
    data = [json.loads(line) for line in f]
    data = [(sp.encode_as_ids(json_object["prompt"]),sp.encode_as_ids(json_object["completion"])) for json_object in data]

epochs = 30
batch_size = 8
idxs = torch.randperm(len(data))
train_data = [data[idxs[i]] for i in range(math.ceil(len(data)*0.8))]
val_data = [data[idxs[i]] for i in range(math.ceil(len(data)*0.8),len(data))]
print(len(data),len(train_data),len(val_data))

model = models.LSTM(128,256,4,vocab_size=10000,dropout=0.2)
model.to(device)
loss_func = torch.nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
for epoch in range(epochs):
    idxs = torch.randperm(len(train_data))
    model.train()
    total = 0
    start = time.time()
    for batch_idx in range(math.ceil(len(train_data)/batch_size)):
        optimizer.zero_grad()
        batch = idxs[batch_idx*batch_size:min(batch_idx*batch_size+batch_size,len(train_data))]
        # x is entire prompt + completion
        # y is x shifted to the right by one token with an end token(2)
        x = [torch.Tensor(train_data[idx][0] + train_data[idx][1]).int() for idx in batch]
        y = [torch.Tensor(train_data[idx][0][1:] + train_data[idx][1] + [2]).int() for idx in batch]
        lens = [len(b) for b in x]
        mask = pad_sequence([torch.ones(len(b)) for b in x],batch_first=True,padding_value=0).to(device)
        x = pad_sequence(x,batch_first=True,padding_value = 10000).to(device)
        y = pad_sequence(y, batch_first=True, padding_value=0).to(device)
        out = model(x)
        loss = ((loss_func(out.permute(0,-1,-2),y.long())) * mask).mean()
        total += loss.item()
        loss.backward()
        optimizer.step()
    scheduler.step()
    model.eval()
    idxs = torch.randperm(len(val_data))
    val_total = 0
    with torch.no_grad():
        for batch_idx in range(math.ceil(len(val_data)/batch_size)):
            batch = idxs[batch_idx*batch_size:min(batch_idx*batch_size+batch_size,len(val_data))]
            x = [torch.Tensor(val_data[idx][0] + val_data[idx][1]).int() for idx in batch]
            y = [torch.Tensor(val_data[idx][0][1:] + val_data[idx][1] + [2]).int() for idx in batch]
            lens = [len(b) for b in x]
            mask = pad_sequence([torch.ones(len(b)) for b in x], batch_first=True, padding_value=0).to(device)
            x = pad_sequence(x, batch_first=True, padding_value=10000).to(device)
            y = pad_sequence(y, batch_first=True, padding_value=0).to(device)
            out = model(x)
            loss = ((loss_func(out.permute(0, -1, -2), y.long())) * mask).mean()
            val_total += loss.item()

        prompt = sp.encode_as_ids("which do you prefer? dogs or cats?")
        for i in range(50):
            x = torch.Tensor(prompt).int().to(device)
            x = model(x)
            y = (x / 1).softmax(-1)
            predicted_token = torch.multinomial(y[-1], 1).item()
            if predicted_token == 2:
                break
            prompt.append(predicted_token)
    print(sp.decode_ids(prompt))

    print("Epoch %d:\tTrain Loss: %.3f\tVal Loss: %.3f\tTime Taken: %.3f" % (epoch+1,total/math.ceil(len(train_data)/batch_size),val_total/math.ceil(len(val_data)/batch_size),time.time()-start))