import time
import matplotlib.pyplot as plt
import torch
import sentencepiece as spm
import json
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
import models
from torcheval.metrics import Perplexity, BLEUScore

sp = spm.SentencePieceProcessor()
sp.load('data/tokenizer.model')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('data/train.jsonl') as f:
    data = [json.loads(line) for line in f]

epochs = 30
batch_size = 8
idxs = torch.randperm(len(data))
train_data = [data[idxs[i]] for i in range(math.ceil(len(data)*0.8))]
val_data = [data[idxs[i]] for i in range(math.ceil(len(data)*0.8),len(data))]
print(len(data),len(train_data),len(val_data))

params = {
    "input_size": 128,
    "hidden_size": 256,
    "num_layers": 4,
    "vocab_size": 10000,
    "dropout": 0.2,
}
print(params)
model = models.LSTM(**params)
model.to(device)
loss_func = torch.nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
train_loss = []
val_loss = []
for epoch in range(epochs):
    idxs = torch.randperm(len(train_data))
    model.train()
    total = 0
    start = time.time()
    optimizer.zero_grad()
    for batch_idx in range(math.ceil(len(train_data)/batch_size)):
        # optimizer.zero_grad()
        batch = idxs[batch_idx*batch_size:min(batch_idx*batch_size+batch_size,len(train_data))]
        # x is entire prompt + completion
        # y is x shifted to the right by one token with an end token(2)
        sentences = [sp.encode_as_ids(train_data[idx]["prompt"],enable_sampling=True, alpha=0.1) +
                sp.encode_as_ids(train_data[idx]["completion"],enable_sampling=True, alpha=0.1) for idx in batch]
        x = [torch.Tensor(sentence[:-1]).int() for sentence in sentences]
        y = [torch.Tensor(sentence[1:]).int() for sentence in sentences]
        lens = [len(b) for b in x]
        mask = pad_sequence([torch.ones(len(b)) for b in x],batch_first=True,padding_value=0).to(device)
        x = pad_sequence(x,batch_first=True,padding_value = 9999).to(device)
        y = pad_sequence(y, batch_first=True, padding_value=0).to(device)
        out = model(x)
        loss = ((loss_func(out.permute(0,-1,-2),y.long())) * mask).mean()
        total += loss.item()
        loss.backward()
        if(batch_idx % 16 == 0 or batch_idx == math.ceil(len(train_data)/batch_size)):
            optimizer.step()
            optimizer.zero_grad()
    # scheduler.step()
    model.eval()
    idxs = torch.randperm(len(val_data))
    val_total = 0
    with torch.no_grad():
        for batch_idx in range(math.ceil(len(val_data)/batch_size)):
            batch = idxs[batch_idx*batch_size:min(batch_idx*batch_size+batch_size,len(val_data))]
            sentences = [sp.encode_as_ids(val_data[idx]["prompt"]) + sp.encode_as_ids(val_data[idx]["completion"]) for idx in batch]
            x = [torch.Tensor(sentence[:-1]).int() for sentence in sentences]
            y = [torch.Tensor(sentence[1:]).int() for sentence in sentences]
            lens = [len(b) for b in x]
            mask = pad_sequence([torch.ones(len(b)) for b in x], batch_first=True, padding_value=0).to(device)
            x = pad_sequence(x, batch_first=True, padding_value=9999).to(device)
            y = pad_sequence(y, batch_first=True, padding_value=0).to(device)
            out = model(x)
            loss = ((loss_func(out.permute(0, -1, -2), y.long())) * mask).mean()
            val_total += loss.item()
        prompt = sp.encode_as_ids("<bos>Which do you prefer? Dogs or cats?")
        opt_response = []
        response = []
        for i in range(50):
            x = torch.Tensor(prompt).int().to(device)
            x = model(x)
            y = (x / 0.5).softmax(-1)
            predicted_token = torch.multinomial(y[-1], 1).item()
            if predicted_token == 2:
                break
            prompt.append(predicted_token)
            response.append(predicted_token)
    print(sp.decode_ids(prompt))
    print(sp.decode_ids(response))
    train_loss.append(total / math.ceil(len(train_data) / batch_size))
    val_loss.append(val_total / math.ceil(len(val_data) / batch_size))
    print("Epoch %d:\tTrain Loss: %.3f\tVal Loss: %.3f\tTime Taken: %.3f" % (epoch + 1, train_loss[epoch], val_loss[epoch], time.time() - start))

plt.plot(train_loss,label='Training')
plt.plot(val_loss,label='Validation')
plt.show()

metric = Perplexity()
metric.to(device)
metric2 = BLEUScore(n_gram=4)

with torch.no_grad():
    model.eval()
    with open('data/test.jsonl') as f:
        t_data = [json.loads(line) for line in f]
        t_data = [sp.encode_as_ids(d["prompt"]) + sp.encode_as_ids(d["completion"]) for d in data]
    for d in t_data:
        x = torch.Tensor(d[:-1]).int().to(device)
        y = torch.Tensor(d[1:]).int().to(device)
        x = model(x)
        x = x.softmax(-1)
        metric.update(x.unsqueeze(0),y.unsqueeze(0))
        ref = x.argmax(-1)
        metric2.update([sp.decode_ids(ref.tolist())],[sp.decode_ids(d[1:])])
    print("Perplexity:", metric.compute())
    print("BLEU-Score:", metric2.compute())
