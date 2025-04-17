import random
import time
import matplotlib.pyplot as plt
import torch
import sentencepiece as spm
import json
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
import models
from torcheval.metrics import Perplexity, BLEUScore
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path="models"
model_path = "/run5-transformer"
params = {
    "input_size": 128,
    "hidden_size": 512,
    "n_head": 8,
    "num_layers": 6,
    "vocab_size": 10000,
    "dropout": 0.3,
}
print(params)
model = models.Transformer(**params)
model.to(device)

os.makedirs(path + model_path, exist_ok=True)
os.makedirs(path + model_path + '/checkpoints', exist_ok=True)
epochs = 30
batch_size = 16
lr = 0.001
with open(path + model_path + '/params.json','w') as f:
  f.write(json.dumps(params))
with open(path + model_path + '/config.txt','w') as f:
  f.write("epochs: %d\n" % epochs)
  f.write("batch_size: %d\n" % batch_size)
  f.write("lr: %f\n" % lr)

sp = spm.SentencePieceProcessor()
sp.load('data/tokenizer.model')

with open('data/train.jsonl') as f:
    data = [json.loads(line) for line in f]

idxs = torch.randperm(len(data))
train_data = [data[idxs[i]] for i in range(math.ceil(len(data)*0.8))]
val_data = [data[idxs[i]] for i in range(math.ceil(len(data)*0.8),len(data))]
print(len(data),len(train_data),len(val_data))


loss_func = torch.nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.AdamW(model.parameters(), lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.9)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.5, verbose=True)
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
        prompts = [sp.encode_as_ids(train_data[idx]["prompt"],enable_sampling=True, alpha=0.1) for idx in batch]
        completions = [sp.encode_as_ids(train_data[idx]["completion"],enable_sampling=True, alpha=0.1) for idx in batch]
        sentences = [prompt + completion for prompt,completion in zip(prompts,completions)]
        x = [torch.Tensor(sentence[:-1]).int() for sentence in sentences]
        y = [torch.Tensor(sentence[1:]).int() for sentence in sentences]
        mask = pad_sequence([torch.ones(len(b)) for b in x],batch_first=True,padding_value=0).to(device)
        x = pad_sequence(x,batch_first=True,padding_value = 9999).to(device)
        y = pad_sequence(y, batch_first=True, padding_value=0).to(device)
        pad_mask = pad_sequence([torch.zeros(len(b)) for b in x],batch_first=True,padding_value=float('-Inf')).to(device)
        out = model(x,pad_mask)
        loss = ((loss_func(out.permute(0,-1,-2),y.long())) * mask).sum() / mask.sum()
        total += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
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
            out = model(x,mask==0)
            loss = ((loss_func(out.permute(0, -1, -2), y.long())) * mask).sum() / mask.sum()
            val_total += loss.item()
        prompt = sp.encode_as_ids("<bos>Which do you prefer? Dogs or cats?")
        opt_response = []
        response = []
        for i in range(50):
            x = torch.Tensor(prompt).int().to(device).unsqueeze(0)
            x = model(x)
            y = (x / 0.7).softmax(-1).squeeze(0)
            predicted_token = torch.multinomial(y[-1], 1).item()
            prompt.append(predicted_token)
            response.append(predicted_token)
            if predicted_token == 2:
                break
    print(sp.decode_ids(prompt))
    print(response)
    train_loss.append(total / math.ceil(len(train_data) / batch_size))
    val_loss.append(val_total / math.ceil(len(val_data) / batch_size))
    scheduler.step()
    print("Epoch %d:\tTrain Loss: %.3f\tVal Loss: %.3f\tTime Taken: %.3f" % (epoch + 1, train_loss[epoch], val_loss[epoch], time.time() - start))
    with open(path + model_path + '/config.txt', 'a') as f:
        f.write("Epoch %d:\tTrain Loss: %.3f\tVal Loss: %.3f\tTime Taken: %.3f\n" % (
        epoch + 1, train_loss[epoch], val_loss[epoch], time.time() - start))
    torch.save(model.state_dict(), path + model_path + '/checkpoints/model' + str(epoch + 1) + '.pt')

plt.plot(train_loss,label='Training')
plt.plot(val_loss,label='Validation')
plt.show()

metric = Perplexity()
metric.to(device)
metric2 = BLEUScore(n_gram=1)
metric3 = BLEUScore(n_gram=2)

with torch.no_grad():
    model.eval()
    with open('data/test.jsonl') as f:
        t_data = [json.loads(line) for line in f]
        len_prompt = [len(sp.encode_as_ids(d["prompt"])) for d in t_data]
        t_data = [sp.encode_as_ids(d["prompt"]) + sp.encode_as_ids(d["completion"]) for d in t_data]
    for d,l in zip(t_data,len_prompt):
        x = torch.Tensor(d[:-1]).int().to(device).unsqueeze(0)
        y = torch.Tensor(d[l:]).int().to(device)
        x = model(x)
        metric.update(x[:,l-1:],y.unsqueeze(0))
        ref = x[0].argmax(-1)
        metric3.update([sp.decode_ids(ref.tolist())],[sp.decode_ids(d[1:])])
        metric2.update([sp.decode_ids(ref.tolist())],[sp.decode_ids(d[1:])])
    print("Perplexity:", metric.compute())
    print("BLEUScore n_gram=1:", metric2.compute())
    print("BLEUScore n_gram=2:", metric3.compute())
