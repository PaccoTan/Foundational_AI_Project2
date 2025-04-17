import os
import random
import re
import matplotlib.pyplot as plt
import torch
import sentencepiece as spm
import json
from torcheval.metrics import Perplexity, BLEUScore
import models

path = "final_models/rnn-2"
tokenizer_path = 'data/tokenizer.model'
with open(path + "/params.json", 'r') as f:
    params = json.load(f)
print(params)
device = "cuda" if torch.cuda.is_available() else "cpu"
if "lstm" in path:
    model = models.LSTM(**params)
elif "transformer" in path:
    model = models.Transformer(**params)
elif "rnn" in path:
    model = models.RNN(**params)
else:
    print("model type has to be in path")
    exit()
model.load_state_dict(torch.load(path + "/model.pt"))
model.to(device)
sp = spm.SentencePieceProcessor()
sp.load(tokenizer_path)

with open(path + "/config.txt", "r") as file:
    log = file.read()
train_losses = re.findall(r'Train Loss:\s*([\d.]+)',log)
val_losses = re.findall(r'Val Loss:\s*([\d.]+)',log)
train_losses = list(map(float, train_losses))
val_losses = list(map(float, val_losses))
plt.plot(train_losses, label="Training")
plt.plot(val_losses, label="Validation")
plt.title("Loss vs Epoch")
plt.legend()
plt.show()

metric = Perplexity()
metric.to(device)
metric2 = BLEUScore(n_gram=1)
metric3 = BLEUScore(n_gram=2)
metric4 = BLEUScore(n_gram=3)
metric5 = BLEUScore(n_gram=4)
preds = []
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
        out = sp.decode_ids(ref.tolist())
        if out.count(" ") > 0:
            metric3.update([out],[sp.decode_ids(d[1:])])
        if out.count(" ") > 1:
            metric4.update([out],[sp.decode_ids(d[1:])])
        if out.count(" ") > 2:
            metric5.update([out], [sp.decode_ids(d[1:])])
        metric2.update([out],[sp.decode_ids(d[1:])])
    print("Perplexity:", metric.compute())
    print("BLEUScore n_gram=1:", metric2.compute())
    print("BLEUScore n_gram=2:", metric3.compute())
    print("BLEUScore n_gram=3:", metric4.compute())
    print("BLEUScore n_gram=4:", metric5.compute())

model = models.LanguageModel(model, sp)
prompt = "Which do you prefer? Dogs or cats?"
print(prompt + ":")
print("Multinomial")
print(model.prompt(prompt,50,temperature=0.7))
print("Top-K")
print(model.prompt(prompt,50,temperature=0.7,sampling_method="top-k",k=30))
print("Top-P")
print(model.prompt(prompt,50,temperature=0.7,sampling_method="top-p",p=0.7))

prompt = "Hi Winnie the Pooh "
print(prompt + ":")
print("Multinomial")
print(model.prompt(prompt,50,temperature=0.7))
print("Top-K")
print(model.prompt(prompt,50,temperature=0.7,sampling_method="top-k",k=30))
print("Top-P")
print(model.prompt(prompt,50,temperature=0.7,sampling_method="top-p",p=0.7))