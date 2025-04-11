import sentencepiece as spm
import json

with open('data/train.jsonl') as f:
    data = [json.loads(line) for line in f]

print(data[0])
with open("data/train.txt","w") as f:
    for d in data:
        f.write(d["prompt"] + " " + d["completion"] + "\n")
spm.SentencePieceTrainer.train('--input=data/train.txt --model_prefix=data/tokenizer --vocab_size=10000')

sp = spm.SentencePieceProcessor()
sp.load('data/tokenizer.model')
with open('data/train.jsonl') as f:
    data = [json.loads(line) for line in f]

ids = sp.encode_as_ids(data[0]["prompt"])
print(ids)