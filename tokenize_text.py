import sentencepiece as spm
import json
# spm.SentencePieceTrainer.train('--input=data/train.jsonl --model_prefix=data/tokenizer --vocab_size=10000')

sp = spm.SentencePieceProcessor()
sp.load('data/tokenizer.model')
with open('data/train.jsonl') as f:
    data = [json.loads(line) for line in f]

ids = sp.encode_as_ids(data[0]["prompt"])
print(sp.pad_id())