import sentencepiece as spm
import json

with open('data/train.jsonl') as f:
    data = [json.loads(line) for line in f]

with open("data/train.txt","w") as f:
    for d in data:
        f.write(d["prompt"] + " " + d["completion"] + "\n")
spm.SentencePieceTrainer.train('--input=data/train.txt --user_defined_symbols=<bos>,<eos> --model_prefix=data/tokenizer --vocab_size=10000 --model_type=bpe --character_coverage=1 --pad_id=9999 --bos_piece=<bos> --eos_piece=<eos>')

sp = spm.SentencePieceProcessor()
sp.load('data/tokenizer.model')
with open('data/test.jsonl') as f:
    data = [json.loads(line) for line in f]
    data = [(sp.encode_as_ids(json_object["prompt"]),sp.encode_as_ids(json_object["completion"])) for json_object in data]
    lens = [len(x) + len(y) for x, y in data]
print(data[(lens).index(min(lens))],(lens).index(min(lens)))

count = 0
for x,y in data:
    if(0 in x or 0 in y):
        count += 1
print(count)

for i in range(5):
    ids = sp.encode("<bos>Which do you prefer? Dogs or cats?")
    print(ids)

print(sp.decode_ids(ids))