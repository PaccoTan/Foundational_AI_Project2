import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import sentencepiece as spm
from sympy.physics.units import temperature


class LanguageModel():
    def __init__(self, model: nn.Module, tokenizer: spm.SentencePieceProcessor):
        self.model = model
        self.tokenizer = tokenizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
    def forward(self, x, temperature = 1):
        prompt = self.tokenizer.encode_as_ids(x)
        x = torch.tensor(prompt).int().to(self.device).unsqueeze(0)
        logits = self.model(x)
        y = (logits/temperature).softmax(-1).squeeze(0)
        predicted_token = torch.multinomial(y[-1],1).item()
        return predicted_token

    def prompt(self, sentence, max_seq_length, sampling_method="temperature",k=10,p=0.5, temperature=1):
        prompt = self.tokenizer.encode_as_ids(sentence)
        response = []
        self.model.eval()
        for i in range(max_seq_length):
            x = torch.tensor(prompt + response).int().to(self.device).unsqueeze(0)
            x = self.model(x)
            if sampling_method == "top-p":
                vals, tokens = torch.topk(x[0, -1], x.shape[-1])
                sum = 0
                max_idx = 0
                for idx,val in enumerate(vals.softmax(-1)):
                    sum += val
                    if sum > p:
                        max_idx = idx
                        break
                idx = torch.multinomial((vals[:max_idx+1]/temperature).softmax(-1), 1).item()
                predicted_token = tokens[idx].item()
            elif sampling_method == "top-k":
                vals, tokens = torch.topk(x[0,-1],k)
                idx = torch.multinomial((vals/temperature).softmax(-1),1).item()
                predicted_token = tokens[idx].item()
            else:
                y = (x / temperature).softmax(-1).squeeze(0)
                predicted_token = torch.multinomial(y[-1], 1).item()
            response.append(predicted_token)
            if predicted_token == 2:
                break
        return self.tokenizer.decode_ids(response)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,nonlinearity="tanh",vocab_size=10000,dropout=0.0, *args, **kwargs):
        super(RNN, self).__init__()
        self.rnn = torch.nn.RNN(input_size,hidden_size,num_layers,nonlinearity,batch_first=True,dropout=dropout)
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size,embedding_dim=input_size,padding_idx=vocab_size-1)
        self.linear = torch.nn.Linear(hidden_size,vocab_size)

    def forward(self,x):
        x = self.embedding(x)
        output , _ = self.rnn(x)
        output = self.linear(output)
        return output

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,vocab_size=10000,dropout=0.0, *args, **kwargs):
        super(LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size,hidden_size,num_layers,batch_first=True,dropout=dropout)
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size,embedding_dim=input_size,padding_idx=vocab_size-1)
        self.linear = torch.nn.Linear(hidden_size,vocab_size)

    def forward(self,x):
        x = self.embedding(x)
        output , _ = self.lstm(x)
        output = self.linear(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size,n_head=4, num_layers=4,vocab_size=10000,dropout=0.0, *args, **kwargs):
        super(Transformer, self).__init__()
        transformer  = torch.nn.Transformer(
            d_model=input_size,
            dim_feedforward=hidden_size,
            nhead=n_head,
            num_encoder_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.pe = PositionalEncoding(input_size)
        self.transformer = transformer.encoder
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size,embedding_dim=input_size)
        self.linear = torch.nn.Linear(input_size,vocab_size)

    def forward(self,x, pad_mask=None,device="cuda"):
        x = self.embedding(x)
        x = self.pe(x)
        mask = torch.nn.Transformer.generate_square_subsequent_mask(x.shape[1],device=device)
        if pad_mask == None:
            out = self.transformer(x, mask, is_causal=True)
        else:
            out = self.transformer(x, mask, pad_mask, is_causal=True)
        out = self.linear(out)
        return out