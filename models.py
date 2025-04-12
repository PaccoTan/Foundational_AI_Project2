import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import sentencepiece as spm

class LanguageModel(ABC, nn.Module):
    def __init__(self, model: nn.Module, tokenizer: spm.SentencePieceProcessor):
        self.model = model
        self.tokenizer = tokenizer
        
    def forward(self, x, temperature = 1):
        logits = self.model(x)
        y = (logits/temperature).softmax(-1)
        predicted_token = torch.multinomial(y,1).item()
        return predicted_token

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,nonlinearity="tanh",vocab_size=10000,dropout=0.0 *args, **kwargs):
        super(RNN, self).__init__()
        self.rnn = torch.nn.RNN(input_size,hidden_size,num_layers,nonlinearity,batch_first=True,dropout=dropout)
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size+1,embedding_dim=input_size,padding_idx=vocab_size-1)
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
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size+1,embedding_dim=input_size,padding_idx=vocab_size-1)
        self.linear = torch.nn.Linear(hidden_size,vocab_size)

    def forward(self,x):
        x = self.embedding(x)
        output , _ = self.lstm(x)
        output = self.linear(output)
        return output