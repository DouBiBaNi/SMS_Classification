# -*- coding: utf-8 -*-
# @Author  : JUN
# @Time    : 2022/10/21 15:13
# @Software: PyCharm
from torch import nn

class Config(object):
    def __init__(self):
        # SMS spam classifiaction"
        self.train_path = "./SMSDatasets/data/train.txt"
        self.test_path = "./SMSDatasets/data/test.txt"
        self.vocab_path = "./SMSDatasets/data/vocab.pkl"
        self.save_path = "./SMSDatasets/saved_dict/LSTM.ckpt"
        self.bidirect = False
        self.log_path = f"./SMSDatasets/log/lstm{2 if self.bidirect else 1}"
        self.dropout = 0.5
        self.require_improvement = 1000
        self.n_vocab = 0
        self.num_classes = 2
        self.class_list = ["ham","spam"]
        self.num_epochs = 10
        self.max_len = 32
        self.batch_size = 64
        self.pad_size = 32
        self.learning_rate = 1e-3
        self.embedding_pretrained = None
        self.embed = self.embedding_pretrained.size(1) if self.embedding_pretrained is not None else 300
        self.hidden_size = 128
        self.num_layers = 2


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab-1)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=config.bidirect, batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size*2 if config.bidirect else config.hidden_size, config.num_classes)

    def forward(self, x):
        out = self.embedding(x)
        out, _ = self.lstm(out)
        out = self.fc(out[:,-1,:])
        return out