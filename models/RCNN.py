# -*- coding: utf-8 -*-
# @Author  : JUN
# @Time    : 2022/10/22 16:47
# @Software: PyCharm
import torch
from torch import nn
import torch.nn.functional as F

class Config(object):
    def __init__(self):
        self.train_path = "./SMSDatasets/data/train.txt"
        self.test_path = "./SMSDatasets/data/test.txt"
        self.vocab_path = "./SMSDatasets/data/vocab.pkl"
        self.save_path = "./SMSDatasets/saved_dict/RCNN.ckpt"
        self.log_path = "./SMSDatasets/log/rcnn"
        self.class_list = ["ham", "spam"]
        self.num_classes = 2
        self.embedding_pretrained = None
        self.dropout = 1.0
        self.require_improvement = 1000
        self.n_vocab = 0
        self.num_epochs = 10
        self.batch_size = 64
        self.pad_size = 32
        self.max_len = 32
        self.learning_rate = 1e-3
        self.embed = self.embedding_pretrained.size(1) if self.embedding_pretrained is not None else 300
        self.hidden_size = 256
        self.num_layers = 1


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.maxpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.hidden_size * 2 + config.embed, config.num_classes)

    def forward(self, x):
        x = x
        embed = self.embedding(x)  # [batch_size, seq_len, embeding]=[64, 32, 64]
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out