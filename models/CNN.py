# -*- coding: utf-8 -*-
# @Author  : JUN
# @Time    : 2022/10/22 14:32
# @Software: PyCharm
import torch
from torch import nn
import torch.nn.functional as F

class Config(object):
    def __init__(self):
        # textCNN pram"
        self.train_path = "./SMSDatasets/data/train.txt"
        self.test_path = "./SMSDatasets/data/test.txt"
        self.vocab_path = "./SMSDatasets/data/vocab.pkl"
        self.save_path = "./SMSDatasets/saved_dict/CNN.ckpt"
        self.log_path = f"./SMSDatasets/log/cnn"
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
        self.filter_sizes = (2, 3, 4)
        self.num_filters = 256


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes]
        )
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self,x,conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out,conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


