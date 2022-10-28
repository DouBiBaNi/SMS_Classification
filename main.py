# -*- coding: utf-8 -*-
# @Author  : JUN
# @Time    : 2022/10/21 14:59
# @Software: PyCharm
import torch.cuda

from utils import *
import argparse
from importlib import import_module

parser = argparse.ArgumentParser(description='SMS Classification')
parser.add_argument('-m','--model', type=str, required=True, help='choose a model: CNN, LSTM, RCNN, DPCNN, Transformer')
parser.add_argument('-e','--epoch', default=10, type=int, help='choose epochs you what to train')
parser.add_argument('-bs','--batch_size', default=64, type=int, help='set the batch_size')
parser.add_argument('-lr','--learning_rate', default=1e-3, type=float, help='set the learning rate')
args = parser.parse_args()

if __name__ == '__main__':
    model_name = args.model
    sms = import_module("models."+model_name)
    config = sms.Config()
    config.num_epochs = args.epoch
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    np.random.seed(1)
    torch.manual_seed(1)

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time Usage:",time_dif)

    config.n_vocab = len(vocab)
    model = sms.Model(config)
    print(model.parameters)
    train(config, model, train_iter, test_iter)


