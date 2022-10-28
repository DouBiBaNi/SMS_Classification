# -*- coding: utf-8 -*-
# @Author  : JUN
# @Time    : 2022/10/21 15:06
# @Software: PyCharm
from libs import *
import logging

def tokenlize(sentence):
    fileters = ['!', '"', ',', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>', '\?', '@', '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\n', '\x97', '\x96', '”', '“', ]
    sentence = sentence.lower()
    sentence = re.sub("|".join(fileters), " ", sentence)
    result = [i for i in sentence.split(" ") if len(i) > 0]
    return result


class WordSequence:
    MAX_VOCAB_SIZE = 10000  # 词表长度限制
    UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号

    def __init__(self):
        self.word_index_dict = {}
        self.index_word_dict = {}
        self.fited = False

    def __len__(self):
        return len(self.word_index_dict)

    def build_vocab(self, data, tokenizer=tokenlize, max_size=MAX_VOCAB_SIZE, min_freq=1):
        vocab_dict = {}
        for text in data:
            text = text.strip()
            if not text:
                continue
            for word in tokenizer(text):
                vocab_dict[word] = vocab_dict.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dict.items() if _[1] > min_freq], key=lambda x: x[1], reverse=True)[
                     :max_size]
        vocab_dict = {word[0]: idx for idx, word in enumerate(vocab_list)}
        vocab_dict.update({self.UNK: len(vocab_dict), self.PAD: len(vocab_dict) + 1})
        self.word_index_dict = vocab_dict
        self.index_word_dict = dict(zip(self.word_index_dict.values(),self.word_index_dict.keys()))
        self.fited = True
        return self.word_index_dict

    def toIndex(self,word):
        assert self.fited == True
        return self.word_index_dict.get(word,self.word_index_dict[self.UNK])

    def toWord(self, index):
        assert self.fited == True
        if index in self.index_word_dict:
            return self.index_word_dict[index]
        return self.UNK

    def transform(self, data, max_len=None):
        res = []
        for sentence in data:
            sentence = tokenlize(sentence)
            if len(sentence) > max_len:
                sentence = sentence[:max_len]
            else:
                sentence =sentence + [self.PAD]*(max_len-len(sentence))
            index_seq = [self.toIndex(word) for word in sentence]
            res.append(index_seq)
        return res

def build_dataset(config):
    df = pd.read_csv("./SMSDatasets/data/spam.csv", delimiter=",", encoding="latin1")
    df["target"] = df["v1"].factorize()[0]
    # dummies = pd.get_dummies(df["v1"])
    X_train, X_test, Y_train, Y_test = train_test_split(df["v2"].values, df["target"].values, test_size=0.2)

    ws = WordSequence()
    d = ws.build_vocab(list(X_train) + list(X_test))
    config.n_vocab = len(ws)
    # print(ws.word_index_dict)
    X_train, X_test = ws.transform(X_train, max_len=config.max_len), ws.transform(X_test, max_len=config.max_len)

    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, "rb"))
    else:
        vocab = d
        # vocab = ws.build_vocab(df.v2.values)
        pkl.dump(vocab, open(config.vocab_path, "wb"))
        # config.n_vocab = len(vocab)
    print(f"Vocab size: {len(vocab)}")

    def load_dataset(X, Y):
        res = []
        for idx, val in enumerate(X):
            res.append([val,Y[idx]])
        return res

    train = load_dataset(X_train, Y_train)
    test = load_dataset(X_test, Y_test)

    return vocab, train, test

class DatasetIterater(object):
    def __init__(self, batches, batch_size):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas])
        y = torch.LongTensor([_[1] for _ in datas])
        return x, y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index *self.batch_size:len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index*self.batch_size:(self.index+1)*self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size)
    return iter

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return time_dif

def save_log(config, msg):
    logging.basicConfig(filename=config.log_path+"/"+time.strftime('%m-%d_%H.%M', time.localtime())+".txt"
                        , filemode="w",level=logging.INFO)
    print(msg)
    logging.info(msg)

def train(config, model, train_iter, test_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    total_batch = 0
    best_loss = float("inf")
    last_improve = 0
    flag = False
    writer = SummaryWriter(log_dir=config.log_path+'/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        msg = 'Epoch [{}/{}]'.format(epoch + 1, config.num_epochs)
        save_log(config, msg)
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                if loss < best_loss:
                    best_loss = loss
                    torch.save(model.state_dict(), config.save_path)
                    last_improve = total_batch
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Time: {3}'.format(total_batch,
                                                                                                      loss.item(), train_acc, time_dif,)
                save_log(config, msg)
                writer.add_scalar("train_loss", loss.item(), total_batch)
                writer.add_scalar("train_acc", train_acc, total_batch)

                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                msg = "No optimization for a long time, auto-stopping..."
                save_log(config, msg)
                flag = True
                break
            if flag:
                break
        writer.close()
        test(config, model, test_iter)

def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'.format(test_loss, test_acc)
    save_log(config, msg)
    save_log(config, "Precision, Recall and F1-Score...")
    save_log(config, test_report)
    save_log(config, "Confusion Matrix...")
    save_log(config, test_confusion)
    time_dif = get_time_dif(start_time)
    msg = f"Time usage:{time_dif}"
    save_log(config, msg)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)