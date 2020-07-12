import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.tag2idx = {}
        self.idx2tag = []
    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word)-1
    def add_tag(self, tag):
        if tag not in self.tag2idx:
            self.idx2tag.append(tag)
            self.tag2idx[tag] = len(self.idx2tag)-1

class Corpus():
    def __init__(self):
        self.dictionary = Dictionary()
    def readfile(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f1:
            lines = [line.strip().split() for line in f1.readlines()]
        stns = []
        tags = []
        sent = []
        tag = []
        for line in lines:
            if line!=[]:
                sent.append(line[1])
                tag.append(line[3])
            else:
                stns.append(sent)
                tags.append(tag)
                sent = []
                tag = []
        return stns, tags
    def add_dict(self, stns, tags):
        for sent in stns:
            for word in sent:
                self.dictionary.add_word(word)
        self.dictionary.add_word('<UNK>')
        for tag in tags:
            for e in tag:
                self.dictionary.add_tag(e)
    def toidx(self, stns, tags):
        sentsidx = []
        tagsidx = []
        for sent in stns:
            sentsidx.append(torch.LongTensor([self.dictionary.word2idx[word] if word in self.dictionary.word2idx else self.dictionary.word2idx['<UNK>'] for word in sent ]))
        for tag in tags:
            tagsidx.append(torch.LongTensor([self.dictionary.tag2idx.get(e, -1) for e in tag]))
        return sentsidx, tagsidx

class LSTMModel(nn.Module):
    def __init__(self, ntoken, ntag, ninp, nhid, nlayers, device, bsz = 1,dropout=0.5):
        #ntoken词表长度，ninp词向量维度，nhid为lstm内部向量维度，nlayers为lstm的层数
        super(LSTMModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout, bidirectional=True)
        self.decoder = nn.Linear(nhid*2, ntag)
        #*2是因为双向
        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers
        self.hidden = self.init_hidden(bsz, device)
        
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
    def forward(self, input):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb.unsqueeze(1), self.hidden)
        #rnn的输入默认第二个维度是批数
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, hidden
    def init_hidden(self, bsz, device):
        #bsz即batchsize
        weight = next(self.parameters())
        #new_zeros用于创建同类型的指定维度的张量
        #*2是因为双向
        return (weight.new_zeros(self.nlayers*2, bsz, self.nhid).to(device),
                weight.new_zeros(self.nlayers*2, bsz, self.nhid).to(device))
#建立模型
corpus = Corpus()
stns, tags = corpus.readfile('bigdata/train')
corpus.add_dict(stns, tags)
sentidx, tagidx = corpus.toidx(stns, tags)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model = LSTMModel(len(corpus.dictionary.word2idx), len(corpus.dictionary.tag2idx), 256, 256, 2, device)


model.to(device)

criterion = nn.CrossEntropyLoss().to(device)
lr = 0.01
optimizer = optim.SGD(model.parameters(), lr = lr)



devsents, devtags = corpus.readfile('bigdata/test')
devsentidx, devtagidx = corpus.toidx(devsents, devtags)



def train(epoch, sentidx, tagidx):
    model.train()
    for i in range(epoch):       
        for j in range(len(sentidx)):
            model.zero_grad()
            final, _ = model.forward(sentidx[j].to(device))
            loss = criterion(final.squeeze(1), tagidx[j].to(device))            
            loss.backward()
            optimizer.step()

def evaluate(devsentidx, devtagidx):
    model.eval()
    count = 0.
    right = 0.
    for i in range(len(devsentidx)):
        final, _ = model.forward(devsentidx[i].to(device))
        predict = final.squeeze(1)
        
        predict = predict.argmax(1)
        devtagidx[i].to(device)
        count += devtagidx[i].shape[0]
        right += torch.sum(torch.eq(predict, devtagidx[i].to(device)))
    return right.float()/count

train(1, sentidx, tagidx)
print('测试集上的精度为：', evaluate(devsentidx, devtagidx))
train(1, sentidx, tagidx)
print('测试集上的精度为：', evaluate(devsentidx, devtagidx))
train(1, sentidx, tagidx)
print('测试集上的精度为：', evaluate(devsentidx, devtagidx))
#print(evaluate(devsentidx, devtagidx))

