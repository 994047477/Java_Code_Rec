#!/usr/bin/env python
# encoding: utf-8
'''
@author: leexuan
@contact: xuanli19@fudan.edu.cn
@Time : 2019-11-02 18:48
@desc: 
'''

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import os
import random
import nce
import config
from Model import Code_Rec_Model
from dataset import CodeDataset, read_data_counter
import os

from torch.utils.data.sampler import SubsetRandomSampler

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def prepare_train_valid_loaders(trainset, valid_size=0.2,
                                batch_size=32):
    '''
    Split trainset data and prepare DataLoader for training and validation

    Args:
        trainset (Dataset): data
        valid_size (float): validation size, defalut=0.2
        batch_size (int) : batch size, default=128
    '''

    # obtain training indices that will be used for validation
    num_train = len(trainset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               sampler=valid_sampler)

    return train_loader, valid_loader


def train():
    dataset = CodeDataset(data_num=300)
    # print(dataset)
    batch_size = config.batch_size
    vocab_size = config.vocab_size  # >5的时候size是1494348
    embedding_size = config.embedding_size
    hidden_size = config.hidden_size

    train_loader, valid_loader = prepare_train_valid_loaders(dataset, 0.1, batch_size)

    # dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
    model = Code_Rec_Model(vocab_size, embedding_size, hidden_size).cuda()


    # model = torch.nn.DataParallel(model)
    di = read_data_counter()
    di_5 = set([k for k, v in di.items() if v > 50])
    di_5.add('<unk>')
    words = sorted(list(di_5))
    weights = []
    di_test = { k:v for k, v in di.items() if v > 50}
    di_test['<unk>']=1000

    freq = [0] * len(words)
    for idx in range(len(words)):
        if words[idx] in ['<str>','<num>','<unk>']:
            freq[idx] = 1000
        else:
            freq[idx] = di[words[idx]]
    Q = np.array(freq)
    Q = Q/Q.sum().astype(float)
    Q = torch.FloatTensor(Q).cuda()
    # loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights).cuda() ).cuda()
    loss_fn = nn.CrossEntropyLoss().cuda()
    # Q = Q_from_tokens(corpus.train, corpus.dictionary)

    # loss_fn = nce.NCELoss(Q, 25, 9.5).cuda()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)

    epochs = 100
    hidden = None
    for epoch in range(epochs):
        model.train()
        print(f'epoch[{epoch}]/[{epochs}]')
        for i, (X, Y) in enumerate(train_loader):
            hidden = model.init_hidden()
            if torch.cuda.is_available():
                X = X.cuda()
                Y = Y.cuda()
            optimizer.zero_grad()
            out_vocab, hidden = model(X, hidden)
            # out_vocab = out_vocab.view(-1, vocab_size)
            loss = loss_fn(out_vocab.view(-1, vocab_size), Y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        if True:
            # validation
            model.eval()
            acc = 0.0
            sum1 = 0
            sum2 = 0
            all_sum = 0
            for i1, (X1, Y1) in enumerate(valid_loader):
                if torch.cuda.is_available():
                    X1 = X1.cuda()
                    Y1 = Y1.cuda()
                out_vocab, hidden = model(X1)
                out_vocab = out_vocab.view(-1, vocab_size)
                # _, pred = torch.max(out_vocab, 1)
                # sum1 += torch.sum(pred == Y1.view(-1).data).item()
                _1,pred1= torch.topk(out_vocab,5)

                for j in range( len(Y1.view(-1)) ):
                    if Y1.view(-1)[j] in pred1[j]:
                        sum2 =sum2+1
                    if Y1.view(-1)[j] ==pred1[j][0]:
                        sum1 = sum1+1
                # print(pred1)
                # print(np.shape(pred1))
                # print('----')
                all_sum += len(Y1.view(-1))
            acc = sum1 * 1.0 / all_sum
            acc_top5 = sum2 * 1.0 / all_sum
            # print(sum1, all_sum)
            print(f'[{epoch}]/[{epochs}]  accuracy:{acc:.6f}')
            print(f'[{epoch}]/[{epochs}]  top5 accuracy:{acc_top5:.6f}')
            # top5 acc

            if epoch % 2 == 1:
                torch.save(model.state_dict(), f'pkls/params_{epoch}_{acc:.6f}.pkl')


if __name__ == '__main__':
    train()
