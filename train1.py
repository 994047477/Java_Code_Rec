#!/usr/bin/env python
# encoding: utf-8
'''
@author: leexuan
@contact: xuanli19@fudan.edu.cn
@Time : 2019/12/2 下午6:49
@desc: 训练 TextCNN model
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

import config
from Model import Code_Rec_Model, Code_Rec_Model_enhanced
from dataset import CodeDataset, read_data_counter
import os

from torch.utils.data.sampler import SubsetRandomSampler


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
    dataset = CodeDataset(data_num=10)
    # print(dataset)
    batch_size = config.batch_size
    vocab_size = config.vocab_size  # >5的时候size是1494348
    embedding_size = config.embedding_size
    hidden_size = config.hidden_size

    train_loader, valid_loader = prepare_train_valid_loaders(dataset, 0.1, batch_size)

    # dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
    model = Code_Rec_Model_enhanced().cuda()
    # model = torch.nn.DataParallel(model)
    di = read_data_counter()
    di_5 = set([k for k, v in di.items() if v > 50])
    di_5.add('<unk>')
    words = sorted(list(di_5))

    loss_fn = nn.CrossEntropyLoss().cuda()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epochs = 100
    hidden = None
    for epoch in range(epochs):
        model.train()
        print(f'epoch[{epoch}]/[{epochs}]')
        for i, (X, Y) in enumerate(train_loader):

            if torch.cuda.is_available():
                X = X.cuda()
                Y = Y.cuda()
            optimizer.zero_grad()
            out_vocab = model(X)
            # out_vocab = out_vocab.view(-1, vocab_size)
            print(out_vocab)
            print(np.shape(out_vocab))
            _,out1 = torch.max(out_vocab,0)
            print(out1)
            print(np.shape(out1))


            loss = loss_fn(out_vocab.view(-1, vocab_size), Y.view(-1))

            loss.backward()
            # torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optimizer.step()
        if True:
            # validation
            model.eval()
            acc = 0.0
            sum1 = 0
            all_sum = 0
            for i1, (X1, Y1) in enumerate(valid_loader):
                if torch.cuda.is_available():
                    X1 = X1.cuda()
                    Y1 = Y1.cuda()
                out_vocab, hidden = model(X1)
                out_vocab = out_vocab.view(-1, vocab_size)
                _, pred = torch.max(out_vocab, 1)
                # loss = loss_fn(out_vocab, Y1)
                # print(pred)
                # print(np.shape(pred))
                # print(np.shape(Y1.data) )
                # print(np.shape(pred))# shape:[32]
                # print(np.shape(Y1))
                sum1 += torch.sum(pred == Y1.view(-1).data).item()

                all_sum += len(Y1.view(-1))
                acc = sum1 * 1.0 / all_sum

            # print(sum1, all_sum)
            print(f'[{epoch}]/[{epochs}]  accuracy:{acc:.6f}')
            if epoch % 2 == 1:
                torch.save(model.state_dict(), f'pkls/params_{epoch}_{acc:.6f}.pkl')


if __name__ == '__main__':
    train()
