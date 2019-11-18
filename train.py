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
from Model import Code_Rec_Model
from dataset import CodeDataset


def train():
    dataset = CodeDataset()
    # print(dataset)
    batch_size = 32
    vocab_size = 1494348
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
    model = Code_Rec_Model(vocab_size, 1000, 10).cuda()

    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)


    for i, (X, Y) in enumerate(dataloader):
        if torch.cuda.is_available():
            X = X.cuda()
            Y = Y.cuda()

        print(i)
        # print(X)
        # print(np.shape(X))
        print(Y)
        # hidden = model.init_hidden(batch_size)
        out_vocab,hidden= model(X)
        out_vocab = out_vocab.view(-1, vocab_size)
        loss = loss_fn(out_vocab,Y)
        print(loss)
        print(out_vocab)
        print(Y)
        # print(out_vocab)
        print(np.shape(out_vocab))
        print(np.shape(Y))
        break

    # model = Code_Rec_Model(vocab_size=1494348 ,embedding_size=100000,hidden_size= )
    # if torch.cuda.is_available():
    #     model = model.gpu()
    #     X = X.cuda()
    #     Y = Y.cuda()
    # res = model(X)


if __name__ == '__main__':
    train()
