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
import os


# os.environ['CUDA_VISIBLE_DEVICES']='0,1'

def train():
    dataset = CodeDataset()
    # print(dataset)
    batch_size = 32
    vocab_size = 1494348
    embedding_size = 300
    hidden_size =10
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
    model = Code_Rec_Model(vocab_size, embedding_size,hidden_size).cuda()

    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)

    epochs = 100
    for epoch in range(epochs):
        print(f'epoch[{epoch}]/[{epochs}]')
        for i, (X, Y) in enumerate(dataloader):
            if torch.cuda.is_available():
                X = X.cuda()
                Y = Y.cuda()
            # hidden = model.init_hidden(batch_size)
            out_vocab, hidden = model(X)
            out_vocab = out_vocab.view(-1, vocab_size)
            loss = loss_fn(out_vocab, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss)
            print(f'i:{i} , loss:{loss}')
            # print(out_vocab)
            # print(Y)
            # print(np.shape(out_vocab))
            # print(np.shape(Y))


if __name__ == '__main__':
    train()
