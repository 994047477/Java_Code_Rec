#!/usr/bin/env python
# encoding: utf-8
'''
@author: leexuan
@contact: xuanli19@fudan.edu.cn
@Time : 2019/11/26 下午4:50
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
import config
from Model import Code_Rec_Model
from dataset import CodeDataset, read_data_counter
import os
from torch.utils.data.sampler import SubsetRandomSampler

def test():
    dataset = CodeDataset(data_num=500)
    # print(dataset)
    batch_size = config.batch_size
    vocab_size = config.vocab_size  # >5的时候size是1494348
    embedding_size = config.embedding_size
    hidden_size = config.hidden_size
    train_loader = DataLoader(dataset, 1)

    # dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
    model = Code_Rec_Model(vocab_size, embedding_size, hidden_size).cuda()
    # with open('pkls/params_9_0.394187.pkl') as f:
    #     model.load_state_dict(f)
    model.load_state_dict(torch.load('pkls/params_15_0.420176.pkl'))
    model.eval()
    # model = torch.nn.DataParallel(model)
    di = read_data_counter()
    di_5 = set([k for k, v in di.items() if v > 50])
    di_5.add('<unk>')
    words = sorted(list(di_5))

    def convert_to_word(ten):
        return [words[tn.item()] for tn in ten]

    weights = []
    loss_fn = nn.CrossEntropyLoss().cuda()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for i1, (X1, Y1) in enumerate(train_loader):
        if torch.cuda.is_available():
            X1 = X1.cuda()
            Y1 = Y1.cuda()
        out_vocab, hidden = model(X1)
        out_vocab = out_vocab.view(-1, vocab_size)
        _, pred = torch.topk(out_vocab, 5)
        print( convert_to_word( X1[0]) )
        print(words[Y1[0][-1]])
        print( [words[wo] for  wo in pred[-1]] )
        print('-----------------------')



if __name__ == '__main__':
    test()
