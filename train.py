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
    dataloader = DataLoader(dataset=dataset,batch_size=4 )
    model = Code_Rec_Model()
    for i,(X,Y) in enumerate(dataloader):
        print(i)
        print(X)
        print(np.shape(X))
        print(Y)


        break

    # model = Code_Rec_Model(vocab_size=1494348 ,embedding_size=100000,hidden_size= )
    # if torch.cuda.is_available():
    #     model = model.gpu()
    #     X = X.cuda()
    #     Y = Y.cuda()
    # res = model(X)


if __name__ == '__main__':
    train()

