#!/usr/bin/env python
# encoding: utf-8
'''
@author: leexuan
@contact: xuanli19@fudan.edu.cn
@Time : 2019-11-02 18:40
@desc: 
'''
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

import config


class Code_Rec_Model(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(Code_Rec_Model, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size,num_layers=config.num_layers)
        self.drop = nn.Dropout(0.5)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, text,hidden=None):
        emb = self.word_embedding(text)  # batch_size ,sequence , embedding
        output, hidden = self.lstm(emb,hidden)  # 2,30,10  ,   2个tuple(30,10)
        output = self.drop(output)
        decoded = self.linear(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden
    # TODO 检查hidden
    def init_hidden(self, requires_grad=True):
        # return (weight.new_zeros((30, bsz, 10), requires_grad=requires_grad),
        #             weight.new_zeros((30, bsz, 10), requires_grad=requires_grad))
        return (torch.rand((config.num_layers, config.max_size, config.hidden_size),requires_grad=requires_grad).cuda(),\
               torch.rand(((config.num_layers, config.max_size, config.hidden_size)),requires_grad=requires_grad).cuda())


if __name__ == '__main__':
    model = Code_Rec_Model()
