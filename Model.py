#!/usr/bin/env python
# encoding: utf-8
'''
@author: leexuan
@contact: xuanli19@fudan.edu.cn
@Time : 2019-11-02 18:40
@desc: 
'''
import math

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
    '''
        a simple lstm model
    '''
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


class Code_Rec_Model_enhanced(nn.Module):
    '''
        Text CNN model
    '''
    def __init__(self):
        super(Code_Rec_Model_enhanced, self).__init__()
        param = {
            'vocab_size': config.vocab_size,
            'embed_dim': config.embedding_size,
            'class_num': config.vocab_size,
            "kernel_num": 16,
            "kernel_size": [3, 4, 5],
            "dropout": 0.5,
        }
        ci = 1  # input chanel size
        kernel_num = param['kernel_num']  # output chanel size
        kernel_size = param['kernel_size']
        vocab_size = param['vocab_size']
        embed_dim = param['embed_dim']
        dropout = param['dropout']
        class_num = param['class_num']
        self.param = param
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.conv11 = nn.Conv2d(ci, kernel_num, (kernel_size[0], embed_dim))
        self.conv12 = nn.Conv2d(ci, kernel_num, (kernel_size[1], embed_dim))
        self.conv13 = nn.Conv2d(ci, kernel_num, (kernel_size[2], embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(kernel_size) * kernel_num, class_num)

    @staticmethod
    def conv_and_pool(x, conv):
        # x: (batch, 1, sentence_length,  )
        x = conv(x)
        # x: (batch, kernel_num, H_out, 1)
        x = F.relu(x.squeeze(3))
        # x: (batch, kernel_num, H_out)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        #  (batch, kernel_num)
        return x

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, input):
        # x: (batch, sentence_length)
        x = self.embed(input)
        # x: (batch, sentence_length, embed_dim)
        # TODO init embed matrix with pre-trained
        x = x.unsqueeze(1)
        # x: (batch, 1, sentence_length, embed_dim)
        x1 = self.conv_and_pool(x, self.conv11)  # (batch, kernel_num)
        x2 = self.conv_and_pool(x, self.conv12)  # (batch, kernel_num)
        x3 = self.conv_and_pool(x, self.conv13)  # (batch, kernel_num)
        x = torch.cat((x1, x2, x3), 1)  # (batch, 3 * kernel_num)
        x = self.dropout(x)
        logit = F.log_softmax(self.fc1(x), dim=1)
        return logit


if __name__ == '__main__':
    # model = Code_Rec_Model()
    pass
    model = Code_Rec_Model_enhanced()
    print(model)



