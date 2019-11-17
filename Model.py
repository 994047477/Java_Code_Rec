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



class Code_Rec_Model(nn.Module):

    def __init__(self,vocab_size ,embedding_size ,hidden_size):
        super(Code_Rec_Model, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size,embedding_size)
        self.lstm = nn.LSTM(embedding_size,hidden_size)
        self.linear = nn.Linear(hidden_size,vocab_size)


    def forward(self, text,hidden):
        emb = self.word_embedding(text)
        output,hidden = self.lstm(emb,hidden)
        output = output.view(-1 ,output.shape[2])
        out_vocab = self.linear(output)
        out_vocab = out_vocab.view(output.size(0),output.size(1),out_vocab[-1] )
        return out_vocab,hidden


if __name__ == '__main__':
    model = Code_Rec_Model()



