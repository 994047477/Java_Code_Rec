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


    def forward(self, text):

        emb = self.word_embedding(text) # batch_size ,sequence , embedding
        output,hidden = self.lstm(emb)  # 2,30,10  ,   2个tuple(30,10)
        # print(np.shape(output),np.shape(hidden))
        shape1,shape2 = output.shape[0],output.shape[1]
        output = output[:,-1,:]
        print(np.shape(output))

        output = output.view(-1 ,output.shape[1]) # (2*30)*10
        # print(np.shape(output))
        out_vocab = self.linear(output)
        # print(np.shape(out_vocab))#  60,1494348
        out_vocab = out_vocab.view( shape1,-1 )
        return out_vocab,hidden

    def init_hidden(self, bsz, requires_grad=True):
        weight = next(self.parameters())
        # return (weight.new_zeros((30, bsz, 10), requires_grad=requires_grad),
        #             weight.new_zeros((30, bsz, 10), requires_grad=requires_grad))
        return torch.rand((30, bsz, 10),requires_grad=requires_grad),\
               torch.rand(((30, bsz, 10)),requires_grad=requires_grad)


if __name__ == '__main__':
    model = Code_Rec_Model()



