#!/usr/bin/env python
# encoding: utf-8
'''
@author: leexuan
@contact: xuanli19@fudan.edu.cn
@Time : 2019/11/23 下午8:28
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
from dataset import CodeDataset, read_data_counter
import os
from code_parse import Parser

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# 给定一段代码预测后续一段代码 。
# model.load_state_dict(torch.load('params.pkl'))
batch_size = 32
vocab_size = 1494348
embedding_size = 300
hidden_size = 10

# dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
model = Code_Rec_Model(vocab_size, embedding_size, hidden_size).cuda()
model.load_state_dict(torch.load('pkls/params_13_0.371697.pkl'))
model.eval()
# print(model)
parser = Parser()
res = parser.tokenize_string('''
import java.security.GeneralSecurityException;
import java.security.PrivateKey;
public class Demo{
    public byte[] sign(String message, String digestAlgorithm, PrivateKey pk) throws GeneralSecurityException {
        byte[] messageByte = message.getBytes();
        String signMode = null;
        if(pk == null){
            pk = getPrivateKey("RSA");
            String encryptionAlgorithm = pk.getAlgorithm();
            signMode = combine(encryptionAlgorithm,digestAlgorithm);
        }else{ 
            String encryptionAlgorithm = pk.getAlgorithm();
            signMode = $hole$
    }
}
''')
print(res)


def get_index(res):
    for i in range(len(res)):
        if res[i].type == 'NAME' and res[i].value == '$hole$':
            return i


index = get_index(res)
print(index)
lis = []
di = read_data_counter()
di_5 = set([k for k, v in di.items() if v > 5])
di_5.add('<unk>')
words = sorted(list(di_5))
word_indices = dict((word, index) for index, word in enumerate(words))
if index > 29:
    for i in range(index - 30, index):
        word = res[i]
        if word.type == 'NUM':
            lis.append('<num>')
        elif word.type == 'STRING_LITERAL':
            lis.append('<str>')
        elif word.value not in di_5:
            lis.append('<unk>')
        else:
            lis.append(word.value)
print(lis)
lis = [word_indices[i] for i in lis]
print(lis)
input = torch.tensor(lis).unsqueeze(0).cuda()
out_vocab, hidden = model(input)
out_vocab = out_vocab.view(-1, vocab_size)
_, pred = torch.max(out_vocab, 1)
print(pred)
print([words[i] for i in pred])
max_pred_size = 30
for i in range(max_pred_size):
    lis = lis[1:]
    lis.append(pred.item())
    # print(lis)
    input = torch.tensor(lis).unsqueeze(0).cuda()
    out_vocab, hidden = model(input)
    out_vocab = out_vocab.view(-1, vocab_size)
    _, pred = torch.max(out_vocab, 1)
    # print(_,pred)
    print([words[i] for i in pred])
