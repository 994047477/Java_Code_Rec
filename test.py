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

import config
from Model import Code_Rec_Model
from dataset import CodeDataset, read_data_counter
import os
from code_parse import Parser

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 给定一段代码预测后续一段代码
vocab_size = config.vocab_size
embedding_size = config.embedding_size
hidden_size = config.hidden_size
# 183298 512 512 占用显存4000M

# dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
model = Code_Rec_Model(vocab_size, embedding_size, hidden_size).cuda()
model.load_state_dict(torch.load('pkls/params_1_0.400220.pkl'))
model.eval()
# print(model)
parser = Parser()
res = parser.tokenize_string('''
public class MyMapper implements PartitionMapper {

    @Override
    public PartitionPlan mapPartitions() throws Exception {
        return new PartitionPlanImpl() {
            @Override
            public Properties[] getPartitionProperties() {
                
                Properties[] props = new Properties[getPartitions()];
                $hole$
                for (int i = 0; i < getPartitions(); i++) {
                    props[i] = new Properties();
                    props[i].setProperty("start", String.valueOf(i * 10 + 1));
                    props[i].setProperty("end", String.valueOf((i + 1) * 10));
                }
                return props;
            }
        };
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
di_5 = set([k for k, v in di.items() if v > 50])
di_5.add('<unk>')
words = sorted(list(di_5))
word_indices = dict((word, index) for index, word in enumerate(words))
if index > config.max_size-1:
    for i in range(index - config.max_size, index):
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
# _, pred = torch.max(out_vocab, 1)
_,pred = out_vocab.topk(5,1,True,True)

print(pred)
print([ words[i] for i in pred[-1]])
pred = pred[-1][0]
# print([words[i] for i in pred])
max_pred_size = 30
for i in range(max_pred_size):
    lis = lis[1:]
    # print(pred[-1][0])
    lis.append(pred.item())
    print(words[pred.item()])
    input = torch.tensor(lis).unsqueeze(0).cuda()
    print([ words[inp.item()] for inp in input[0]])
    out_vocab, hidden = model(input)
    _, pred = out_vocab.topk(5, 1, True, True)
    # print(pred)
    pred = pred[0]
    # print([words[i] for i in pred[-1]])
    pred = pred[-1][0]
