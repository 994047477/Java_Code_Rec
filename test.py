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


# dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
model = Code_Rec_Model(vocab_size, embedding_size, hidden_size).cuda()
model.load_state_dict(torch.load('pkls/params_75_0.421216.pkl'))
model.eval()
# print(model)
parser = Parser()
res = parser.tokenize_string('''
 public void formatQueryAsyncV1NoSuchTagId() throws Exception {
    setupFormatQuery();
    HttpQuery query = NettyMocks.getQuery(tsdb, "");
    HttpJsonSerializer serdes = new HttpJsonSerializer(query);
    final TSQuery data_query = getTestQuery(false);
    validateTestQuery(data_query);

    final DataPoints dps = new MockDataPoints().getMock();
    final List<DataPoints[]> results = new ArrayList<DataPoints[]>(1);
    results.add(new DataPoints[] { dps });

    when(dps.getTagsAsync())
      .thenReturn(Deferred.<Map<String, String>>fromError(
          new NoSuchUniqueId("No such tagv", new byte[] { 0, 0, 1 })));

    serdes.formatQueryAsyncV1(data_query, results,
        Collections.<Annotation> emptyList()).joinUninterruptibly();
  }
@Test (expected = DeferredGroupException.class)
public void formatQueryAsyncV1NoSuchAggTagId() throws Exception {
    setupFormatQuery();
    HttpQuery query = NettyMocks.getQuery(tsdb, "");
    
    HttpJsonSerializer  serdes = new HttpJsonSerializer(query);
    final TSQuery data_query = getTestQuery(false);
    validateTestQuery(data_query);
    
    final DataPoints dps = new MockDataPoints().getMock();
    
    final List<DataPoints[]> results = new ArrayList<DataPoints[]>(1);
    results.add(new DataPoints[] { dps });

    when(dps.getAggregatedTagsAsync())
      .thenReturn(Deferred.<List<String>>fromError(
          new NoSuchUniqueId("No such tagk", new byte[] { 0, 0, 1 })));

    serdes.formatQueryAsyncV1(data_query, results,
        Collections.<Annotation> emptyList()).joinUninterruptibly();
  }
private TSQuery getTestQuery(final boolean show_stats, final boolean show_summary) {
    final TSQuery data_query = new TSQuery();
    
    data_query.setStart("1356998400");
    data_query.setEnd("1388534400");
    data_query.setShowStats(show_stats);
    data_query.setShowSummary(show_summary);
    $hole$
    final TSSubQuery sub_query = new TSSubQuery();
    
    sub_query.setMetric("sys.cpu.user");
    sub_query.setAggregator("sum");
    
    final ArrayList<TSSubQuery> sub_queries = new ArrayList<TSSubQuery>(1);
    sub_queries.add(sub_query);
    data_query.setQueries(sub_queries);

    return data_query;
  }
''')


# print(res)


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
if index > config.max_size - 1:
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


def predict(lis, topK=5, ):
    input = torch.tensor(lis).unsqueeze(0).cuda()
    out_vocab, hidden = model(input)
    out_vocab = out_vocab.view(-1, vocab_size)
    # _, pred = torch.max(out_vocab, 1)
    _, pred = out_vocab.topk(10, 1)
    return pred[-1]


predict(lis)
result_list = []

candidate_list = [[]]


def return_list(lis, candidate_list):
    max_len = 4

    for j in range(max_len):

        size1 = len(candidate_list)
        print('size1:', size1)
        for i in range(size1):

            size2 = len(candidate_list[i])

            if candidate_list[i]:
                lis3 = lis[size2:].copy()
                lis3.extend(candidate_list[i])
                new_lis = lis3
            else:
                new_lis = lis[size2:]
            # new_lis =  lis[size2:].extend( candidate_list[i] ) if candidate_list[i] else lis[size2:]
            # print(lis[size2:],candidate_list[i] )
            res_top_5 = predict(new_lis)

            for res in res_top_5:
                new_lis1 = candidate_list[i].copy()
                new_lis1.append(res.item())

                # print(candidate_list[i],res.item())
                # print(new_lis1)

                if words[res.item()] in [';', '}', '{'] or len(candidate_list[i]) >= max_len:
                    result_list.append(new_lis1)
                else:
                    candidate_list.append(new_lis1)

        for i in range(size1):
            candidate_list.pop(0)

    print(result_list)
    for i in result_list:
        print([words[s] for s in i])


return_list(lis, candidate_list)

