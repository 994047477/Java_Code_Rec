#!/usr/bin/env python
# encoding: utf-8
'''
@author: leexuan
@contact: xuanli19@fudan.edu.cn
@Time : 2019/11/28 下午3:21
@desc: 
'''
max_size = 50
batch_size = 48
hidden_size = 512
vocab_size = 183298   # >5的时候size是1494348
embedding_size = 500
num_layers = 1

# 35,64,512,183298,500,2   acc:0.416 lstm
# 40,64,512,183298,300,2   acc:0.420 lstm



