#!/usr/bin/env python
# encoding: utf-8
'''
@author: leexuan
@contact: xuanli19@fudan.edu.cn
@Time : 2019/11/28 下午3:21
@desc: 
'''

max_size = 40
batch_size = 512*2
hidden_size = 512
vocab_size = 6290 # 183298   # >5的时候size是1494348
embedding_size = 500
num_layers = 1


# 实验 : 2 * gtx 1080Ti
# 35,64,512,183298,500,2   acc:0.416
# 40,64,512,183298,300,2   acc:0.420
# 45,32,512,183298,500,1   acc:xxxxx  显存:10161MB
# dropout, earlystoping
# NCE loss

#
# max_size = 45
# batch_size = 32
# hidden_size = 512
# vocab_size = 183298   # >5的时候size是1494348
# embedding_size = 500
# num_layers = 1
# epoch[0]/[100]  dataset:300
# [0]/[100]  accuracy:0.405953
# [0]/[100]  top5 accuracy:0.694536
# epoch[1]/[100]
# [1]/[100]  accuracy:0.406660
# [1]/[100]  top5 accuracy:0.695369
# epoch[2]/[100]
# [2]/[100]  accuracy:0.408135
# [2]/[100]  top5 accuracy:0.695997
# epoch[3]/[100]
# [3]/[100]  accuracy:0.408692
# [3]/[100]  top5 accuracy:0.696398
# epoch[4]/[100]
# [4]/[100]  accuracy:0.408816
# [4]/[100]  top5 accuracy:0.695936
# epoch[5]/[100]
# [5]/[100]  accuracy:0.409157
# [5]/[100]  top5 accuracy:0.696637
# epoch[6]/[100]
# [6]/[100]  accuracy:0.409320
# [6]/[100]  top5 accuracy:0.696985
# epoch[7]/[100]
# [7]/[100]  accuracy:0.409700
# [7]/[100]  top5 accuracy:0.697106
# epoch[8]/[100]
# [8]/[100]  accuracy:0.410099
# [8]/[100]  top5 accuracy:0.697265
# epoch[9]/[100]
# [9]/[100]  accuracy:0.409479
# [9]/[100]  top5 accuracy:0.697005
# epoch[10]/[100]
# [10]/[100]  accuracy:0.409032
# [10]/[100]  top5 accuracy:0.697365
# epoch[11]/[100]
# [11]/[100]  accuracy:0.410068
# [11]/[100]  top5 accuracy:0.696966
# epoch[12]/[100]