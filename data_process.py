#!/usr/bin/env python
# encoding: utf-8
'''
@author: leexuan
@contact: xuanli19@fudan.edu.cn
@Time : 2019-11-02 19:58
@desc: 
'''
import pickle
from collections import Counter


def read_data_counter(file_path='all_word.pkl'):
    with open(file_path, 'rb') as f:
        di = pickle.load(f)
    return di


if __name__ == '__main__':
    di = read_data_counter()
    print(len(di))
    # di_3 = {k:v for k, v in di.items() if v>5}
    di_5 = set([k for k, v in di.items() if v > 5])
    di_5.add('<unk>')
    print(len(di_5))  # 1494348  one-hot vector
    words = sorted(list(di_5))
    word_indices = dict((word, index) for index, word in enumerate(words))
    print(words[1407])
    print(words[1406])
    print(words[1417])
    print(words[1422])


    # counter  = Counter(di_3)

    # sentence len 50
    # embedding-dim 1000
    # 高频词 1000000   次数>=5
