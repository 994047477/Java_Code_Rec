#!/usr/bin/env python
# encoding: utf-8
'''
@author: leexuan
@contact: xuanli19@fudan.edu.cn
@Time : 2019-10-28 15:16
@desc: 
'''
# 包括 去掉注释段，提取函数段里面的主要代码
# 将数字换成<num> ,字符串换成<str>，低频词汇换成<unk>
# Model :
#   代码token 转换成one-hot 向量
#   one-hot -> embedding
#   embedding -> LSTM
#   学习出来的就是LSTM语言模型

import glob
import re
import copy
from collections import Counter


def code_process(listname='../ASE_javacode/*'):
    lis = glob.glob(listname)
    # print(lis[:10])
    for filename in lis[2:3]:
        read_javacode(filename)


def read_javacode(filename):
    with open(filename, 'r')as f:
        # 去除代码空行
        code = "".join(line for line in f.readlines() if line != '\n')

        # 代码注释的去除
        while True:
            try:
                start_index = code.index(r'/*')
                end_index = code.index(r'*/')
            except Exception:
                break

            # print(code[start_index:end_index + 2])
            code = code[:start_index] + code[end_index + 3:]

        # 代码词频统计
        # print(code)
        code_split = re.split('\n|;|\t| |\)|\(|\.|=', code)
        split_token = ['\n', '', ';', '\t', ' ', ')', '(', '=']
        # |.|;|:| |(|)|@|{|}|[|]
        code_split1 = copy.deepcopy(code_split)
        for i in split_token:
            try:
                code_split1 = re.split(i, code_split1)
                # 在 code_split1的list两两之间插入分隔符.
                code_split2 = []
                for idx in len(code_split1):
                    code_split2.append(code_split1[idx])
                    if idx != len(code_split1):
                        code_split2.append(i)

                # print(code_split2)
                code_split1 = code_split2

            except Exception:
                continue
            finally:
                print(code_split2)
        print(code_split2)
        counter = Counter(code_split2)
        print(counter)


if __name__ == '__main__':
    code_process()
