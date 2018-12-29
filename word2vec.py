# -*- coding: UTF-8 -*-
from gensim.models import word2vec
import datetime


class wordItor():
    def __init__(self, path):
        self.path = path
        self.content = self.readData()

    def __iter__(self):
        for i in range(len(self.content)):
            # 去除标点
            text = self.content[i]
            # 分词
            if i % 10000 == 0:
                print(i, ":  " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            yield text.split(" ")

    def readData(self):
        file = open(self.path, 'r', encoding='utf-8')
        content = []
        for line in file:
            content.append(line)
        return content


# 读取train中content部分，经过预处理以及分词后，返回一个word的迭代器
sen = wordItor("content.txt")
# 传入迭代器，开始训练
# 4分钟，6遍
size = [300, 380, 400]
for i in size:
    print('start:', i)
    model = word2vec.Word2Vec(sentences=sen, size=i, window=5)
    # save
    model.wv.save_word2vec_format('./model/model_s_' + str(i) + '_w_5.bin', binary=True)
    print('end:', i)
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

'''
model.most_similar('不推荐', topn=10)

'''
