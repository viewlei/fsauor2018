# python3
# -*- coding:utf-8 -*-
import linecache
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd

'''

观察下tensorboard中loss的变化，震荡情况是否较 6600随机走的模型好。
注意设置好训练为2轮， 6600 * 32 

'''


class LoadBatch:
    '''
    如果有bug，考虑是否是while True的问题
    '''

    def __init__(self, method='train', label=-1, batchSize=32,
                 maxSegLength=350, numDimensions=300):
        assert -1 < label  # 确保传入有效的label
        linecache.clearcache()  # 清除缓存，防止读取脏数据
        self.numDimensions = numDimensions
        self.maxSegLength = maxSegLength
        self.batchSize = batchSize
        self.method = method
        self.label = label
        if method is 'train':
            self.length = 104999
        else:
            self.length = 14999
        self.content_file_path = './%s/%s_content.txt' % (method, method)
        self.label_file_path = './%s/label/label_%s.txt' % (method, label)
        self.model_path = 'model_s_300_w_5.bin'
        self.model = KeyedVectors.load_word2vec_format(self.model_path, binary=True)
        # index = self.batch_index()
        self.segBatch = self.getSegBatch(index)
        self.labelBatch = self.getLabelBatch(index)
        print('done')

    def next(self):
        senMatrix, senLength = next(self.segBatch)
        labelVec = next(self.labelBatch)
        return senMatrix, senLength, labelVec

    def batch_index(self, num=[1, 1, 2, 3]):
        '''
            训练时，返回一个长度为32 的list，里面每一个值是应该取得的评论的索引。
            min(resut) = 0
            max(result) = 104999
        :return: result
        '''
        label = self.label
        print('111')
        if self.method == 'test':
            return [x for x in range(15000)]
        else:
            data = pd.read_csv('./train/train.csv')
            index = {-2: None, -1: None, 0: None, 1: None}
            lll = data.iloc[:, label + 2]
            for i in range(-2, 2, 1):
                index[i] = list(lll[lll == i].index)
            result = []
            q = [0, 0, 0, 0]  # 记录位置
            tag = [True, True, True, True]  # 记录是否迭代到尽头。迭代完毕后 置为False
            while True:
                if tag[0]:
                    for i in range(num[0]):
                        result.append(index[-2][q[0]])
                        q[0] += 1
                        if q[0] == len(index[-2]):
                            tag[0] = False
                            break

                if tag[1]:
                    for i in range(num[1]):
                        result.append(index[-1][q[1]])
                        q[1] += 1
                        if q[1] == len(index[-1]):
                            tag[1] = False
                            break

                if tag[2]:
                    for i in range(num[2]):
                        result.append(index[0][q[2]])
                        q[2] += 1
                        if q[2] == len(index[0]):
                            tag[2] = False
                            break

                if tag[3]:
                    for i in range(num[3]):
                        result.append(index[1][q[3]])
                        q[3] += 1
                        if q[3] == len(index[1]):
                            tag[3] = False
                            break

                if tag == [False, False, True, True]:
                    break
            return result

    def getSegBatch(self, index_list):
        # index_list 长度为32
        linecache.clearcache()
        maxSeqLength = self.maxSegLength
        numDimensions = self.numDimensions
        file_path = self.content_file_path
        idx = 0
        while True:
            segResult = None
            segLen = []
            if idx + 32 >= self.length:
                idx = 0
            for i in index_list[idx: idx + 32]:
                line = None
                currSeqLength = 0
                for word in linecache.getline(file_path, i + 1).split():
                    try:
                        vec = self.model.get_vector(word).reshape(1, numDimensions)
                        if line is None:
                            line = vec
                        else:
                            line = np.append(line, vec, axis=0)
                    except KeyError:
                        continue
                    currSeqLength += 1
                    if currSeqLength >= maxSeqLength:
                        break
                segLen.append(currSeqLength)
                while currSeqLength < maxSeqLength:
                    line = np.append(line, np.zeros((1, numDimensions)), axis=0)
                    currSeqLength += 1
                line = line.reshape(1, maxSeqLength, numDimensions)
                if segResult is None:
                    segResult = line
                else:
                    segResult = np.append(segResult, line, axis=0)
            # segResult [batchSize, maxSeqLength, depth]
            # segLen (32,)
            idx += 32
            yield segResult, np.asarray(segLen)

    def getLabelBatch(self, index_list):
        linecache.clearcache()
        file_path = self.label_file_path
        idx = 0
        while True:
            labelResult = None
            if idx + 32 >= self.length:
                idx = 0
            for i in index_list[idx:idx + 32]:
                line = [0, 0, 0, 0]
                num = int(linecache.getline(file_path, i + 1))
                num += 2
                line[num] += 1
                line = np.asarray(line).reshape(1, 4)
                if labelResult is None:
                    labelResult = line
                else:
                    labelResult = np.append(labelResult, line, axis=0)
            # [batchSize, 4]
            idx += 32
            yield labelResult


if __name__ == '__main__':
    test = LoadBatch('train', label=15)
    index = test.batch_index()

