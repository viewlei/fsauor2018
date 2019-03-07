# python3
# -*- coding:utf-8 -*-
import linecache
from gensim.models import KeyedVectors
import numpy as np


class Load_batch:

    '''
    如果有bug，考虑是否是while True的问题

    example:
        >> load_batch = Load_batch(method='train', label=0)
        >> seg_itor = load_batch.seg_itor
        >> label_itor = load_batch.label_itor
        >> segMatrix, segLen = next(seg_itor)
        >> labelVec = next(label_itor)
        >> feed_dict = {input: segMatrix, input_length: segLen, label:labelVec}

    '''

    def __init__(self, method='train', label=-1, batchSize=32,
                 maxSeqLength=350, numDimensions=300):
        # 确保传入有效的label
        if -1 == int(label):
            raise KeyError('请输入合法label.')
        linecache.clearcache()  # 清除缓存，防止读取脏数据
        self.numDimensions = numDimensions
        self.maxSeqLength = maxSeqLength
        self.batchSize = batchSize
        if method is 'train':
            self.length = 104999
        else:
            self.length = 14999
        self.content_file_path = './%s/%s_content.txt' % (method, method)
        self.label_file_path = './%s/label/label_%s.txt' % (method, label)
        self.model_path = 'model_s_300_w_5.bin'
        self.model = KeyedVectors.load_word2vec_format(self.model_path, binary=True)
        self.seg_itor = self.get_seg_batch()
        self.label_itor = self.get_label_batch()

    def next(self):
        segMatrix, segLen = next(self.seg_itor)
        labelVec = next(self.label_itor)
        return segMatrix, segLen, labelVec

    def get_seg_batch(self):
        maxSeqLength = self.maxSeqLength
        numDimensions = self.numDimensions
        file_path = self.content_file_path
        idx = 0
        while True:
            if idx + self.batchSize > self.length:
                idx = 0
            start = idx
            idx += self.batchSize
            segMatrix = None
            segLen = []
            for i in range(self.batchSize):
                line = None
                currSeqLength = 0
                for word in linecache.getline(file_path, start + i + 1).split():
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
                if segMatrix is None:
                    segMatrix = line
                else:
                    segMatrix = np.append(segMatrix, line, axis=0)
            # segResult [batchSize, maxSeqLength, depth]
            # segLen (32,)
            yield segMatrix, np.asarray(segLen)

    def get_label_batch(self):
        batchSize = self.batchSize
        file_path = self.label_file_path
        idx = 0
        while True:
            if idx + batchSize > self.length:
                idx = 0
            start = idx
            idx += batchSize
            label = None
            for i in range(batchSize):
                line = [0, 0, 0, 0]
                num = int(linecache.getline(file_path, start + i + 1))
                num += 2
                line[num] += 1
                line = np.asarray(line).reshape(1, 4)
                if label is None:
                    label = line
                else:
                    label = np.append(label, line, axis=0)
            # [batchSize, 4]
            yield label
