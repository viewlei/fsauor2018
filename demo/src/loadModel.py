import tensorflow as tf
import gensim
import numpy as np
import jieba
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
model_path = './model/model_s_300_w_5.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
numDimensions = 300
maxSeqLength = 350

modelName = '-6600.meta'
d = {'0': '交通便利性',
     '1': '距离商圈距离',
     '2': '是否容易寻找',
     '3': '排队等候时间',
     '4': '服务人员态度',
     '5': '停车方便性',
     '6': '上菜速度',
     '7': '价格水平',
     '8': '性价比',
     '9': '折扣力度',
     '10': '装修情况',
     '11': '嘈杂情况',
     '12': '就餐空间',
     '13': '卫生情况',
     '14': '饭菜分量',
     '15': '饭菜口感',
     '16': '饭菜外感',
     '17': '推荐程度',
     '18': '本次消费感受',
     '19': '再次消费的意愿',
     }


def getPre(label):
    modelPath = './model/label_' + str(label) + '/'
    jieba.initialize()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess = tf.Session(config=config)
    saver = tf.train.import_meta_graph(modelPath + modelName)
    saver.restore(sess, tf.train.latest_checkpoint(modelPath))
    myGraph = sess.graph

    input_data = myGraph.get_tensor_by_name('Inputs/input_data:0')
    prediction = myGraph.get_tensor_by_name('Prediction/add:0')

    return sess, input_data, prediction, d[str(label)]


def getMatrix(text):
    line = None
    currSeqLength = 0
    for word in jieba.lcut(text):
        try:
            vec = model.get_vector(word).reshape(1, numDimensions)
            if line is None:
                line = vec
            else:
                line = np.append(line, vec, axis=0)
        except KeyError:
            continue
        currSeqLength += 1
        if currSeqLength >= maxSeqLength:
            break
    while currSeqLength < maxSeqLength:
        if line is None:
            line = np.zeros((1, numDimensions))
        else:
            line = np.append(line, np.zeros((1, numDimensions)), axis=0)
        currSeqLength += 1
    return np.expand_dims(line, axis=0)
