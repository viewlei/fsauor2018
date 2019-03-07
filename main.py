# python3
# -*- coding:utf-8 -*-
'''
@project: xldqgfx
@author: Erbenner
@file: main.py
@ide: PyCharm
@time: 2019-02-04 19:35:19
@e-mail: jblei@mail.ustc.edu.cn
'''

import argparse
from time import time
from datetime import datetime
from load_model import Load_model
from load_batch import Load_batch
import tensorflow as tf


def train(**kwargs):
    # python main.py --type train --label 2 --iter 6600
    model_id = int(time())
    kwargs['id'] = model_id

    model = Load_model(**kwargs)
    data = Load_batch(method=kwargs['type'], label=kwargs['label'])
    graph = model.graph

    log_file = open('./model/log.txt', 'a', encoding='utf-8-sig')
    init_train_log(log_file, kwargs, model)

    config = tf.ConfigProto(
        device_count={'CPU': 32},
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1,
    )
    config.gpu_options.allow_growth = True

    with tf.Session(graph=graph, config=config) as sess:
        sess.run(tf.global_variables_initializer())
        optimizer = graph.get_operation_by_name('Optimise/Adam')
        input_data = graph.get_tensor_by_name('Inputs/input_data:0')
        input_length = graph.get_tensor_by_name('Inputs/input_length:0')
        labels = graph.get_tensor_by_name('Inputs/labels:0')
        merged = graph.get_tensor_by_name('Prediction/Merge/MergeSummary:0')
        # test
        prediction = graph.get_tensor_by_name('Prediction/ArgMax:0')
        confusion_matrix = graph.get_tensor_by_name('Prediction/confusion_matrix/SparseTensorDenseAdd:0')
        writer = tf.summary.FileWriter(logdir=model.model_path, graph=sess.graph)
        saver = tf.train.Saver(max_to_keep=1)
        for i in range(kwargs['iter'] + 1):
            '''
            每10步，收集一次数据，绘在tensorboard内。
            每100步，存储当前模型到工作路径。
            '''
            segMatrix, segLen, labelVec = data.next()
            feed_dict = {input_data: segMatrix, input_length: segLen, labels: labelVec}
            sess.run([optimizer], feed_dict=feed_dict)
            if i % 10 == 0 and i != 0:
                summary = sess.run(merged, feed_dict=feed_dict)
                writer.add_summary(summary, i)
            if i % 100 == 0 and i != 0:
                save_path = saver.save(sess, model.model_path, global_step=i)
            print('train...' + str(i))
        writer.close()
        log_file.write('train完毕，共耗时：'+str((int(time()) - model_id) / 60) + 'min.')
    log_file.close()
    # 开始测试
    test(**kwargs)

def test(**kwargs):
    # python main.py --type test --label 1 --id 1287471
    from sklearn.metrics import confusion_matrix
    import numpy as np

    log_file = open('./model/log.txt', 'a', encoding='utf-8-sig')
    kwargs['type'] = 'test'
    config = tf.ConfigProto(
        device_count={'CPU': 32},
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1,
    )
    meta_path, meta_name = get_meta_name(**kwargs)
    if meta_name is None:
        raise EOFError('meta文件不存在')

    saver = tf.train.import_meta_graph(meta_path + '/' + meta_name)
    sess = tf.Session(config=config)
    saver.restore(sess, tf.train.latest_checkpoint(meta_path))
    graph = sess.graph
    input_data = graph.get_tensor_by_name('Inputs/input_data:0')
    input_length = graph.get_tensor_by_name('Inputs/input_length:0')
    labels = graph.get_tensor_by_name('Inputs/labels:0')
    prediction = graph.get_tensor_by_name('Prediction/ArgMax:0')
    data = Load_batch(method=kwargs['type'], label=kwargs['label'])
    result = None
    for i in range(468):
        print('test...' + str(i))
        segMatrix, segLen, labelVec = data.next()
        feed_dict = {input_data: segMatrix, input_length: segLen, labels: labelVec}
        predict = sess.run(prediction, feed_dict=feed_dict)
        res = confusion_matrix(np.argmax(labelVec, 1), predict, labels=[0,1,2,3])
        if result is None:
            result = res
        else:
            result += res
    log_file.write('test完毕，开始分析结果...\n')
    # 注意，此处result可能含0导致下一步发生‘除0’错误
    accRes, recallRes = check(result, log_file)
    log(accRes, recallRes, log_file)
    log_file.write('结果打印完毕, 当前时间为:' + str(datetime.now()) + '\n')
    log_file.write('================================')
    log_file.close()

# def total(**kwargs):
#     # python main.py --type total --label 0 --iter 6600
#     model_id = train(**kwargs)
#     kwargs['id'] = model_id
#     test(**kwargs)


def init_train_log(log_file, kwargs, model):
    log_file.write('================================\n')
    log_file.write('当前时间:' + str(datetime.now()) + '\n')
    log_file.write('当前参数为：\n')
    log_file.write('label:' + str(kwargs['label']) + '\n')
    log_file.write('id:' + str(kwargs['id']) + '\n')
    log_file.write('迭代次数:' + str(kwargs['iter']) + '\n学习率:' + str(model.learn_rate) + '\nDropOutRate:' + str(model.dropOutRate) + '\n')
    log_file.write('batchSize:' + str(kwargs['batchSize']) + '\n')
    log_file.write('lstmUnits:' + str(model.lstmUnits) + '\n')
    log_file.write('attentionSize:' + str(model.attentionSize) + '\n')
    log_file.write('maxSeqLength:' + str(model.maxSeqLength) + '\n')
    log_file.write('numDimensions:' + str(model.numDimensions) + '\n')
    log_file.write('开始训练...\n')
    log_file.flush()


def check(res, log_file):
    # re.shape = (4,4)
    import numpy as np

    log_file.write('result:\n')
    log_file.write(str(res) + '\n')
    Actual = np.sum(res, 1)
    Pre = np.sum(res, 0)
    acc = []
    acc.append(float(res[0][0] / Pre[0]))
    acc.append(float(res[1][1] / Pre[1]))
    acc.append(float(res[2][2] / Pre[2]))
    acc.append(float(res[3][3] / Pre[3]))
    recall = []
    recall.append(float(res[0][0] / Actual[0]))
    recall.append(float(res[1][1] / Actual[1]))
    recall.append(float(res[2][2] / Actual[2]))
    recall.append(float(res[3][3] / Actual[3]))
    return acc, recall


def log(acc, recall, log_file):
    log_file.write('结果得分为:\n')
    log_file.write('label \tacc \trecall f1\n')
    sum = 0
    for r in range(4):
        f = 2 / (float(1 / acc[r]) + float(1 / recall[r]))
        sum += f
        log_file.write(str(r - 2) + ': \t' + str(acc[r]) + ' \t' + str(recall[r]) + ' \t' + str(f) + '\n')
    log_file.write('Total F1:' + str(float(sum / 4)) + '\n')


def get_meta_name(**kwargs):
    # 加载模型
    import os
    model_id = kwargs['id']
    model_label = kwargs['label']
    model_path = './model/label_' + str(model_label) + '/' + str(model_id)
    path = os.listdir(model_path)
    for file in path:
        if '.meta' in file:
            return model_path, file
    return model_path, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('--batchSize', type=int, help='batch_size', default=32)
    parser.add_argument('--lr', type=float, help='learning_rate', default=0.0003)
    parser.add_argument('--iter', type=int, help='the number of batch.', nargs='?')
    parser.add_argument('--id', type=int, help='the id of model in testing task.', nargs='?')
    parser.add_argument('--type', type=str, help='train or test?')
    parser.add_argument('--label', type=int, help='label which from 0 to 19.')
    FLAGS = parser.parse_args()

    if FLAGS.type == 'train':
        if 'iter' not in FLAGS:
            raise KeyError('expect the argument \'iter\' while training.')
        if 'id' in FLAGS:
            raise KeyError('too many argument while training: id.')
        train(**vars(FLAGS))
    elif FLAGS.type == 'test':
        if 'iter' in FLAGS:
            raise KeyError('too many argument while testing: iter.')
        if 'id' not in FLAGS:
            raise KeyError('expect the argument \'id\' while testing.')
        test(**vars(FLAGS))
    else:
        raise KeyError('Illegal Argument!')
