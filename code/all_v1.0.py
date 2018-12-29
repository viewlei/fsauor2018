import os
import numpy as np
import tensorflow as tf
from code.loadBatch_with_batch import LoadBatch
from datetime import datetime
from sklearn.metrics import confusion_matrix
import random

startTime = datetime.now()

# 模型的参数
lstmUnits = 64
iterations = 6600
attentionSize = 128
dropOutRate = float(1)
learn_rate = 0.0003
numClasses = 4

# 数据的参数
labelNum = '16'
maxSeqLength = 350
numDimensions = 300
batchSize = 32

# argv = [labelNum, iterations, dropOutRate, learn_rate, batchSize]

'''
labelNum = sys.argv[1]
iterations = int(sys.argv[2])
dropOutRate = float(sys.argv[3])
learn_rate = float(sys.argv[4])
batchSize = int(sys.argv[5])
'''

# tf模型保存的路径
id = int(random.random() * 100)
model_path = './branch/label_' + labelNum + '/model_' + str(id) + '/'

if os.path.exists('branch') is False:
    os.mkdir('branch')
work_path = model_path
if os.path.exists(work_path) is False:
    os.mkdir(work_path)
logFile = open('./branch/label_' + labelNum + '/with_len.txt', 'w', encoding='utf-8-sig')

myGraph = tf.Graph()
with myGraph.as_default():
    with tf.variable_scope('Inputs'):
        labels = tf.placeholder(tf.float32, [None, numClasses], name='labels')
        input_data = tf.placeholder(tf.float32, [None, maxSeqLength, numDimensions], name='input_data')
        input_length = tf.placeholder(tf.int32, [batchSize], name='input_length')

    with tf.variable_scope('BiLSTM'):
        encoder_fw = tf.nn.rnn_cell.BasicLSTMCell(lstmUnits)
        encoder_fw = tf.nn.rnn_cell.DropoutWrapper(cell=encoder_fw,
                                                   output_keep_prob=dropOutRate)

        encoder_bw = tf.nn.rnn_cell.BasicLSTMCell(lstmUnits)
        encoder_bw = tf.nn.rnn_cell.DropoutWrapper(cell=encoder_bw,
                                                   output_keep_prob=dropOutRate)

        (
            (encoder_fw_output, encoder_bw_output),
            (encoder_fw_state, encoder_bw_state)
        ) = (
            tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_fw,
                                            cell_bw=encoder_bw,
                                            inputs=input_data,
                                            sequence_length=input_length,
                                            dtype=tf.float32)
        )

        output = tf.concat((encoder_fw_output, encoder_bw_output), 2)

    with tf.variable_scope('Attention'):
        att_in = tf.expand_dims(output, axis=2)

        w_att = tf.Variable(
            tf.random_normal([1, 1, 2 * lstmUnits, attentionSize],
                             stddev=0.1
                             ), name='w_att'
        )

        b_att = tf.Variable(tf.random_normal([attentionSize], stddev=0.1), name='b_att')

        u_att = tf.Variable(
            tf.random_normal([1, 1, attentionSize, 1],
                             stddev=0.1
                             ), name='u_att'
        )

        v_att = tf.tanh(
            tf.nn.conv2d(att_in, w_att, strides=[1, 1, 1, 1],
                         padding='SAME')
            + b_att
        )

        betas = tf.nn.conv2d(v_att, u_att, strides=[1, 1, 1, 1], padding='SAME')

        exp_betas = tf.reshape(tf.exp(betas), [-1, maxSeqLength])

        alphas = exp_betas / tf.reshape(tf.reduce_sum(exp_betas, 1), [-1, 1])

        last = tf.reduce_sum(output * tf.reshape(alphas, [-1, maxSeqLength, 1]), 1)

    with tf.variable_scope('Prediction'):
        weight = tf.Variable(
            tf.random_normal([2 * lstmUnits, numClasses],
                             stddev=0.1),
            name='weight'
        )

        bias = tf.Variable(tf.random_normal([numClasses], stddev=0.1), name='bias')

        prediction = (tf.matmul(last, weight) + bias)

        correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))

        accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels)
        )
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)

    with tf.variable_scope('Optimise'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(loss)

    saver = tf.train.Saver(max_to_keep=1)

logFile.write('================================\n')
logFile.write('当前时间:' + str(datetime.now()) + '\n')
logFile.write('模型加载完毕...in ' + str(datetime.now() - startTime) + '.\n')
logFile.write('当前参数为：\n')
logFile.write('label:' + labelNum + '\n')
logFile.write('id:' + str(id) + '\n')
logFile.write('迭代次数:' + str(iterations) + '\n学习率:' + str(learn_rate) + '\nDropOutRate:' + str(dropOutRate) + '\n')
logFile.write('batchSize:' + str(batchSize) + '\n')
logFile.write('lstmUnits:' + str(lstmUnits) + '\n')
logFile.write('attentionSize:' + str(attentionSize) + '\n')
logFile.write('maxSeqLength:' + str(maxSeqLength) + '\n')
logFile.write('numDimensions:' + str(numDimensions) + '\n')
logFile.write('当前路径为:' + os.getcwd() + '\n')
logFile.write('开始训练...\n')
logFile.flush()

startTime = datetime.now()
logFile.write(str(startTime) + '\n')
modelName = '-' + str(int((iterations + 1) / 300) * 300) + '.meta'

data = LoadBatch(method='train', label=16)
with tf.Session(graph=myGraph,
                config=tf.ConfigProto(
                    device_count={'CPU': 32},
                    inter_op_parallelism_threads=1,
                    intra_op_parallelism_threads=1,
                )) as sess:
    sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(logdir=work_path, graph=sess.graph)

    for i in range(iterations + 1):
        # time_1 = datetime.now()
        aaa, bbb, ccc = data.next()
        sess.run([optimizer], feed_dict={input_data: aaa, input_length: bbb, labels: ccc})
        # Write summary to Tensorboard
        if i % 100 == 0 and i != 0:
            summary = sess.run(merged, feed_dict={input_data: aaa, input_length: bbb, labels: ccc})
            writer.add_summary(summary, i)
            # Save the network every 10 training iterations
        if i % 300 == 0 and i != 0:
            # logFile.write(str(i) + '\t' + str(los) + '\t' + str(datetime.now()) + '\n')
            # logFile.flush()
            save_path = saver.save(sess, work_path, global_step=i)
            # print('saved to %s ' % save_path, datetime.now())
            # time_2 = datetime.now()
        print(i)
        writer.close()

endTime = datetime.now()
logFile.write('训练完毕，共耗时' + str(endTime - startTime) + '\n')
logFile.write('模型保存路径为：' + model_path + '\n')
logFile.write('当前时间:' + str(datetime.now()) + '\n')
logFile.flush()

# 开始测试
# 首先是训练集
iterations = 6600
logFile.write('现在开始训练集的测试...\n')


def check(res):
    # re.shape = (4,4)
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


def log(acc, recall):
    logFile.write('结果得分为:\n')
    logFile.write('label \tacc \trecall f1\n')
    sum = 0
    for r in range(4):
        f = 2 / (float(1 / acc[r]) + float(1 / recall[r]))
        sum += f
        logFile.write(str(r - 2) + ': \t' + str(acc[r]) + ' \t' + str(recall[r]) + ' \t' + str(f) + '\n')
    logFile.write('Total F1:' + str(float(sum / 4)) + '\n')


sess = tf.Session()
saver = tf.train.import_meta_graph(model_path + modelName)
saver.restore(sess, tf.train.latest_checkpoint(model_path))
myGraph = sess.graph
input_data = myGraph.get_tensor_by_name('Inputs/input_data:0')
input_length = myGraph.get_tensor_by_name('Inputs/input_length:0')
labels = myGraph.get_tensor_by_name('Inputs/labels:0')
prediction = myGraph.get_tensor_by_name('Prediction/add:0')

data = LoadBatch(method='train', label=16)

tmp = None
for i in range(iterations):
    aaa, bbb, ccc = data.next()
    res = sess.run(prediction, feed_dict={input_data: aaa, input_length: bbb, labels: ccc})
    label = np.argmax(ccc, 1)
    res = np.argmax(res, 1)
    f_m = confusion_matrix(label, res, labels=[0, 1, 2, 3])
    # print(i)
    if tmp is None:
        tmp = f_m
    else:
        tmp = tmp + f_m

logFile.write('训练集测试完毕，开始分析结果...\n')
accRes, recallRes = check(tmp)
log(accRes, recallRes)
logFile.write('结果打印完毕, 当前时间为:' + str(datetime.now()) + '\n')
logFile.flush()

# 接着开始验证集
logFile.write('现在开始验证集的测试...\n')
sess = tf.Session()
saver = tf.train.import_meta_graph(model_path + modelName)
saver.restore(sess, tf.train.latest_checkpoint(model_path))
myGraph = sess.graph
input_data = myGraph.get_tensor_by_name('Inputs/input_data:0')
input_length = myGraph.get_tensor_by_name('Inputs/input_length:0')
labels = myGraph.get_tensor_by_name('Inputs/labels:0')
prediction = myGraph.get_tensor_by_name('Prediction/add:0')

data = LoadBatch(method='test', label=16)
tmp = None
for i in range(468):
    aaa, bbb, ccc = data.next()
    res = sess.run(prediction, feed_dict={input_data: aaa, input_length: bbb, labels: ccc})
    label = np.argmax(ccc, 1)
    res = np.argmax(res, 1)
    f_m = confusion_matrix(label, res, labels=[0, 1, 2, 3])
    if tmp is None:
        tmp = f_m
    else:
        tmp = tmp + f_m

logFile.write('运行完毕，开始分析结果...\n')
accRes, recallRes = check(tmp)
log(accRes, recallRes)
logFile.write('结果打印完毕, 当前时间为:' + str(datetime.now()) + '\n')
logFile.write('================================')
logFile.close()
