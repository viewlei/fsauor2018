import tensorflow as tf
import os


class Load_model:

    def __init__(self, **kwargs):
        # 模型的参数
        self.lstmUnits = 64
        self.attentionSize = 128
        self.dropOutRate = 1.0
        self.learn_rate = 0.0003
        self.numClasses = 4
        # 数据的参数
        self.maxSeqLength = 350
        self.numDimensions = 300
        self.iterations = kwargs['iter']
        self.label = kwargs['label']
        self.id = kwargs['id']
        self.model_path = './model/label_' + str(self.label) + '/' + str(self.id) + '/'

        if os.path.exists('./model/') is False:
            os.mkdir('./model/')
        if os.path.exists('./model/label_' + str(self.label)) is False:
            os.mkdir('./model/label_' + str(self.label))
        if os.path.exists(self.model_path) is False:
            os.mkdir(self.model_path)

        self.graph = self.init_graph()

    def init_graph(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.variable_scope('Inputs'):
                labels = tf.placeholder(tf.float32, [None, self.numClasses], name='labels')
                input_data = tf.placeholder(tf.float32, [None, self.maxSeqLength, self.numDimensions],
                                            name='input_data')
                input_length = tf.placeholder(tf.int32, [None], name='input_length')

            with tf.variable_scope('BiLSTM'):
                encoder_fw = tf.nn.rnn_cell.BasicLSTMCell(self.lstmUnits)
                encoder_fw = tf.nn.rnn_cell.DropoutWrapper(cell=encoder_fw,
                                                           output_keep_prob=self.dropOutRate)

                encoder_bw = tf.nn.rnn_cell.BasicLSTMCell(self.lstmUnits)
                encoder_bw = tf.nn.rnn_cell.DropoutWrapper(cell=encoder_bw,
                                                           output_keep_prob=self.dropOutRate)

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

                # 初始化选择方法：https://blog.csdn.net/fanzonghao/article/details/82851327?utm_source=blogxgwz1
                # Xavier initialization
                # W = tf.Variable(np.random.randn(node_in, node_out).astype('float32'))/np.sqrt(node_in)
                # 对于relu激活，He initialization
                # W = tf.Variable(np.random.randn(node_in,node_out)) / np.sqrt(node_in/2)
                w_att = tf.Variable(
                    tf.random_normal([1, 1, 2 * self.lstmUnits, self.attentionSize],
                                     stddev=0.1
                                     ), name='w_att'
                )

                b_att = tf.Variable(tf.random_normal([self.attentionSize], stddev=0.1), name='b_att')

                u_att = tf.Variable(
                    tf.random_normal([1, 1, self.attentionSize, 1],
                                     stddev=0.1
                                     ), name='u_att'
                )

                v_att = tf.tanh(
                    tf.nn.conv2d(att_in, w_att, strides=[1, 1, 1, 1],
                                 padding='SAME')
                    + b_att
                )

                betas = tf.nn.conv2d(v_att, u_att, strides=[1, 1, 1, 1], padding='SAME')

                exp_betas = tf.reshape(tf.exp(betas), [-1, self.maxSeqLength])

                alphas = exp_betas / tf.reshape(tf.reduce_sum(exp_betas, 1), [-1, 1])

                last = tf.reduce_sum(output * tf.reshape(alphas, [-1, self.maxSeqLength, 1]), 1)

            with tf.variable_scope('Prediction'):
                recall_matrix = tf.Variable(tf.zeros([4, 4], dtype=tf.float64, name='recall_matrix'))

                acc_matrix = tf.Variable(tf.zeros([4, 4], dtype=tf.float64, name='acc_matrix'))

                weight = tf.Variable(
                    tf.random_normal([2 * self.lstmUnits, self.numClasses],
                                     stddev=0.1),
                    name='weight'
                )

                bias = tf.Variable(tf.random_normal([self.numClasses], stddev=0.1), name='bias')

                y_ = (tf.matmul(last, weight) + bias)  # 32,4
                prediction = tf.argmax(y_, 1)
                correctPred = tf.equal(prediction, tf.argmax(labels, 1))
                accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

                # 计算混淆矩阵
                confusion_matrix = tf.confusion_matrix(
                    labels=tf.argmax(labels, 1),  # 需要加一个 1
                    predictions=prediction,
                    num_classes=4,
                    dtype=tf.int32
                )
                # 横着除是召回率，纵着除是准确率
                col = tf.reduce_sum(confusion_matrix, 1)
                row = tf.reduce_sum(confusion_matrix, 0)

                recall_matrix = tf.add(recall_matrix, confusion_matrix / col)
                acc_matrix = tf.add(recall_matrix, confusion_matrix / row)

                loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=labels)
                )

                tf.summary.scalar('loss', loss)
                tf.summary.scalar('recall of \'-2\'', recall_matrix[0][0])
                tf.summary.scalar('recall of \'-1\'', recall_matrix[1][1])
                tf.summary.scalar('recall of \'-0\'', recall_matrix[2][2])
                tf.summary.scalar('recall of \'1\'', recall_matrix[3][3])
                tf.summary.scalar('accuarcy of \'-2\'', acc_matrix[0][0])
                tf.summary.scalar('accuarcy of \'-1\'', acc_matrix[1][1])
                tf.summary.scalar('accuarcy of \'0\'', acc_matrix[2][2])
                tf.summary.scalar('accuarcy of \'1\'', acc_matrix[3][3])

                merged = tf.summary.merge_all()

            with tf.variable_scope('Optimise'):
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(loss)

        return graph


if __name__ == '__main__':
    kw = {}
    kw['iter'] = 6600
    kw['label'] = 1
    kw['id'] = 111
    model = Load_model(**kw)
    graph = model.graph
