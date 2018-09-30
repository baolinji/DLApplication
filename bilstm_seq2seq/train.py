

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
from bilstm_seq2seq.utils import data2train

FILE_PATH = "data/people/"

hidden_size = 64
layer_num = 3
batch_size = 1
label_num = 5
epoch = 100
timestep = 32


def model(X):
    def lstm_cell():
        lstmcell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0)
        return tf.nn.rnn_cell.DropoutWrapper(lstmcell, input_keep_prob=0.5)

    x_embeding = tf.nn.embedding_lookup(embeding, X)

    # cell = lstm_cell()
    # state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
    # with tf.variable_scope("rnn", reuse=True):
    #     out = list()
    #     for i in range(timestep):
    #         (out_bw, state_fw) = cell(X[:,i,:], state)
    #     out.append(out_bw)

    ##实现
    with tf.variable_scope("bilstm"):
        with tf.variable_scope("lstm_fw", reuse=tf.AUTO_REUSE):
            out_fw = []
            cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(layer_num)])
            state_fw = cell_fw.zero_state(batch_size, dtype=tf.float32)
            for i in range(timestep):
                # x_embeding[:, i, :]:每个instance的第i个时间步长的值,(?,100)
                # out_f:每个步长的输出值,(batch_size,hidden_size)
                (out_f, state_fw) = cell_fw(x_embeding[:, i, :], state_fw)
                # out_fw:所有步长的输出值:(time_steps,batch_size,hidden_size)
                out_fw.append(out_f)
        with tf.variable_scope("lstm_bw", reuse=tf.AUTO_REUSE):
            # inpurt reverse.每个batch的instance逆序
            x_re = tf.reverse(x_embeding, [0])
            out_bw = []
            cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(layer_num)])
            state_bw = cell_bw.zero_state(batch_size, dtype=tf.float32)
            for i in range(timestep):
                (out_b, state_bw) = cell_bw(x_re[:, i, :], state_bw)
                out_bw.append(out_b)
        # 根据instance翻转
        out_bw = tf.reverse(out_bw, [0])
    # 按维度拼接，不破坏batch_size跟time_steps
    outputs = tf.concat([out_fw, out_bw], -1)

    ##调用接口
    # lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for l in range(layer_num)])
    # (outputs_fw, outputs_bw), _ = tf.nn.bidirectional_dynamic_rnn(lstm,
    #                                                               lstm,
    #                                                               x_embeding,
    #                                                               dtype=tf.float32,
    #                                                               time_major=False)
    #
    # outputs_bw = tf.reverse(outputs_bw, axis=[1])
    # outputs = tf.concat([outputs_fw, outputs_bw], -1)

    return tf.reshape(outputs, [-1, 2 * hidden_size])


train,label,word_index = data2train.proess(FILE_PATH, timestep)

train_steps = np.array(train).shape[0] / batch_size

train_dataset = tf.data.Dataset.from_tensor_slices((np.array(train), np.array(label)))
train_dataset = train_dataset.batch(batch_size)
iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
train_initializer = iterator.make_initializer(train_dataset)

n_sample, n_dims = len(word_index) + 1, 50
# print("n_sample:", n_sample, "n_dims:", n_dims)
embeding = tf.Variable(tf.truncated_normal([n_sample, n_dims],
                                           stddev=0.1, dtype=tf.float32))

X, y = iterator.get_next()

out = model(X)
W = tf.Variable(tf.truncated_normal([2 * hidden_size, label_num], stddev=0.1))
b = tf.Variable(tf.constant(1.0, shape=[label_num]))
y_ = tf.matmul(out, W) + b

# flatten label cal loss
y_p = tf.nn.softmax(y_)
y_pred = tf.cast(tf.argmax(y_p, axis=1), tf.int32)
y_label = tf.cast(tf.reshape(y, [-1]), tf.int32)

correct_prediction = tf.equal(y_pred, y_label)
acc1 = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cost1 = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_label, logits=y_))

opt1 = tf.train.AdamOptimizer(0.01).minimize(cost1)


# other cal loss
y_onehot = tf.one_hot(y, label_num, 1.0, 0.0)
y_t = tf.reshape(y_p, [-1, timestep, label_num])  ##还原步长

corr_pred = tf.equal(tf.argmax(y_t, 2), tf.argmax(y_onehot, 2))
acc2 = tf.reduce_mean(tf.cast(corr_pred, tf.float32))
cost2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_t))

opt2 = tf.train.AdagradOptimizer(0.01).minimize(cost2)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(epoch):
    print(i)
    sess.run(train_initializer)
    for j in range(int(train_steps)):
        loss, acc, _ = sess.run([cost2, acc2, opt2])
        if j % 100 == 0:
            print("loss:", loss, "acc:", acc)
            print("------------")

sess.close()