# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

# configuration
#                        O * W + b -> 10 labels for each image, O[? 28], W[28 10], B[10]
#                       ^ (O: output 28 vec from 28 vec input)
#                       |
#      +-+  +-+       +--+
#      |1|->|2|-> ... |28| time_step_size = 28
#      +-+  +-+       +--+
#       ^    ^    ...  ^
#       |    |         |
# img1:[28] [28]  ... [28]
# img2:[28] [28]  ... [28]
# img3:[28] [28]  ... [28]
# ...
# img128 or img256 (batch_size or test_size 256)
#      each input size = input_vec_size=lstm_size=28

# configuration variables
input_vec_size = 28  # 输入向量的维度
hidden_size = 128  # 隐藏层神经元个数
time_step_size = 28  # 循环层长度

batch_size = 128
test_size = 256

def init_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01))


def model(X, W, B, lstm_size):
    # X, input shape: (batch_size, time_step_size, input_vec_size)
    # XT shape: (time_step_size, batch_size, input_vec_size)
    XT = tf.transpose(X, [1, 0, 2])  # permute time_step_size and batch_size,[28, 128, 28]

    # XR shape: (time_step_size * batch_size, input_vec_size)
    XR = tf.reshape(XT, [-1, input_vec_size])  # each row has input for each lstm cell (lstm_size=input_vec_size)

    # Each array shape: (batch_size, input_vec_size)
    X_split = tf.split(XR, time_step_size,
                       0)  # split them to time_step_size (28 arrays),shape = [(128, 28),(128, 28)...]

    # Make lstm with lstm_size (each input vector size). num_units=lstm_size; forget_bias=1.0
    def lstm_cell():
        lstm = rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
        lstm = rnn.DropoutWrapper(lstm, output_keep_prob=0.5)
        return lstm

    def gru_cell():
        gru = tf.nn.rnn_cell.GRUCell(hidden_size)
        return gru

    fw_lstm = rnn.MultiRNNCell([lstm_cell() for _ in range(3)], state_is_tuple=True)
    init_state = fw_lstm.zero_state(hidden_size, dtype=tf.float32)

    bw_lstm = rnn.MultiRNNCell([lstm_cell() for _ in range(3)],state_is_tuple=True)
    ((outputs_fw, outputs_bw), (outputs_state_fw, outputs_state_bw)) = tf.nn.bidirectional_dynamic_rnn(
                                                       gru_cell(),gru_cell(),x,
                                                       dtype = tf.float32,
                                                       time_major=False)
    print(outputs_state_fw.get_shape())
    print(outputs_bw.get_shape())
    outputs = tf.concat([outputs_fw, outputs_bw], 2)
    print(outputs.get_shape())
    outputs = tf.transpose(outputs,[1,0,2])
    # Linear activation
    # Get the last output
    return tf.matmul(outputs[-1], W) + B, bw_lstm.state_size # State size to initialize the stat

mnist = input_data.read_data_sets("data/", one_hot=True)  # 读取数据

# mnist.train.images是一个55000 * 784维的矩阵, mnist.train.labels是一个55000 * 10维的矩阵

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

x = tf.reshape(X, [-1, 28, 28])
W = init_weights([2*hidden_size, 10])  # 输出层权重矩阵28×10
B = init_weights([10])  # 输出层bais

py_x, state_size = model(x, W, B, hidden_size)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.AdagradOptimizer(0.015).minimize(cost)
predict_op = tf.equal(tf.argmax(py_x, 1), tf.argmax(Y, 1))
acc = tf.reduce_mean(tf.cast(predict_op, dtype=tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter("logs", sess.graph)
for i in range(1000):
    x_, y_ = mnist.train.next_batch(batch_size)
    _, loss, accracy = sess.run([train_op, cost, acc], feed_dict={X: x_, Y: y_})
    print(loss, accracy)

sess.close()
