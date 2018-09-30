
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def weight_init(shape):
    weight = tf.truncated_normal(shape, stddev=0.1
                                 )
    return tf.Variable(weight)


def bais_init(shape):
    bais = tf.constant(0.1, shape=shape)
    return tf.Variable(bais)


def conv_init(x, w):
    """

    :param x: input data.shape:[batch_num,hight,weight,channel_szie]
    :param w: filter. shape:[conv_hight,conv_weight,input_channel_szie,output_channel_size]
    :return: filter.shape:
    """
    conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1],
                        padding="SAME",
                        )
    return conv


def max_pooling_2x2init(x):
    max_pooling = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    return max_pooling


def bn_init(x):
    size = x.get_shape().as_list()[3]
    axis = list(range(len(x.get_shape()) - 1))
    x_mean, x_var = tf.nn.moments(x, axis)
    scala = tf.Variable(tf.ones([size]))
    offset = tf.Variable(tf.zeros([size]))
    bn = tf.nn.batch_normalization(x, x_mean, x_var, offset, scala, 0.001)
    return bn


input_data = input_data.read_data_sets('data/', one_hot=True)

sess = tf.InteractiveSession()

x = tf.placeholder("float",shape=[None,784])
y = tf.placeholder("float",shape=[None,10])

x_input = tf.reshape(x, [-1, 28, 28, 1])
w_conv1 = weight_init([3, 3, 1, 128])
b_1 = bais_init([128])

o1_conv = tf.nn.relu(conv_init(x_input, w_conv1) + b_1)
# o1_bn = bn_init(o1_conv)
layer_1 = tf.nn.dropout(o1_conv, keep_prob=0.5)
layer_1_max = max_pooling_2x2init(layer_1)

w_conv2 = weight_init([3, 3, 128, 256])
b_2 = bais_init([256])

o2_conv = tf.nn.relu(conv_init(layer_1_max, w_conv2) + b_2)
# o2_bn = bn_init(o2_conv)
layer_2 = tf.nn.dropout(o2_conv, keep_prob=0.6)
layer_2_max = max_pooling_2x2init(layer_2)

# w_conv3 = weight_init([3, 3, 256, 128])
# b_3 = bais_init([128])
#
# o3_conv = tf.nn.relu(conv_init(layer_2_max, w_conv3))
# # o3_bn = bn_init(o3_conv)
# layer_3 = tf.nn.dropout(o3_conv, keep_prob=0.8)
# layer_3_max = max_pooling_2x2init(layer_3)

l3 = tf.reshape(layer_2_max, [-1, 7 * 7 * 128])

w4_fc = weight_init([7 * 7 * 128, 128])
b_4 = bais_init([128])
o4 = tf.nn.relu(tf.matmul(l3, w4_fc) + b_4)
# o4_drop = tf.nn.dropout(o4,keep_prob=0.5)

w5_fc = weight_init([128, 10])
b_5 = bais_init([10])

y_p = tf.nn.softmax(tf.matmul(o4, w5_fc) + b_5)

print(y_p.get_shape())

cross_entropy = -tf.reduce_mean(y * tf.log(tf.clip_by_value(y_p,1e-10,1.0)))

opt = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.global_variables_initializer())

for i in range(3000):
    x_, y_ = input_data.train.next_batch(1000)
    _, loss, acc = sess.run([opt, cross_entropy, accuracy], feed_dict={x: x_, y: y_})
    print(loss, acc)

print("test accuracy %.3f" % accuracy.eval(feed_dict={
    x: input_data.test.images, y: input_data.test.labels}))
sess.close()
