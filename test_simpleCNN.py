# -*- coding: utf-8 -*-

"""
CIFAR-10 Convolutional Neural Networks(CNN) Example
next_batch function is copied from edo's answer
https://stackoverflow.com/questions/40994583/how-to-implement-tensorflows-next-batch-for-own-data
Author : solaris33
Project URL : http://solarisailab.com/archives/2325
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



# ?? ??? ???? ?? next_batch ???? ??? ?????.
def next_batch(num, data, labels):
    '''
    `num` ?? ??? ??? ???? ????? ?????.
    '''
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


# CNN ??? ?????.
class Model:
    def __init__(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        self.y = tf.placeholder(tf.float32, shape=[None, 10])
        self.keep_prob = tf.placeholder(tf.float32)

        # cnn layer1
        W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 64], stddev=5e-2))
        b_conv1 = tf.Variable(tf.constant(0.1, shape=[64]))
        h_conv1 = tf.nn.relu(tf.nn.conv2d(self.x, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

        # Pooling layer1
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        # cnn layer2
        W_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 64, 64], stddev=5e-2))
        b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

        # pooling layer2
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        # cnn layer3
        W_conv3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], stddev=5e-2))
        b_conv3 = tf.Variable(tf.constant(0.1, shape=[128]))
        h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)

        # cnn layer4
        W_conv4 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=5e-2))
        b_conv4 = tf.Variable(tf.constant(0.1, shape=[128]))
        h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, W_conv4, strides=[1, 1, 1, 1], padding='SAME') + b_conv4)

        # cnn layer5
        W_conv5 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=5e-2))
        b_conv5 = tf.Variable(tf.constant(0.1, shape=[128]))
        h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4, W_conv5, strides=[1, 1, 1, 1], padding='SAME') + b_conv5)

        # fc
        W_fc1 = tf.Variable(tf.truncated_normal(shape=[8 * 8 * 128, 32], stddev=5e-2))
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[32]))

        h_conv5_flat = tf.reshape(h_conv5, [-1, 8 * 8 * 128])
        self.h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1) #32

        #pearson_mat_tf= tf.contrib.metrics.streaming_pearson_correlation(tf.transpose(self.h_fc1))
        tmp= tf.transpose(self.h_fc1)
        for i in range(32):
            for j in range(32-i):
                


        # Fully Connected Layer 2 - 384?? ???(feature)? 10?? ???-airplane, automobile, bird...-? ??(maping)???.
        W_fc2 = tf.Variable(tf.truncated_normal(shape=[32, 10], stddev=5e-2))
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
        self.logits = tf.matmul(self.h_fc1, W_fc2) + b_fc2
        self.y_pred = tf.nn.softmax(self.logits)

        #loss train
        # Cross Entropy? ????(loss function)?? ????, RMSPropOptimizer? ???? ?? ??? ??????.
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits))
        self.train_step = tf.train.RMSPropOptimizer(1e-3).minimize(self.loss)

        #acc
        self.correct_prediction = tf.equal(tf.argmax(self.y_pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


m= Model()

# CIFAR-10 data load
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# scalar 0~9 --> One-hot Encoding
y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10), axis=1)
y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 10), axis=1)


plot_test_acc=[]
plot_corr_mean=[]
plot_dead_relu=[]

with tf.Session() as sess:
    # ?? ???? ?????.
    sess.run(tf.global_variables_initializer())

    # 10000 Step?? ???? ?????.
    for i in range(10000):
        batch = next_batch(128, x_train, y_train_one_hot.eval())

        # 100 Step?? training ????? ?? ???? loss? ?????.
        if i % 100 == 0:
            train_accuracy = m.accuracy.eval(feed_dict={m.x: batch[0], m.y: batch[1], m.keep_prob: 1.0})
            loss_print = m.loss.eval(feed_dict={m.x: batch[0], m.y: batch[1], m.keep_prob: 1.0})


            test_batch = (x_test, y_test_one_hot.eval())
            test_h_fc1 = m.h_fc1.eval(feed_dict={m.x: test_batch[0], m.y: test_batch[1], m.keep_prob: 1.0})
            test_acc =m.accuracy.eval(feed_dict={m.x: test_batch[0], m.y: test_batch[1], m.keep_prob: 1.0})

            pearson_mat= m.pearson_mat_tf.eval(feed_dict={m.x: test_batch[0], m.y: test_batch[1], m.keep_prob: 1.0})

            #print("                           ",pearson_mat.shape)

            pearson_flat=pearson_mat.flatten()
            original_len= len(pearson_flat)

            #delete nan
            pearson_flat_nnan = [x for x in pearson_flat if str(x) != 'nan']
            nnan_len= len(pearson_flat_nnan)
            #print(pearson_flat_nnan)

            plot_dead_relu.append( (original_len-nnan_len)/original_len )

            corr_mean= np.absolute(pearson_flat_nnan).mean()

            original_len
            print("[Epoch %d]  train_acc: %f, loss: %f, test_acc: %f, corr_mean: %f" % (i, train_accuracy, loss_print, test_acc, corr_mean))
            plot_test_acc.append(test_acc)
            plot_corr_mean.append(corr_mean)

            print("len:", original_len, nnan_len)
            print(pearson_mat)
        # train with Dropout
        sess.run(m.train_step, feed_dict={m.x: batch[0], m.y: batch[1], m.keep_prob: 0.8})

    test_accuracy = 0.0
    for i in range(10):
        test_batch = next_batch(1000, x_test, y_test_one_hot.eval())
        test_accuracy = test_accuracy + m.accuracy.eval(feed_dict={m.x: test_batch[0], m.y: test_batch[1], m.keep_prob: 1.0})
    test_accuracy = test_accuracy / 10;
    print("test_acc: %f" % test_accuracy)

    """
    plot_test_acc=[]
    plot_corr_mean=[]
    plot_dead_relu=[]
    """

    idx=[]
    for i in range(100):
        idx.append(i)

    plt.title("Plot")
    plt.plot(idx, plot_test_acc, "r.-", idx, plot_corr_mean, "g.-", plot_dead_relu, "b.-")
    plt.show()
