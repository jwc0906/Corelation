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

from resnet_model import *
import time



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
    for i in range(50000):
        if i==0:
            start_time = time.time()
        batch = next_batch(128, x_train, y_train_one_hot.eval())

        # 100 Step?? training ????? ?? ???? loss? ?????.
        if i % 10 == 0:
            train_accuracy = m.accuracy.eval(feed_dict={m.x: batch[0], m.y: batch[1]})
            loss_print = m.loss.eval(feed_dict={m.x: batch[0], m.y: batch[1]})


            test_batch = (x_test, y_test_one_hot.eval())
            test_middle = m.middle.eval(feed_dict={m.x: test_batch[0], m.y: test_batch[1]})
            test_acc =m.accuracy.eval(feed_dict={m.x: test_batch[0], m.y: test_batch[1]})

            pearson_mat=np.corrcoef(test_middle.T)

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
            print("[Epoch %d]  train_acc: %f, loss: %f, test_acc: %f, corr_mean: %f, time: %f" % (i, train_accuracy, loss_print, test_acc, corr_mean, (time.time() - start_time)))
            start_time = time.time()

            plot_test_acc.append(test_acc)
            plot_corr_mean.append(corr_mean)

            print("len:", original_len, nnan_len)
            print(pearson_mat)
        # train with Dropout
        sess.run(m.train_step, feed_dict={m.x: batch[0], m.y: batch[1]})

    test_accuracy = 0.0
    for i in range(10):
        test_batch = next_batch(1000, x_test, y_test_one_hot.eval())
        test_accuracy = test_accuracy + m.accuracy.eval(feed_dict={m.x: test_batch[0], m.y: test_batch[1]})
    test_accuracy = test_accuracy / 10;
    print("test_acc: %f" % test_accuracy)

    """
    plot_test_acc=[]
    plot_corr_mean=[]
    plot_dead_relu=[]
    """

    idx=[]
    for i in range(500):
        idx.append(i)

    plt.title("Plot")
    plt.plot(idx, plot_test_acc, "r.-", idx, plot_corr_mean, "g.-", plot_dead_relu, "b.-")
    plt.show()
