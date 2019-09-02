import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from resnet_model_2 import Model
import time

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True


learning_rate = 0.001


def normalize(data):
    data = (data - 128) / 128
    return data


def shuffle(data, labels):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)

    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


#time
start_time = time.time()
# CIFAR-10 data load
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, y_train = shuffle(x_train, y_train)

y_train_one_hot = np.zeros([y_train.shape[0],10])
y_test_one_hot = np.zeros([y_test.shape[0],10])
y_train_one_hot[np.arange(y_train.shape[0]), y_train[:,0]] = 1
y_test_one_hot[np.arange(y_test.shape[0]), y_test[:,0]] = 1


x_train = normalize(x_train)
x_test = normalize(x_test)

sess = tf.Session()

epoch = 60000
batch_size = 100

total_batch = int(x_train.shape[0] / batch_size)

m = Model(sess, total_batch)

sess.run(tf.global_variables_initializer())



plot_test_acc = []
plot_corr_mean = []
plot_dead = []
plot_loss = []


print("--- start: %s seconds ---" %(time.time() - start_time))
for e in range(epoch):
    if e==0:
        start_time = time.time()
    avg_loss = 0
    for batch in range(total_batch):
        x_batch, y_batch = x_train[batch*batch_size:(batch+1)*batch_size], y_train_one_hot[batch*batch_size:(batch+1)*batch_size]
        c, _ = m.train(x_batch, y_batch)
        avg_loss += c/total_batch

    if e % 1 == 0:

        test_acc, lout = m.get_accuracy(x_test, y_test_one_hot)
        stdev = np.std(lout, 0)
        ind_zero = np.where(stdev==0)[0]

        lout = np.delete(lout, (ind_zero), 1)
        cov = np.cov(lout.T)
        cov_mean = cov.mean()
        corr = np.abs(np.corrcoef(lout.T))

        mean_corr = 0
        div = 0

        for i in range(corr.shape[0]):
            mean_corr += corr[i][i+1:].sum()
            div += (corr.shape[0]-i-1)
        mean_corr /= div

        plot_dead.append(len(ind_zero)/lout.shape[0])
        plot_corr_mean.append(mean_corr)
        plot_test_acc.append(test_acc)
        plot_loss.append(avg_loss)

        print('Epoch {} - loss: {:.5}, test_acc: {:.5}, corr: {:.5}, cov: {:.5}, dead: {}, time:{:.5}'.format(e+1, avg_loss, test_acc, mean_corr, cov_mean, len(ind_zero)/lout.shape[0], (time.time() - start_time)))
        start_time = time.time()

plt.title("Plot")
plt.plot(np.arange(len(plot_dead)), plot_test_acc, "r.-", np.arange(len(plot_dead)), plot_dead, "k.-", np.arange(len(plot_dead)), plot_loss, "y.-", np.arange(len(plot_dead)), plot_corr_mean, "g.-")
plt.show()
