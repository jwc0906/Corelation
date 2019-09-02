import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from resnet_model_3 import Model
import time

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

batch_size = 128

def normalize(data):
    data = (data - 128) / 128
    return data

class c10_train_iter():
    def __init__(self, data_x, data_y):
        self.x = normalize(data_x)
        self.y = np.zeros([y_train.shape[0],10])
        self.y[np.arange(data_y.shape[0]), data_y[:, 0]] = 1
        self.pos = 0

    def shuffle(self):
        idx = np.arange(0, len(self.x))
        np.random.shuffle(idx)

        data_shuffle = [self.x[i] for i in idx]
        labels_shuffle = [self.y[i] for i in idx]

        self.x = np.asarray(data_shuffle)
        self.y = np.asarray(labels_shuffle)

    def iternext(self):
        if self.pos + 128 >= self.x.shape[0]:
            x_size, y_size = list(self.x.shape), list(self.y.shape)
            x_size[0], y_size[0] = batch_size, batch_size
            x_iter, y_iter = np.empty(x_size, dtype=float), np.empty(y_size, dtype=float)
            x_iter[:self.x.shape[0]-self.pos] = self.x[self.pos:self.x.shape[0]]
            x_iter[self.x.shape[0]-self.pos:] = self.x[:batch_size - self.x.shape[0] + self.pos]
            y_iter[:self.y.shape[0] - self.pos] = self.y[self.pos:self.y.shape[0]]
            y_iter[self.y.shape[0] - self.pos:] = self.y[:batch_size - self.y.shape[0] + self.pos]
            self.pos += (128 - self.x.shape[0])
            return x_iter, y_iter
        else:
            x_iter, y_iter = self.x[self.pos:self.pos + 128], self.y[self.pos:self.pos + 128]
            self.pos += 128
            return x_iter, y_iter



#time
start_time = time.time()
# CIFAR-10 data load
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
cifar_train = c10_train_iter(x_train, y_train)
cifar_train.shuffle()

x_test = normalize(x_test)

y_test_one_hot = np.zeros([y_test.shape[0],10])
y_test_one_hot[np.arange(y_test.shape[0]), y_test[:,0]] = 1

sess = tf.Session()

iter = 64000


total_batch = int(x_train.shape[0] / batch_size)

m = Model(sess)

sess.run(tf.global_variables_initializer())



plot_test_acc = []
plot_corr_mean = []
plot_dead = []
plot_loss = []


print("--- start: %s seconds ---" %(time.time() - start_time))

for i in range(iter):
    if i==0:
        start_time = time.time()
    x_batch, y_batch = cifar_train.iternext()
    loss, _ = m.train(x_batch, y_batch)

    if i % 500 == 499:
        test_acc, lout = m.get_accuracy(x_test, y_test_one_hot)
        stdev = np.std(lout, 0)
        ind_zero = np.where(stdev==0)[0]

        lout = np.delete(lout, (ind_zero), 1)
        cov = np.cov(lout.T)
        cov_mean = cov.mean()
        corr = np.abs(np.corrcoef(lout.T))

        mean_corr = 0
        div = 0

        for j in range(corr.shape[0]):
            mean_corr += corr[j][j+1:].sum()
            div += (corr.shape[0]-j-1)
        mean_corr /= div


        plot_dead.append(len(ind_zero)/lout.shape[0])
        plot_corr_mean.append(mean_corr)
        plot_test_acc.append(test_acc)
        plot_loss.append(loss)

        print('Iteration {} - loss: {:.5}, test_acc: {:.5}, corr: {:.5}, cov: {:.5}, dead: {}, time:{:.5}'.format(i+1, loss, test_acc, mean_corr, cov_mean, len(ind_zero)/lout.shape[0], (time.time() - start_time)))

        start_time = time.time()

plt.title("Plot")
plt.plot(np.arange(len(plot_dead)), plot_test_acc, "r.-", np.arange(len(plot_dead)), plot_dead, "k.-", np.arange(len(plot_dead)), plot_loss, "y.-", np.arange(len(plot_dead)), plot_corr_mean, "g.-")
plt.show()
