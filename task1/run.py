import numpy as np
import tensorflow as tf
import pandas as pd
import keras
import os

x = pd.read_csv('./task1_data.csv')
y = pd.read_csv('./task1_label.csv')

x = x.values
y = y.values
y = y.reshape(-1)
categorical = len(set(y))
y = keras.utils.to_categorical(y, categorical)
y = y.astype(np.int)


def batch_iter(x, y, batch_size=32):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


class Model():

    def __init__(self, x, y, batch_size, epoch, learning_rate=0.001):
        self.batch_size = batch_size
        self.epoch = epoch
        self.lr = learning_rate
        self.length = x.shape[1]
        self.clip_grad = 5.
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        # self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(x,y,train_size=0.9,random_state=0)
        self.x_train = x
        self.y_train = y
        self.x_test = x[-3200:, :]
        self.y_test = y[-3200:, :]

        self._build_graph()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        if not os.path.exists('./model'):
            os.mkdir('./model')
        if not os.path.exists('./summary'):
            os.mkdir('./summary')
        self.writer = tf.summary.FileWriter('./summary')

    def _build_graph(self):

        self.x = tf.placeholder(dtype=tf.float32, shape=(None, self.length), name='x')
        self.y = tf.placeholder(dtype=tf.int32, shape=(None, 10), name='y')

        with tf.name_scope('Dense'):
            fc1 = tf.contrib.layers.fully_connected(self.x, 10, activation_fn=None)
            y_hat = tf.contrib.layers.fully_connected(fc1, 10, activation_fn=None)

        self.loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=y_hat))

        self.l2_loss = tf.contrib.layers.apply_regularization(
                        regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                        weights_list=tf.trainable_variables())

        self.loss = self.loss1+self.l2_loss

        self.correct_pred = tf.equal(tf.argmax(self.y, 1), tf.argmax(y_hat, 1))
        self.acc = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        optimizer = tf.train.AdamOptimizer(self.lr)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.clip_grad)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def train(self):
        step = 0
        for e in range(self.epoch):
            self.batch_data_train = batch_iter(self.x_train, self.y_train, self.batch_size)
            print('=========Epoch %d=========' % e)
            for x, y in self.batch_data_train:
                step += 1
                loss, train_op, acc = self.sess.run([self.loss, self.train_op, self.acc],
                                                    feed_dict={self.x: x, self.y: y})
                loss_sum = tf.Summary(value=[tf.Summary.Value(
                    tag="model/loss", simple_value=loss), ])
                self.writer.add_summary(loss_sum, step)

                acc_sum = tf.Summary(value=[tf.Summary.Value(
                        tag="model/acc", simple_value=acc), ])
                self.writer.add_summary(acc_sum, step)
                print('>>> Step%d\'s    Loss:%.4f    ACC:%.4f' % (step, loss, acc))
                filename = os.path.join(
                    'model', "model_{}.ckpt".format(e))
                self.saver.save(self.sess, filename)

    def test(self):
        score = []
        self.batch_data_test = batch_iter(self.x_test, self.y_test, self.batch_size)
        for x, y in self.batch_data_test:
            pred = self.sess.run(self.correct_pred, feed_dict={self.x: x, self.y: y})
            score.append(pred)
        result = []
        for i in score:
            for j in i:
                result.append(j)
        score = np.sum(result) / len(result)
        return score


model = Model(x, y, 32, 1, 0.01)
model.train()
score = model.test()
print()
print('>>> 最后100Batch 平均准确率：', score)
