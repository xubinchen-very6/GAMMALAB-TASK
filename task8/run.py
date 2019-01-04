import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
import tensorflow.contrib as tc
import os


x = pd.read_csv('./task8_train_input.csv',header=None)
y = pd.read_csv('./task8_train_output.csv',header=None)
z = pd.read_csv('./task8_test_input.csv',header=None)

x=x.values
y=y.values
y = keras.utils.to_categorical(y,10)
test=z.values
result=z.values[:,::-1]
result = keras.utils.to_categorical(result,10)

print('>>> type x',type(x))
print('>>> type y',type(y))
print('>>> type test',type(test))
print('>>> type result',type(result))

def batch_iter(x,y,batch_size=32):
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

def batch_iter_for_test(x,batch_size=32):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x[start_id:end_id]

class Model():
    def __init__(self,x,y,test,result,epoch,batch_size,lr):
        self.epoch = epoch
        self.batch_size = batch_size
        self.lr = lr
        self.input = x
        self.output = y
        self.test = test
        self.result = result
        self.clip_grad = 5.
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

        self._build_graph()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        if not os.path.exists('./model'):
            os.mkdir('./model')
        if not os.path.exists('./summary'):
            os.mkdir('./summary')
        self.writer = tf.summary.FileWriter('./summary')

            
    def _build_graph(self):
        self.x = tf.placeholder(dtype=tf.int32,shape=(None,20))
        self.y = tf.placeholder(dtype=tf.int32,shape=(None,20,10))
        self.keep_prob = tf.placeholder(dtype=tf.float32)

        self.x_mask = tf.cast(self.x, tf.bool)
        self.length = tf.reduce_sum(tf.cast(self.x_mask, tf.int32), axis=1)
        print('>>> Sequence Length',self.length)

        emb = tf.Variable(tf.random_normal([10,8], mean=0.0, stddev=1.0, dtype=tf.float32))
        enc = tf.nn.embedding_lookup(emb,self.x)
        
        with tf.name_scope('Encoder-layer'):
            seq_length = self.length
            to_dec, state = self.rnn('bi-gru', enc, seq_length, 32)
            to_dec, state = self.rnn('bi-lstm',to_dec, seq_length, 32)
            
        with tf.name_scope('Decoder-layer'):
            dec_output, state = self.rnn('lstm',to_dec,seq_length,20)

        with tf.name_scope('Dense_layer'):
            output=tf.contrib.layers.fully_connected(dec_output,10)

        self.loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=output))
        self.l2_loss = tf.contrib.layers.apply_regularization(
            regularizer=tf.contrib.layers.l2_regularizer(0.0001),
            weights_list=tf.trainable_variables())
        self.loss = self.loss1+self.l2_loss

        self.correct_pred = tf.equal(tf.argmax(self.y, 2), tf.argmax(output, 2))
        self.final = tf.argmax(output,2)
        self.acc = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = self.optimizer.minimize(self.loss)
        optimizer = tf.train.AdamOptimizer(self.lr)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.clip_grad)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))



    def rnn(self,rnn_type, inputs, length, hidden_size, layer_num=1, dropout_keep_prob=None, concat=True):
        if not rnn_type.startswith('bi'):
            cell = self.get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
            outputs, states = tf.nn.dynamic_rnn(cell, inputs, sequence_length=length, dtype=tf.float32)
            if rnn_type.endswith('lstm'):
                c = [state.c for state in states]
                h = [state.h for state in states]
                states = h
        else:
            cell_fw = self.get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
            cell_bw = self.get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                cell_bw, cell_fw, inputs, sequence_length=length, dtype=tf.float32
            )
            states_fw, states_bw = states
            if rnn_type.endswith('lstm'):
                c_fw = [state_fw.c for state_fw in states_fw]
                h_fw = [state_fw.h for state_fw in states_fw]
                c_bw = [state_bw.c for state_bw in states_bw]
                h_bw = [state_bw.h for state_bw in states_bw]
                states_fw, states_bw = h_fw, h_bw
            if concat:
                outputs = tf.concat(outputs, 2)
                states = tf.concat([states_fw, states_bw], 1)
            else:
                outputs = outputs[0] + outputs[1]
                states = states_fw + states_bw
        return outputs, states

    def get_cell(self, rnn_type, hidden_size, layer_num=1, dropout_keep_prob=None):
        cells = []
        for i in range(layer_num):
            if rnn_type.endswith('lstm'):
                cell = tc.rnn.LSTMCell(num_units=hidden_size, state_is_tuple=True)
            elif rnn_type.endswith('gru'):
                cell = tc.rnn.GRUCell(num_units=hidden_size)
            elif rnn_type.endswith('rnn'):
                cell = tc.rnn.BasicRNNCell(num_units=hidden_size)
            else:
                raise NotImplementedError('Unsuported rnn type: {}'.format(rnn_type))
            if dropout_keep_prob is not None:
                cell = tc.rnn.DropoutWrapper(cell,
                                             input_keep_prob=dropout_keep_prob,
                                             output_keep_prob=dropout_keep_prob)
            cells.append(cell)
        cells = tc.rnn.MultiRNNCell(cells, state_is_tuple=True)
        return cells

    def train(self):
        step = 0
        for e in range(self.epoch):
            self.batch_data_train = batch_iter(self.input,self.output,32)
            print('=========Epoch %d=========' % e)
            for x, y in self.batch_data_train:
                step += 1
                loss, train_op, acc = self.sess.run([self.loss, self.train_op, self.acc],
                                                    feed_dict={self.x: x, self.y: y,self.keep_prob: 1.})
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
    
    def test8(self):
        self.saver.restore(self.sess, tf.train.latest_checkpoint('./model'))
        batch_data_test = batch_iter_for_test(model.test,model.batch_size)
        for x in batch_data_test:
            final = model.sess.run(model.final, feed_dict={model.x: x,model.keep_prob: 1.})
            print(final)



model = Model(x,y,test,result,1,32,0.02)
model.train()
model.test8()
score = []

batch_data_test = batch_iter_for_test(model.test,model.result,model.batch_size)
print('ok')
for x, y in batch_data_test:
    final = model.sess.run(model.final, feed_dict={model.x: x, model.y: y,model.keep_prob: 1.})
    score.append(final)
    
# result = []
# for i in score:
#     for j in i:
#         result.append(j)
# score = np.sum(result) / len(result) / 20

np.save('result.npy',score)
result = np.load('result.npy')
final = []
for i in result:
    for j in i:
        final.append(j.tolist())
final = np.array(final)
reve = final[:,::-1].tolist()
totest =[]
for i in reve:
    temp = []
    for j in i:
        if j!=0:
            temp.append(j)
    totest.append(temp)
ground_truth = []

for i in test.tolist():
    temp=[]
    for j in i:
        if j!=0:
            temp.append(j)
    ground_truth.append(temp)
total_ = 0
total = 0
for i,j in zip(ground_truth,totest):
    for ii,jj in zip(i,j):
        total_ += 1
        if ii==jj:
            total+=1
            
print('>>> Test准确率:',total/total_)