#Deep activities recognition model

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np 
import tensorflow as tf
import os
import pandas as pd
from scipy import stats
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from MLP import HiddenLayer, MLP
from logisticRegression2 import LogisticRegression 
from rbm_har import  RBM,GRBM
import math
import timeit
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import timeit

class Dataset(object):
    def __init__(self, segments, labels, one_hot = False, dtype = dtypes.float32, reshape = True):
        """Construct a Dataset
        one_hot arg is used only if fake_data is True. 'dtype' can be either unit9 or float32
        """

        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid')

        self._num_examples = segments.shape[0]
        self._segments = segments
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def segments(self):
        return self._segments

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next batch-size examples from this dataset"""

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed +=1

            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._segments = self._segments[perm]
            self._labels = self._labels[perm]

            #start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._segments[start:end,:, :], self._labels[start:end,:]



def windows(data, size):
    start = 0
    while start < data.count():
        yield start, start + size
        start += size 

def segment_signal(data, window_size = 1):
    segments = np.empty((0, window_size, 41))
    labels = np.empty((0))
    num_features = ["duration", "protocol_type","service","flag","src_bytes", "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_srv_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate"
    ]

    pca = PCA(n_components = 20)

    Y = pca.fit_transform(data[num_features])
    segments = np.asarray(Y)
    segments = np.asarray(data[num_features].copy())
    labels = data["label"]

    return segments, labels

def read_data(filename):
    col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_srv_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label", "successfulPrediction"]
    dataset = pd.read_csv(filename, header = None, names = col_names)
    return dataset      


def read_data_set(dataset1, dataset2, one_hot = False, dtype = dtypes.float32, reshape = True):
    
    segments1, labels1 = segment_signal(dataset1)
    #labels1 = np.asarray(pd.get_dummies(labels1), dtype = np.int8)

    segments2, labels2 = segment_signal(dataset2)
    #labels2 = np.asarray(pd.get_dummies(labels2), dtype = np.int8)
    labels = np.asarray(pd.get_dummies(labels1.append([labels2])), dtype = np.int8)
    labels1 = labels[:len(labels1)]
    labels2 = labels[len(labels1):]
    train_x = segments1.reshape(len(segments1), 1, 1 ,41)
    train_y = labels1

    test_x = segments2.reshape(len(segments2), 1, 1 ,41)
    test_y = labels2

    train = Dataset(train_x, train_y, dtype = dtype , reshape = reshape)
    test = Dataset(test_x, test_y, dtype = dtype, reshape = reshape)
    return base.Datasets(train = train, validation=None, test = test)


class DBN(object):
    """Deep belief network
    A deep belief network is obtained by stacking several RBMs on top of the each other.
    The hidden layer of the RBM at layer 'i' becomes the input of the RBM at layer 'i+1'.
    The first layer RBM gets as input the input of the network, and the hidden layer of
    the last RBM represents the output. When used for classification, the DBN is treated
    as a MLP, by adding a logistic regression layer on top.
    """

    def __init__(self, n_inp = 784, n_out = 10, hidden_layer_sizes = [500, 500]):
        """ This class is made to support a variable number of layers.

        :param n_inps: int, dimension of the input to the DBN
        :param n_outs: int, demension of the output of the network
        :param hidden_layer_sizes: list of ints, intermediate layers size, must contain
        at least one value
        """

        self.sigmoid_layers = []
        self.layers = []
        self.params = []
        self.n_layers = len(hidden_layer_sizes)

        assert self.n_layers > 0

        #define the grape
        height, weight, channel = n_inp
        self.x = tf.placeholder(tf.float32, [None, height, weight, channel])
        self.y = tf.placeholder(tf.float32, [None, n_out])

        for i in range(self.n_layers):
            # Construct the sigmoidal layer

            # the size of the input is either the number of hidden units of the layer
            # below or the input size if we are on the first layer

            if i == 0:
                input_size = height * weight *channel
            else:
                input_size = hidden_layer_sizes[i - 1]

            # the input to this layer is either the activation of the hidden layer below
            # or the input of the DBN if you are on the first layer
            if i == 0:
                layer_input = tf.reshape(self.x, [-1, height*weight*channel])

            else:
                layer_input = self.sigmoid_layers[-1].output


            sigmoid_layer = HiddenLayer(input = layer_input, n_inp = input_size, 
                n_out = hidden_layer_sizes[i], activation = tf.nn.sigmoid)

            #add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)

            # Its arguably a philosophical question... but we are going to only
            # declare that the parameters of the sigmoid_layers are parameters of the DBN.
            # The visible biases in the RBM are parameters of those RBMs, but not of the DBN

            self.params.extend(sigmoid_layer.params)
            if i == 0:
                rbm_layer = GRBM(inp = layer_input, n_visible = input_size, n_hidden = hidden_layer_sizes[i], W = sigmoid_layer.W, hbias = sigmoid_layer.b) 
            else:
                rbm_layer = RBM(inp = layer_input, n_visible = input_size, n_hidden = hidden_layer_sizes[i], W = sigmoid_layer.W, hbias = sigmoid_layer.b)  
            self.layers.append(rbm_layer)
            
        self.logLayer = LogisticRegression(input= self.sigmoid_layers[-1].output, 
            n_inp = hidden_layer_sizes[-1], n_out = n_out)
        self.params.extend(self.logLayer.params)
        #print(self.sigmoid_layers[-1].output)
        #print(hidden_layer_sizes[-1], n_out)
        #compute the cost for second phase of training, defined as the cost of the
        # logistic regression output layer

        self.finetune_cost = self.logLayer.cost(self.y)

        #compute the gradients with respect to the model parameters symbolic variable that
        # points to the number of errors made on the minibatch given by self.x and self.y
        self.pred = self.logLayer.pred
        self.accuracy = self.logLayer.accuracy(self.y)
        """
        # Initialize with 0 the weights W as a matrix of shape (n_inp, n_out)
        out_weights = tf.Variable(tf.zeros([hidden_layer_sizes[-1], n_out]))
        # Initialize the biases b as a vector of n_out 0s
        out_biases  = tf.Variable(tf.zeros([n_out]))

        out_ = tf.nn.softmax(tf.matmul(self.sigmoid_layers[-1].output, out_weights) + out_biases)

        self.finetune_cost = -tf.reduce_mean(tf.reduce_sum(self.y * tf.log(out_)))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(self.finetune_cost)

        correct_prediction = tf.equal(tf.argmax(out_,1), tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        """

    def pretraining(self, sess, train_set_x, batch_size = 100, pretraining_epochs = 200, 
        learning_rate = 0.0001, k = 1, display_step = 1):
        """ Generates a list of functions, for performing one step of gradient descent at
        a given layer. The function will require as input the minibatch index, and to train
        an RBM you just need to iterate, calling the corresponding function in all minibatch 
        indexes.
        :param train_set_x: tensor, contains all datapoints used for traing the RBM
        :param batch_size: int, size of a minibatch
        :param k: number of Gibbs steps to do in CD-k/ PCD-k
        :param learning_rate: learning rate
        :param pretraining_epochs: int, maximal number of epochs to run the optimizer
        """

        #begining of a batch, given index
        start_time = timeit.default_timer()
        batch_num = int(train_set_x.train.num_examples / batch_size)

        #Pretraining layer by layer
        for i in range(self.n_layers):
            # Get the cost and the updates list
            #Using CD-k here for training each RBM
            #TODO: change cost function to reconstruction error
            
            #cost = self.layers[i].get_reconstruction_cost()
            #train_ops = self.layers[i].get_train_ops(lr = learning_rate, persistent = None, k = k)
            #print(self.rbm_layers[i].n_visible, self.rbm_layers[i].n_hidden)
            
            if i ==0:
                learning_rate = 0.001
            else:
                learning_rate = 0.001
            
            cost, train_ops = self.layers[i].get_train_ops(lr = learning_rate, persistent = None, k = k)
            #cost = self.layers[i].get_reconstruction_cost()
            for epoch in range(pretraining_epochs):
                avg_cost = 0.0
                for j in range(batch_num):
                    batch_xs, batch_ys = train_set_x.train.next_batch(batch_size)
                    _ = sess.run(train_ops, feed_dict = {self.x : batch_xs,})
                    c = sess.run(cost, feed_dict = {self.x: batch_xs, })
                    avg_cost += c / batch_num
                    
                if epoch % display_step == 0:
                    print("Pretraining layer {0} Epoch {1}".format(i+1, epoch +1) + " cost {:.9f}".format(avg_cost))        
                    
                    #plt.imsave("new_filters_at_{0}.png".format(epoch),tile_raster_images(X = sess.run(tf.transpose(self.rbm_layers[i].W)), img_shape = (28, 28), tile_shape = (10,10), tile_spacing = (1,1)), cmap = 'gray')
                #plt.show()
                
        end_time = timeit.default_timer()
        print("time {0} minutes".format((end_time - start_time)/ 60.))


    def fine_tuning(self, sess, train_set_x, batch_size = 100 , training_epochs = 200 , learning_rate = 0.1, display_step = 1):
        """ Genarates a function train that implements one step of finetuning, a function validate 
        that computes the error on a batch from the validation set, and a function test that computes
        the error on a batch from the testing set

        :param datasets: tensor, a list contain all the dataset
        :param batch_size: int, size of a minibatch
        :param learning_rate: int, learning rate
        :param training_epochs: int, maximal number of epochs to run the optimizer
        """

        start_time = timeit.default_timer()
        train_ops = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(self.finetune_cost, var_list= self.params)

        #Accuracy
        accuracy = self.accuracy
        batch_num = int(train_set_x.train.num_examples/batch_size)
        ACC_max = 0
        pre_max = 0
        rec_max = 0
        for epoch in range(training_epochs):
            avg_cost = 0.0
            for i in range(batch_num):
                b =[]
                d = []
                batch_xs, batch_ys = train_set_x.train.next_batch(batch_size)
                _= sess.run(train_ops, feed_dict = {self.x :batch_xs, self.y : batch_ys} )
                c =sess.run(self.finetune_cost, feed_dict = {self.x :batch_xs, self.y : batch_ys} )
                avg_cost += c / batch_num
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch +1), "cost:", "{:.9f}".format(avg_cost))
                acc = sess.run(accuracy, feed_dict = {self.x: train_set_x.test.segments, self.y: train_set_x.test.labels})
                pr= sess.run(self.pred, feed_dict = {self.x: train_set_x.test.segments, self.y: train_set_x.test.labels})

                b = np.append(b, sess.run(tf.argmax(pr, axis =1)))
                #np.savetxt('b.txt', b ,delimiter=',')
                d = np.append(d , sess.run(tf.argmax(self.y, axis =1 ), feed_dict ={self.x: train_set_x.test.segments, self.y: train_set_x.test.labels}))
                #np.savetxt('d.txt', d ,delimiter=',')
                
                a = confusion_matrix(d, b)
                FP = a.sum(axis=0) - np.diag(a)
                FN = a.sum(axis=1) - np.diag(a)
                TP = np.diag(a)
                TN = a.sum() - (FP + FN + TP)
                ac = (TP + TN) / (TP + FP + FN + TN)
                ACC = ac.sum() / 5
                precision = precision_score(d, b, average='weighted')
                recall = recall_score(d, b, average='weighted')
                if ACC > ACC_max:
                    ACC_max = ACC
                if precision > pre_max:
                    pre_max = precision
                if recall > rec_max:
                    rec_max = recall

                print(ac.sum() / 5)
                print(a)
                print("ACCURACY: {0}, PRECISION: {1}, RECALL: {2}:".format(ACC_max, pre_max, rec_max))
        end_time = timeit.default_timer()
        print("Time {0} minutes".format((end_time- start_time)/ 60.))

def initlabel(dataset):
    labels = dataset['label'].copy()
    #labels[labels != 'normal'] = 'attack'
    labels[labels == 'back'] = 'dos'
    labels[labels == 'buffer_overflow'] = 'u2r'
    labels[labels == 'ftp_write'] =  'r2l'
    labels[labels == 'guess_passwd'] = 'r2l'
    labels[labels == 'imap'] = 'r2l'
    labels[labels == 'ipsweep'] = 'probe'
    labels[labels == 'land'] = 'dos' 
    labels[labels == 'loadmodule'] = 'u2r'
    labels[labels == 'multihop'] = 'r2l'
    labels[labels == 'neptune'] = 'dos'
    labels[labels == 'nmap'] = 'probe'
    labels[labels == 'perl'] = 'u2r'
    labels[labels == 'phf'] =  'r2l'
    labels[labels == 'pod'] =  'dos'
    labels[labels == 'portsweep'] = 'probe'
    labels[labels == 'rootkit'] = 'u2r'
    labels[labels == 'satan'] = 'probe'
    labels[labels == 'smurf'] = 'dos'
    labels[labels == 'spy'] = 'r2l'
    labels[labels == 'teardrop'] = 'dos'
    labels[labels == 'warezclient'] = 'r2l'
    labels[labels == 'warezmaster'] = 'r2l'

    # NSL KDD
    labels[labels == 'mailbomb'] = 'dos'
    labels[labels == 'processtable'] = 'dos'
    labels[labels == 'udpstorm'] = 'dos'
    labels[labels == 'apache2'] = 'dos'
    labels[labels == 'worm'] = 'dos'

    labels[labels == 'xlock'] = 'r2l'
    labels[labels == 'xsnoop'] = 'r2l'  
    labels[labels == 'snmpguess'] = 'r2l'
    labels[labels == 'snmpgetattack'] = 'r2l'
    labels[labels == 'httptunnel'] = 'r2l'
    labels[labels == 'sendmail'] = 'r2l'    
    labels[labels == 'named'] = 'r2l'   

    labels[labels == 'sqlattack'] = 'u2r'
    labels[labels == 'xterm'] = 'u2r'
    labels[labels == 'ps'] = 'u2r'

    labels[labels == 'mscan'] = 'probe'
    labels[labels == 'saint'] = 'probe'
    return labels

def nominal(dataset1, dataset2):
    protocol1 = dataset1['protocol_type'].copy()
    protocol2 = dataset2['protocol_type'].copy()
    dataset = dataset1.append([dataset2])
    protocol_type = dataset['protocol_type'].unique()
    for i in range(len(protocol_type)):
        protocol1[protocol1 == protocol_type[i]] = i
        protocol2[protocol2 == protocol_type[i]] = i
    service1 = dataset1['service'].copy()
    service2 = dataset2['service'].copy()
    service_type = dataset['service'].unique()
    for i in range(len(service_type)):
        service1[service1 == service_type[i]] = i
        service2[service2 == service_type[i]] = i
    flag1 = dataset1['flag'].copy()
    flag2 = dataset2['flag'].copy()
    flag_type = dataset['flag'].unique()
    for i in range(len(flag_type)):
        flag1[flag1 == flag_type[i]] = i
        flag2[flag2 == flag_type[i]] = i
    return protocol1, service1, flag1, protocol2, service2, flag2

if __name__ == "__main__":
    start = timeit.default_timer()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename1 = dir_path + "/datasets/nslkdd/KDDTrain+.csv"
    filename2 = dir_path + "/datasets/nslkdd/KDDTest+.csv"
    dataset1 = read_data(filename1)
    dataset2 = read_data(filename2)

    dataset1['label'] = initlabel(dataset1)

    dataset2['label'] = initlabel(dataset2)

    dataset1['protocol_type'], dataset1['service'], dataset1['flag'],   dataset2['protocol_type'], dataset2['service'], dataset2['flag'] = nominal(dataset1, dataset2)

    print(dataset2['service'].value_counts())


    print(dataset1['protocol_type'].value_counts())
    print(dataset2['flag'].value_counts())

    num_features = ["duration", "protocol_type", "service", "flag", "src_bytes", 
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_srv_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate"
    ]
    dataset1[num_features] = dataset1[num_features].astype(float)
    #dataset1[num_features] = dataset1[num_features].apply(lambda x:MinMaxScaler().fit_transform(x))
    dataset1[num_features] = MinMaxScaler().fit_transform(dataset1[num_features].values)
    #print(dataset.describe())
    

    dataset2[num_features] = dataset2[num_features].astype(float)
    #dataset2[num_features] = dataset2[num_features].apply(lambda x:MinMaxScaler().fit_transform(x))
    dataset2[num_features] = MinMaxScaler().fit_transform(dataset2[num_features].values)

    print(dataset1['protocol_type'].value_counts())
    print(dataset2['flag'].value_counts())

    print(dataset2['label'].value_counts())

    acc = read_data_set(dataset1, dataset2)


    learning_rate = 0.1
    training_epochs = 100
    batch_size = 50
    display_step = 10

    #DBN structure
    n_inp = [1, 1, 41]
    hidden_layer_sizes = [1000, 1000]
    n_out = 5

    dbn = DBN(n_inp = n_inp, hidden_layer_sizes = hidden_layer_sizes, n_out = n_out)

    tf.set_random_seed(seed = 999)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        dbn.pretraining(sess, train_set_x = acc, k =1)
        dbn.fine_tuning(sess, train_set_x = acc)

stop = timeit.default_timer()

print(stop - start )