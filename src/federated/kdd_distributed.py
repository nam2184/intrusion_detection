#Deep activities recognition model

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from mlxtend.preprocessing import one_hot
import argparse
import random
import sys
import time
#from tensorflow.contrib import slim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from MLP import HiddenLayer, MLP
from logisticRegression2 import LogisticRegression 
from rbm import  RBM,GRBM
import os
import pandas as pd
from scipy import stats
#from tensorflow.compat.v1.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
import math
import timeit
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#----------distributed------------------------
#define cluster
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
parameter_servers = ["localhost:2222"]
workers = [ "localhost:2223", "localhost:2224"]
cluster = tf.compat.v1.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

# Input Flags
tf.compat.v1.app.flags.DEFINE_string("job_name", "", "'ps' / 'worker'")
tf.compat.v1.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.compat.v1.app.flags.FLAGS

 #Set up server
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = True


#config.gpu_options.per_process_gpu_memory_fraction = 0.5
server = tf.compat.v1.train.Server(cluster,
    job_name=FLAGS.job_name,
    task_index=FLAGS.task_index,
    config=config)

final_step = 100000000

LOG_DIR = 'kdd_ddl3-%d' % len(workers)
print('parameters specification finished!')
#--------------------------------------------

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
    num_features = ["duration", "protocol_type","service","flag", "src_bytes", "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_srv_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate"
    ]
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
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]
    dataset = pd.read_csv(filename, header = None, names = col_names)
    return dataset

def normalize(dataset):
    mu = np.mean(dataset, axis = 0)
    sigma = np.std(dataset, axis = 0)
    return (dataset - mu)/sigma

def read_data_set(dataset1, dataset2, one_hot = False, dtype = dtypes.float32, reshape = True):

    segments1, labels1 = segment_signal(dataset1)
    #labels1 = np.asarray(pd.get_dummies(labels1), dtype = np.int8)

    segments2, labels2 = segment_signal(dataset2)
    #labels2 = np.asarray(pd.get_dummies(labels2), dtype = np.int8)
    labels = np.asarray(pd.get_dummies(labels1._append([labels2])), dtype = np.int8)
    labels1 = labels[:len(labels1)]
    labels2 = labels[len(labels1):]
    train_x = segments1.reshape(len(segments1), 1, 1 ,41)
    train_y = labels1

    test_x = segments2.reshape(len(segments2), 1, 1 ,41)
    test_y = labels2

    train = Dataset(train_x, train_y, dtype = dtype , reshape = reshape)
    test = Dataset(test_x, test_y, dtype = dtype, reshape = reshape)
    return train,test

def initlabel(dataset):
    labels = dataset['label'].copy()
    labels[labels == 'back.'] = 'dos'
    labels[labels == 'buffer_overflow.'] = 'u2r'
    labels[labels == 'ftp_write.'] =  'r2l'
    labels[labels == 'guess_passwd.'] = 'r2l'
    labels[labels == 'imap.'] = 'r2l'
    labels[labels == 'ipsweep.'] = 'probe'
    labels[labels == 'land.'] = 'dos'
    labels[labels == 'loadmodule.'] = 'u2r'
    labels[labels == 'multihop.'] = 'r2l'
    labels[labels == 'neptune.'] = 'dos'
    labels[labels == 'nmap.'] = 'probe'
    labels[labels == 'perl.'] = 'u2r'
    labels[labels == 'phf.'] =  'r2l'
    labels[labels == 'pod.'] =  'dos'
    labels[labels == 'portsweep.'] = 'probe'
    labels[labels == 'rootkit.'] = 'u2r'
    labels[labels == 'satan.'] = 'probe'
    labels[labels == 'smurf.'] = 'dos'
    labels[labels == 'spy.'] = 'r2l'
    labels[labels == 'teardrop.'] = 'dos'
    labels[labels == 'warezclient.'] = 'r2l'
    labels[labels == 'warezmaster.'] = 'r2l'
    labels[labels == 'apache2.'] = 'dos'
    labels[labels == 'mailbomb.'] = 'dos'
    labels[labels == 'processtable.'] = 'dos'
    labels[labels == 'udpstorm.'] = 'dos'
    labels[labels == 'mscan.'] = 'probe'
    labels[labels == 'saint.'] = 'probe'
    labels[labels == 'ps.'] = 'u2r'
    labels[labels == 'sqlattack.'] = 'u2r'
    labels[labels == 'xterm.'] = 'u2r'
    labels[labels == 'named.'] = 'r2l'
    labels[labels == 'sendmail.'] = 'r2l'
    labels[labels == 'snmpgetattack.'] = 'r2l'
    labels[labels == 'snmpguess.'] = 'r2l'
    labels[labels == 'worm.'] = 'r2l'
    labels[labels == 'xlock.'] = 'r2l'
    labels[labels == 'xsnoop.'] = 'r2l'
    labels[labels == 'httptunnel.'] = 'r2l'
    return labels

def nomial(dataset1, dataset2):

    dataset = dataset1._append([dataset2])
    protocol1 = dataset1['protocol_type'].copy()
    protocol2 = dataset2['protocol_type'].copy()
    protocol_type = dataset['protocol_type'].unique()
    for i in range(len(protocol_type)):
        protocol1[protocol1 == protocol_type[i]] = i
        protocol2[protocol2 == protocol_type[i]] = i
    dataset1['protocol_type'] = protocol1
    dataset2['protocol_type'] = protocol2

    service1 = dataset1['service'].copy()
    service2 = dataset2['service'].copy()
    service_type = dataset['service'].unique()
    for i in range(len(service_type)):
        service1[service1 == service_type[i]] = i
        service2[service2 == service_type[i]] = i
    dataset1['service'] = service1
    dataset2['service'] = service2

    flag1 = dataset1['flag'].copy()
    flag2 = dataset2['flag'].copy()
    flag_type = dataset['flag'].unique()
    for i in range(len(flag_type)):
        flag1[flag1 == flag_type[i]] = i
        flag2[flag2 == flag_type[i]] = i

    dataset1['flag'] = flag1
    dataset2['flag'] = flag2

if __name__ == "__main__":
    dir_path =  os.path.join(os.getcwd(), "..", "..")

    filename1 = dir_path + "/datasets/kdd/kddcup.data_10_percent_corrected"
    filename2 = dir_path + "/datasets/kdd/corrected"

    dataset11 = read_data(filename1)
    dataset22 = read_data(filename2)

    #dataset22, dataset44 = train_test_split(dataset33, train_size=0.2, random_state=0)

    dataset1, dataset3 = train_test_split(dataset11, train_size=0.5, random_state=0)
    dataset2, dataset4 = train_test_split(dataset22, train_size=0.5, random_state=0)

    #-------------pre-process dataset1,2-----------------------
    nomial(dataset1, dataset2)

    dataset1['label'] = initlabel(dataset1)
    dataset2['label'] = initlabel(dataset2)

    num_features = ["duration","protocol_type","service","flag", "src_bytes", "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
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
    dataset2[num_features] = dataset2[num_features].astype(float)
    #dataset2[num_features] = dataset2[num_features].apply(lambda x:MinMaxScaler().fit_transform(x))
    dataset2[num_features] = MinMaxScaler().fit_transform(dataset2[num_features].values)
    #print(dataset.describe())
    print(dataset1.describe())

    print(dataset1['label'].value_counts())

    labels1 = dataset1['label'].copy()
    print(labels1.unique())

    labels1[labels1 == 'normal.'] = 0
    labels1[labels1 == 'dos'] = 1
    labels1[labels1 == 'u2r'] = 2
    labels1[labels1 == 'r2l'] = 3
    labels1[labels1 == 'probe'] = 4
    dataset1['label'] = labels1

    labels2 = dataset2['label'].copy()
    print(labels2.unique())

    labels2[labels2 == 'normal.'] = 0
    labels2[labels2 == 'dos'] = 1
    labels2[labels2 == 'u2r'] = 2
    labels2[labels2 == 'r2l'] = 3
    labels2[labels2 == 'probe'] = 4
    dataset2['label'] = labels2

    train_set0, test_set0= read_data_set(dataset1, dataset2)
    print(train_set0.labels)

    #-------------pre-process dataset3,4-----------------------
    nomial(dataset3, dataset4)

    dataset3['label'] = initlabel(dataset3)
    dataset4['label'] = initlabel(dataset4)

    num_features = ["duration","protocol_type","service","flag", "src_bytes", "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_srv_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate"
    ]
    dataset3[num_features] = dataset3[num_features].astype(float)
    dataset3[num_features] = MinMaxScaler().fit_transform(dataset3[num_features].values)
    dataset4[num_features] = dataset4[num_features].astype(float)
    dataset4[num_features] = MinMaxScaler().fit_transform(dataset4[num_features].values)

    labels3 = dataset3['label'].copy()
    labels3[labels3 == 'normal.'] = 0
    labels3[labels3 == 'dos'] = 1
    labels3[labels3 == 'u2r'] = 2
    labels3[labels3 == 'r2l'] = 3
    labels3[labels3 == 'probe'] = 4
    dataset3['label'] = labels3

    labels4 = dataset4['label'].copy()
    labels4[labels4 == 'normal.'] = 0
    labels4[labels4 == 'dos'] = 1
    labels4[labels4 == 'u2r'] = 2
    labels4[labels4 == 'r2l'] = 3
    labels4[labels4 == 'probe'] = 4
    dataset4['label'] = labels4

    train_set1, test_set1 = read_data_set(dataset3, dataset4)
    #-------------------------------------------------------------
    num_agg = len(workers)
    #DBN structure
    if FLAGS.job_name == "ps":
        server.join()

    elif FLAGS.job_name == "worker":
        print('Training begin!')
        # Between-graph replication
        is_chief = (FLAGS.task_index == 0) #checks if this is the chief node
        with tf.compat.v1.device(tf.compat.v1.train.replica_device_setter(ps_tasks=1,
            worker_device="/job:worker/task:%d" % FLAGS.task_index)):
            # count the number of updates
            global_step = tf.compat.v1.Variable(0,dtype=tf.int32,trainable=False,name='global_step')

            #--------------------DBN-----------------------------------

            n_inp = [1, 1, 41]
            hidden_layer_sizes = [500, 500, 500]
            n_out = 5

            sigmoid_layers = []
            layers = []
            params = []
            n_layers = len(hidden_layer_sizes)

            learning_rate_pre = 0.001
            learning_rate_tune = 0.1
            k = 1

            assert n_layers > 0

            #define the grape
            height, weight, channel = n_inp
            x = tf.compat.v1.placeholder(tf.float32, [None, height, weight, channel])
            y = tf.compat.v1.placeholder(tf.float32, [None, n_out])

            for i in range(n_layers):
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
                    layer_input = tf.reshape(x, [-1, height*weight*channel])
                else:
                    layer_input = sigmoid_layers[-1].output

                sigmoid_layer = HiddenLayer(input = layer_input, n_inp = input_size,
                    n_out = hidden_layer_sizes[i], activation = tf.nn.sigmoid)

                #add the layer to our list of layers
                sigmoid_layers.append(sigmoid_layer)

                # Its arguably a philosophical question... but we are going to only
                # declare that the parameters of the sigmoid_layers are parameters of the DBN.
                # The visible biases in the RBM are parameters of those RBMs, but not of the DBN

                params.extend(sigmoid_layer.params)
                if i == 0:
                    rbm_layer = GRBM(inp = layer_input, n_visible = input_size, n_hidden = hidden_layer_sizes[i], W = sigmoid_layer.W, hbias = sigmoid_layer.b)
                else:
                    rbm_layer = RBM(inp = layer_input, n_visible = input_size, n_hidden = hidden_layer_sizes[i], W = sigmoid_layer.W, hbias = sigmoid_layer.b)
                layers.append(rbm_layer)

            cost0, train_ops0 = layers[0].get_train_ops(lr = learning_rate_pre, persistent = None, k = k)
            cost1, train_ops1 = layers[1].get_train_ops(lr = learning_rate_pre, persistent = None, k = k)
            cost2, train_ops2 = layers[2].get_train_ops(lr = learning_rate_pre, persistent = None, k = k)

            logLayer = LogisticRegression(input= sigmoid_layers[-1].output, n_inp = hidden_layer_sizes[-1], n_out = n_out)
            params.extend(logLayer.params)

            #compute the cost for second phase of training, defined as the cost of the
            # logistic regression output layer
            finetune_cost = logLayer.cost(y)

            #compute the gradients with respect to the model parameters symbolic variable that
            # points to the number of errors made on the minibatch given by self.x and self.y
            pred = logLayer.pred

            #----optimizer ----------------------
            optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = learning_rate_tune)
            optimizer = tf.compat.v1.train.SyncReplicasOptimizer(optimizer,
                                                    replicas_to_aggregate=num_agg,
                                                    total_num_replicas=num_agg)

            train_ops = optimizer.minimize(finetune_cost, var_list= params, global_step=global_step)
            #-------------------------------------------------------
            c1 = tf.compat.v1.argmax(pred, axis =1)
            c2 = tf.compat.v1.argmax(y, axis =1)

            print('Summaries begin!')

            tf.compat.v1.summary.scalar('loss',finetune_cost)
            tf.compat.v1.summary.histogram('pred_y',pred)
            tf.compat.v1.summary.histogram('gradients',train_ops)
            merged = tf.compat.v1.summary.merge_all()

            init_op = tf.compat.v1.global_variables_initializer()

        sync_replicas_hook = optimizer.make_session_run_hook(is_chief)
        stop_hook = tf.compat.v1.estimator.StopAtStepHook(last_step = final_step)
        summary_hook = tf.compat.v1.estimator.SummarySaverHook(save_secs=600,output_dir= LOG_DIR,summary_op=merged)
        hooks = [sync_replicas_hook, stop_hook,summary_hook]
        scaff = tf.compat.v1.train.Scaffold(init_op = init_op)

        begin_time = time.time()
        print("Waiting for other servers")
        with tf.compat.v1.train.MonitoredTrainingSession(master = server.target,
                                              is_chief = (FLAGS.task_index ==0),
                                              checkpoint_dir = LOG_DIR,
                                              scaffold = scaff,
                                              hooks = hooks) as sess:
            print('Starting training on worker %d'%FLAGS.task_index)
            while not sess.should_stop():
                train_writer = tf.compat.v1.summary.FileWriter(os.path.join(LOG_DIR,'train'), graph = tf.compat.v1.get_default_graph())
                test_writer = tf.compat.v1.summary.FileWriter(os.path.join(LOG_DIR,'test'),graph = tf.compat.v1.get_default_graph())
                print('Starting training on worker %d -------------------------------------------------'%FLAGS.task_index)
                #----pretraining -------------------------------------------------------------------------------
                start_time = timeit.default_timer()
                pretraining_epochs = 3
                batch_size_pre = 100
                display_step_pre = 1
                batch_num_pre = int(globals()['train_set'+str(FLAGS.task_index)].num_examples / batch_size_pre)

                for epoch in range(pretraining_epochs):
                    avg_cost = 0.0
                    for j in range(batch_num_pre):
                        batch_xs, batch_ys = globals()['train_set'+str(FLAGS.task_index)].next_batch(batch_size_pre)
                        c,_ = sess.run([cost0, train_ops0], feed_dict = {x: batch_xs,y : batch_ys})
                        avg_cost += c / batch_num_pre

                    if epoch % display_step_pre == 0:
                        print("Worker {0} Pretraining layer 1 Epoch {1}".format( int(FLAGS.task_index), epoch +1) + " cost {:.9f}".format(avg_cost))
                        end_time = timeit.default_timer()
                        print("time {0} minutes".format((end_time - start_time)/ 60.))

                for epoch in range(pretraining_epochs):
                    avg_cost = 0.0
                    for j in range(batch_num_pre):
                        batch_xs, batch_ys = globals()['train_set'+str(FLAGS.task_index)].next_batch(batch_size_pre)
                        c,_ = sess.run([cost1, train_ops1], feed_dict = {x: batch_xs,y : batch_ys})
                        avg_cost += c / batch_num_pre

                    if epoch % display_step_pre == 0:
                        print("Worker {0} Pretraining layer 2 Epoch {1}".format( int(FLAGS.task_index), epoch +1) + " cost {:.9f}".format(avg_cost))
                        end_time = timeit.default_timer()
                        print("time {0} minutes".format((end_time - start_time)/ 60.))

                for epoch in range(pretraining_epochs):
                    avg_cost = 0.0
                    for j in range(batch_num_pre):
                        batch_xs, batch_ys = globals()['train_set'+str(FLAGS.task_index)].next_batch(batch_size_pre)
                        c,_ = sess.run([cost2, train_ops2], feed_dict = {x: batch_xs,y : batch_ys})
                        avg_cost += c / batch_num_pre

                    if epoch % display_step_pre == 0:
                        print("Worker {0} Pretraining layer 3 Epoch {1}".format( int(FLAGS.task_index), epoch +1) + " cost {:.9f}".format(avg_cost))
                        end_time = timeit.default_timer()
                        print("time {0} minutes".format((end_time - start_time)/ 60.))

                end_time = timeit.default_timer()
                print("time {0} minutes".format((end_time - start_time)/ 60.))

                #--------------------fune-tuning----------------------------------------------------------------
                start_time = timeit.default_timer()

                batch_size_num = 10
                training_epochs = 1000
                display_step_tune = 1

                ACC_max = 0
                pre_max = 0
                rec_max = 0

                batch_num_tune = int(globals()['train_set'+str(FLAGS.task_index)].num_examples/batch_size_num)

                for epoch in range(training_epochs):
                    avg_cost = 0.0
                    for i in range(batch_num_tune):

                        batch_xs, batch_ys = globals()['train_set'+str(FLAGS.task_index)].next_batch(batch_size_num)
                        summary, _, c, step= sess.run([merged, train_ops, finetune_cost, global_step], feed_dict = {x :batch_xs, y : batch_ys} )
                        train_writer.add_summary(summary,i)
                        avg_cost += c / batch_num_tune

                    summary,output_train = sess.run([merged,pred],feed_dict={x: globals()['train_set'+str(FLAGS.task_index)].segments, y: globals()['train_set'+str(FLAGS.task_index)].labels})
                    summary,output_test = sess.run([merged,pred], feed_dict={x: globals()['test_set'+str(FLAGS.task_index)].segments, y: globals()['test_set'+str(FLAGS.task_index)].labels})
                    test_writer.add_summary(summary, epoch)
                    b =[]
                    d = []

                    if epoch % display_step_tune == 0:
                        print("Epoch:", '%04d' % (epoch +1), "cost:", "{:.9f}".format(avg_cost))

                        c_test,pr = sess.run([c1, pred], feed_dict = {x: globals()['test_set'+str(FLAGS.task_index)].segments, y: globals()['test_set'+str(FLAGS.task_index)].labels})
                        b = np.append(b,c_test)

                        d_test, y_test = sess.run([c2, y], feed_dict ={x: globals()['test_set'+str(FLAGS.task_index)].segments, y: globals()['test_set'+str(FLAGS.task_index)].labels})
                        d = np.append(d, d_test)

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
                        print("Epoch:", '%04d' % (epoch +1), "cost:", "{:.9f}".format(avg_cost))
                        print("WORKER: {0}, ACCURACY: {1}, PRECISION: {2}, RECALL: {3}:".format(int(FLAGS.task_index), ACC_max, pre_max, rec_max))
                        end_time = timeit.default_timer()
                        print("Time {0} minutes".format((end_time- start_time)/ 60.))

                end_time = timeit.default_timer()
                print("Time {0} minutes".format((end_time- start_time)/ 60.))
