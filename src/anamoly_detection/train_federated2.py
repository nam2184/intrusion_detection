#!/usr/bin/python

import sys
import os
import pandas as pd
from glob import iglob
import numpy as np
from keras.models import load_model
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import SGD

from mlxtend.preprocessing import one_hot
import argparse
import random
import sys
import time
import timeit
from tensorflow.contrib import slim
from sklearn.metrics import mean_squared_error as MSE

from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, accuracy_score, precision_score, confusion_matrix
import lime
import lime.lime_tabular

#----------distributed------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

#define cluster
parameter_servers = ["localhost:2222"]
workers = [ "localhost:2223", "localhost:2224"]
cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

# Input Flags
tf.app.flags.DEFINE_string("job_name","", "'ps' / 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

 #Set up server
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = True

#config.gpu_options.per_process_gpu_memory_fraction = 0.5
server = tf.distribute.Server(cluster,
    job_name=FLAGS.job_name,
    task_index=FLAGS.task_index,
    config=config)

final_step = 10000000

LOG_DIR = 'train_federated3'
print('parameters specification finished!')
#--------------------------------------------

def load_mal_data():
    #df_mirai = pd.concat((pd.read_csv(f) for f in iglob('/media/donga/LUU TRU 1/Study/Ph.D/AI/Detection of IoT Botnet Attacks Data Set/code/botnet-traffic-analysis-master/data/SimpleHome_XCS7_1003_WHT_Security_Camera/mirai_attacks/*.csv', recursive=True)), ignore_index=True)
    df_mirai = pd.concat((pd.read_csv(f) for f in iglob('/media/donga/LUU TRU 1/Study/Ph.D/AI/Detection of IoT Botnet Attacks Data Set/code/botnet-traffic-analysis-master/data/Samsung_SNH_1011_N_Webcam/gafgyt_attacks/*.csv', recursive=True)), ignore_index=True)
    #df_gafgyt = pd.DataFrame()
    #for f in iglob('/media/donga/LUU TRU 1/Study/Ph.D/AI/Detection of IoT Botnet Attacks Data Set/code/botnet-traffic-analysis-master/data/SimpleHome_XCS7_1003_WHT_Security_Camera/gafgyt_attacks/*.csv', recursive=True):
    #    df_gafgyt = df_gafgyt.append(pd.read_csv(f), ignore_index=True)
    #return df_mirai.append(df_gafgyt)
    return df_mirai

def net(input_layer):

    input_dim = int(input_layer.shape[1])
    net = slim.layers.fully_connected(input_layer, int(0.75 * input_dim), activation_fn=tf.nn.tanh, scope='fully_connected1')
    net = slim.layers.fully_connected(net, int(0.5 * input_dim), activation_fn=tf.nn.tanh, scope='fully_connected2')
    net = slim.layers.fully_connected(net, int(0.33 * input_dim), activation_fn=tf.nn.tanh, scope='fully_connected3')
    net = slim.layers.fully_connected(net, int(0.25 * input_dim), activation_fn=tf.nn.tanh, scope='fully_connected4')
    net = slim.layers.fully_connected(net, int(0.33 * input_dim), activation_fn=tf.nn.tanh, scope='fully_connected5')
    net = slim.layers.fully_connected(net, int(0.5 * input_dim), activation_fn=tf.nn.tanh, scope='fully_connected6')
    net = slim.layers.fully_connected(net, int(0.75 * input_dim), activation_fn=tf.nn.tanh, scope='fully_connected7')
    #net = slim.dropout(net, 0.8, scope = 'dropout1') 
    net = slim.layers.fully_connected(net, input_dim, scope='pred')
    return net
#--------------------------------------------

if __name__ == '__main__':

    top_n_features=115

    #--------------train data----------------------------------------------------

    print("Loading combined training data...")
    df11 = pd.concat((pd.read_csv(f) for f in iglob('/media/donga/LUU TRU 1/Study/Ph.D/AI/Detection of IoT Botnet Attacks Data Set/code/botnet-traffic-analysis-master/data/Samsung_SNH_1011_N_Webcam/benign_traffic.csv', recursive=True)), ignore_index=True)
    df22 = df11.reindex(np.random.permutation(df11.index))
    df1, df2 = train_test_split(df22, train_size=0.5, random_state=42)

    fisher = pd.read_csv('/media/donga/LUU TRU 1/Study/Ph.D/AI/Detection of IoT Botnet Attacks Data Set/code/botnet-traffic-analysis-master/fisher.csv')
    features = fisher.iloc[0:int(top_n_features)]['Feature'].values

    #---------------------train dataset1----------------------------------------
    df1 = df1[list(features)]
    x_train0, x_opt0, x_test0 = np.split(df1.sample(frac=1, random_state=17), [int(1/3*len(df1)), int(2/3*len(df1))])
    scaler0 = StandardScaler()
    scaler0.fit(x_train0.append(x_opt0))
    x_train0 = scaler0.transform(x_train0)
    x_opt0 = scaler0.transform(x_opt0)
    x_test0 = scaler0.transform(x_test0)

    #---------------------train dataset2----------------------------------------
    df2 = df2[list(features)]
    x_train1, x_opt1, x_test1 = np.split(df2.sample(frac=1, random_state=17), [int(1/3*len(df2)), int(2/3*len(df2))])
    scaler1 = StandardScaler()
    scaler1.fit(x_train1.append(x_opt1))
    x_train1 = scaler1.transform(x_train1)
    x_opt1 = scaler1.transform(x_opt1)
    x_test1 = scaler1.transform(x_test1)

    #--------------test data----------------------------------------------------

    df_malicious1t = load_mal_data()
    print("Testing")
    df1t = pd.concat((pd.read_csv(f) for f in iglob('/media/donga/LUU TRU 1/Study/Ph.D/AI/Detection of IoT Botnet Attacks Data Set/code/botnet-traffic-analysis-master/data/Samsung_SNH_1011_N_Webcam/benign_traffic.csv', recursive=True)), ignore_index=True)
    fisher1t = pd.read_csv('/media/donga/LUU TRU 1/Study/Ph.D/AI/Detection of IoT Botnet Attacks Data Set/code/botnet-traffic-analysis-master/fisher.csv')
    features1t = fisher1t.iloc[0:int(top_n_features)]['Feature'].values
    df1t = df1t[list(features1t)]

    x_train1t, x_opt1t, x_test1t = np.split(df1t.sample(frac=1, random_state=17), [int(1/3*len(df1t)), int(2/3*len(df1t))])

    scaler1t = StandardScaler()
    scaler1t.fit(x_train1t.append(x_opt1t))
    df_benign1t = pd.DataFrame(x_test1t, columns=df1t.columns)
    df_benign1t['malicious'] = 0
    df_malicious1t = df_malicious1t.sample(n=df_benign1t.shape[0], random_state=17)[list(features1t)]
    df_malicious1t['malicious'] = 1
    df1t = df_benign1t.append(df_malicious1t)

    df1t1, df1t2 = train_test_split(df1t, train_size=0.5, random_state=42)

    #---------------------test dataset1----------------------------------------
    X_test1t0 = df1t1.drop(columns=['malicious']).values
    X_test_scaled1t0 = scaler1t.transform(X_test1t0)
    Y_test1t0 = df1t1['malicious']

    #---------------------test dataset2----------------------------------------
    X_test1t1 = df1t2.drop(columns=['malicious']).values
    X_test_scaled1t1 = scaler1t.transform(X_test1t1)
    Y_test1t1 = df1t2['malicious']

    #----------------finish dataset-------------------------------------------
    batch_size = 64
    if FLAGS.job_name == "worker":
        n_samples =  globals()['x_train'+str(FLAGS.task_index)].shape[0]
        num_iterations = n_samples//batch_size
    
    learning_rate=0.01
    final_step = 1000000
    LOG_DIR = 'anomaly_log_ddl3'
    epochs = 500

    num_agg = len(workers)
    
    #DBN structure
    if FLAGS.job_name == "ps":
        server.join()

    elif FLAGS.job_name == "worker":
        print('Training begin!')
        # Between-graph replication
        is_chief = (FLAGS.task_index == 0) #checks if this is the chief node
        with tf.device(tf.train.replica_device_setter(ps_tasks=1,
            worker_device="/job:worker/task:%d" % FLAGS.task_index)):

            global_step = tf.Variable(0,dtype=tf.int32,trainable=False,name='global_step')

            with tf.name_scope('input'):
                # user with 3706 ratings goes in
                x = tf.placeholder(tf.float32, [None, 115])
                # output_true shall have the original ratings for error calculations
                y = tf.placeholder(tf.float32, [None, 115])

                Y = net(x)
                #Y = model.predict(x)

            with tf.name_scope('train'):
                # define our cost function
                cost =  tf.reduce_mean(tf.square(Y - y))
                #cost = tf.nn.log_poisson_loss(targets=y, log_input=Y)

                optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)

                optimizer = tf.train.SyncReplicasOptimizer(optimizer,
                                                        replicas_to_aggregate=num_agg,
                                                        total_num_replicas=num_agg)
                opt = optimizer.minimize(cost, global_step=global_step)
                
            print('Summaries begin!')
            tf.compat.v1.summary.scalar('loss',cost) 
            tf.compat.v1.summary.histogram('pred_y',Y)
            tf.compat.v1.summary.histogram('gradients',opt)
            
            merged = tf.compat.v1.summary.merge_all()
            init_op = tf.compat.v1.global_variables_initializer()

        sync_replicas_hook = optimizer.make_session_run_hook(is_chief)
        stop_hook = tf.estimator.StopAtStepHook(last_step = final_step)
        summary_hook = tf.estimator.SummarySaverHook(save_secs=600,output_dir= LOG_DIR,summary_op=merged)
        hooks = [sync_replicas_hook, stop_hook,summary_hook]
        scaff = tf.compat.v1.train.Scaffold(init_op = init_op)

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

                start_time = timeit.default_timer()  
                for epoch in range(epochs):
                    epoch_loss = 0
                    for i in range(int(num_iterations)):
                        batch_x = globals()['x_train'+str(FLAGS.task_index)][i*batch_size:(i+1)*batch_size]
                        summary, _,loss,step = sess.run([merged, opt, cost, global_step],feed_dict={x: batch_x, y: batch_x})
                        train_writer.add_summary(summary,i)
                        epoch_loss += loss
            
                    summary,output_train = sess.run([merged,Y],feed_dict={x: globals()['x_train'+str(FLAGS.task_index)], y: globals()['x_train'+str(FLAGS.task_index)]})
                    summary,output_opt = sess.run([merged,Y],feed_dict={x: globals()['x_opt'+str(FLAGS.task_index)], y: globals()['x_opt'+str(FLAGS.task_index)]}) 

                    test_writer.add_summary(summary, epoch)
                    if epoch % 50 == 0 :
                        print('{0:d}\t{1:d}\t{2:d}\t{3:.4f}\t{4:.4f}'.format(int(step), epoch, int(FLAGS.task_index), MSE(output_train, globals()['x_train'+str(FLAGS.task_index)])**0.5, MSE(output_opt, globals()['x_opt'+str(FLAGS.task_index)])**0.5))       
                
                print("Calculating threshold")                        
                x_opt_predictions = sess.run(Y, feed_dict={x: globals()['x_opt'+str(FLAGS.task_index)], y: globals()['x_opt'+str(FLAGS.task_index)]})
                print("Calculating MSE on optimization set...")
                mse = np.mean(np.power(globals()['x_opt'+str(FLAGS.task_index)] - x_opt_predictions, 2), axis=1)
                print("mean is %.5f" % mse.mean())
                print("min is %.5f" % mse.min())
                print("max is %.5f" % mse.max())
                print("std is %.5f" % mse.std())
                
                tr = mse.mean() + mse.std()
                with open(f'threshold_{top_n_features}_%d'%FLAGS.task_index, 'w') as t:
                    t.write(str(tr))
                print(f"Calculated threshold is {tr} for worker %d"%FLAGS.task_index)

                x_test_predictions = sess.run(Y, feed_dict={x: globals()['x_test'+str(FLAGS.task_index)], y: globals()['x_test'+str(FLAGS.task_index)]})
                print("Calculating MSE on test set...")
                mse_test = np.mean(np.power(globals()['x_test'+str(FLAGS.task_index)] - x_test_predictions, 2), axis=1)
                over_tr = mse_test > tr
                false_positives = sum(over_tr)
                test_size = mse_test.shape[0]
                print(f"{false_positives} false positives on dataset without attacks with size {test_size} of worker %d"%FLAGS.task_index)

                #------------------Testing--------------------------------------
                with open(f'threshold_{top_n_features}_%d'%FLAGS.task_index) as t1t:
                    tr1t = np.float64(t1t.read())
                print(f"Calculated threshold test is {tr1t}")
                x_pred1t = sess.run(Y, feed_dict={x: globals()['X_test_scaled1t'+str(FLAGS.task_index)], y: globals()['X_test_scaled1t'+str(FLAGS.task_index)]})
                mse1t = np.mean(np.power(globals()['X_test_scaled1t'+str(FLAGS.task_index)] - x_pred1t, 2), axis=1)
                y_pred1t = mse1t > tr1t
                Y_pred1t = y_pred1t.astype(int)

                print('Accuracy worker: %d'%FLAGS.task_index)
                print(accuracy_score(globals()['Y_test1t'+str(FLAGS.task_index)], Y_pred1t))

                print('Recall worker: %d'%FLAGS.task_index)
                print(recall_score(globals()['Y_test1t'+str(FLAGS.task_index)], Y_pred1t))

                print('Precision worker: %d'%FLAGS.task_index)
                print(precision_score(globals()['Y_test1t'+str(FLAGS.task_index)], Y_pred1t))
                print(confusion_matrix(globals()['Y_test1t'+str(FLAGS.task_index)], Y_pred1t))   

                end_time = timeit.default_timer()
                print("Time {0} minutes".format((end_time- start_time)/ 60.))      
