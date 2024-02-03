#!/usr/bin/python

import sys
import os
import pandas as pd
from glob import iglob
import numpy as np
from keras.models import load_model
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import SGD
import tf_slim as slim
import time
import timeit
from sklearn.metrics import mean_squared_error as MSE

import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, accuracy_score, precision_score, confusion_matrix
#import lime
#import lime.lime_tabular

def load_mal_data():
    #df_mirai = pd.concat((pd.read_csv(f) for f in iglob('/media/donga/LUU TRU 1/Study/Ph.D/AI/Detection of IoT Botnet Attacks Data Set/code/botnet-traffic-analysis-master/data/SimpleHome_XCS7_1003_WHT_Security_Camera/mirai_attacks/*.csv', recursive=True)), ignore_index=True)
    df_mirai = pd.concat((pd.read_csv(f) for f in iglob('/media/donga/LUU TRU 1/Study/Ph.D/AI/Detection of IoT Botnet Attacks Data Set/code/botnet-traffic-analysis-master/data/Samsung_SNH_1011_N_Webcam/gafgyt_attacks/*.csv', recursive=True)), ignore_index=True)
    #df_gafgyt = pd.DataFrame()
    #for f in iglob('/media/donga/LUU TRU 1/Study/Ph.D/AI/Detection of IoT Botnet Attacks Data Set/code/botnet-traffic-analysis-master/data/SimpleHome_XCS7_1003_WHT_Security_Camera/gafgyt_attacks/*.csv', recursive=True):
    #    if 'tcp.csv' in f or 'udp.csv' in f:
    #        continue
    #    df_gafgyt = df_gafgyt.append(pd.read_csv(f), ignore_index=True)
    #return df_mirai.append(df_gafgyt)
    return df_mirai

def net(input_layer):
    # autoencoder = Sequential()
    # autoencoder.add(Dense(int(0.75 * input_dim), activation="tanh", input_shape=(input_dim,)))
    # autoencoder.add(Dense(int(0.5 * input_dim), activation="tanh"))
    # autoencoder.add(Dense(int(0.33 * input_dim), activation="tanh"))
    # autoencoder.add(Dense(int(0.25 * input_dim), activation="tanh"))
    # autoencoder.add(Dense(int(0.33 * input_dim), activation="tanh"))
    # autoencoder.add(Dense(int(0.5 * input_dim), activation="tanh"))
    # autoencoder.add(Dense(int(0.75 * input_dim), activation="tanh"))
    # autoencoder.add(Dense(input_dim))
    # return autoencoder

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

if __name__ == '__main__':

    top_n_features=115

    #--------------train data----------------------------------------------------

    df = pd.concat((pd.read_csv(f) for f in iglob('/media/donga/LUU TRU 1/Study/Ph.D/AI/Detection of IoT Botnet Attacks Data Set/code/botnet-traffic-analysis-master/data/Samsung_SNH_1011_N_Webcam/benign_traffic.csv', recursive=True)), ignore_index=True)
    fisher = pd.read_csv('/media/donga/LUU TRU 1/Study/Ph.D/AI/Detection of IoT Botnet Attacks Data Set/code/botnet-traffic-analysis-master/fisher.csv')
    features = fisher.iloc[0:int(top_n_features)]['Feature'].values
    df = df[list(features)]

    x_train, x_opt, x_test = np.split(df.sample(frac=1, random_state=17), [int(1/3*len(df)), int(2/3*len(df))])
    scaler = StandardScaler()
    scaler.fit(x_train.append(x_opt))
    x_train = scaler.transform(x_train)
    x_opt = scaler.transform(x_opt)
    x_test = scaler.transform(x_test)

    #--------------test data----------------------------------------------------

    df_malicious1 = load_mal_data()
    print("Testing")
    df1 = pd.concat((pd.read_csv(f) for f in iglob('/media/donga/LUU TRU 1/Study/Ph.D/AI/Detection of IoT Botnet Attacks Data Set/code/botnet-traffic-analysis-master/data/Samsung_SNH_1011_N_Webcam/benign_traffic.csv', recursive=True)), ignore_index=True)
    fisher1 = pd.read_csv('/media/donga/LUU TRU 1/Study/Ph.D/AI/Detection of IoT Botnet Attacks Data Set/code/botnet-traffic-analysis-master/fisher.csv')
    features1 = fisher1.iloc[0:int(top_n_features)]['Feature'].values
    df1 = df1[list(features1)]

    x_train1, x_opt1, x_test1 = np.split(df1.sample(frac=1, random_state=17), [int(1/3*len(df1)), int(2/3*len(df1))])
    scaler1 = StandardScaler()
    scaler1.fit(x_train1.append(x_opt1))

    #print(f"Loading model")
    #saved_model = load_model(f'models/model_{top_n_features}.h5')

    #model1 = AnomalyModel(model, tr, scaler)

    df_benign1 = pd.DataFrame(x_test1, columns=df1.columns)
    df_benign1['malicious'] = 0
    df_malicious1 = df_malicious1.sample(n=df_benign1.shape[0], random_state=17)[list(features1)]
    df_malicious1['malicious'] = 1
    df1 = df_benign1.append(df_malicious1)
    X_test1 = df1.drop(columns=['malicious']).values
    X_test_scaled1 = scaler1.transform(X_test1)
    Y_test1 = df1['malicious']

    #---------------------------------------------------------------------------

    batch_size = 64
    n_samples =  x_train.shape[0]
    num_iterations = n_samples//batch_size
    learning_rate=0.01
    final_step = 1000000
    LOG_DIR = 'anomaly_log'
    epochs = 500

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
        
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step=global_step)

    print('Summaries begin!')
    tf.compat.v1.summary.scalar('loss',cost) 
    tf.compat.v1.summary.histogram('pred_y',Y)
    
    merged = tf.compat.v1.summary.merge_all()
    init_op = tf.compat.v1.global_variables_initializer()


    stop_hook = tf.estimator.StopAtStepHook(last_step = final_step)
    summary_hook = tf.estimator.SummarySaverHook(save_secs=600,output_dir= LOG_DIR,summary_op=merged)
    hooks = [stop_hook,summary_hook]
    scaff = tf.compat.v1.train.Scaffold(init_op = init_op)
        
    begin_time = time.time()
    with tf.compat.v1.train.MonitoredTrainingSession(checkpoint_dir = LOG_DIR,
                                                    scaffold = scaff,
                                                    hooks = hooks) as sess:  

        print('Starting training')

        while not sess.should_stop():
            train_writer = tf.compat.v1.summary.FileWriter(os.path.join(LOG_DIR,'train'), graph = tf.compat.v1.get_default_graph())
            test_writer = tf.compat.v1.summary.FileWriter(os.path.join(LOG_DIR,'test'),graph = tf.compat.v1.get_default_graph())

            start_time = timeit.default_timer()
            for epoch in range(epochs):
                epoch_loss = 0
                for i in range(int(num_iterations)):
                    batch_x = x_train[i*batch_size:(i+1)*batch_size]
                    summary, _,loss,step = sess.run([merged, optimizer,cost,global_step],feed_dict={x: batch_x, y: batch_x})
                    train_writer.add_summary(summary,i)
                    epoch_loss += loss
        
                summary,output_train = sess.run([merged,Y],feed_dict={x: x_train, y: x_train})
                summary,output_test = sess.run([merged,Y], feed_dict={x: x_opt, y: x_opt}) 
                test_writer.add_summary(summary, epoch)
                if epoch % 50 == 0 :
                    print('{0:d}\t{1:d}\t{2:.4f}\t{3:.4f}'.format(int(step), epoch, MSE(output_train, x_train)**0.5, MSE(output_test, x_opt)**0.5))  

            print("Calculating threshold")
            x_opt_predictions = sess.run(Y, feed_dict={x: x_opt, y: x_opt})
            print("Calculating MSE on optimization set...")
            mse = np.mean(np.power(x_opt - x_opt_predictions, 2), axis=1)
            print("mean is %.5f" % mse.mean())
            print("min is %.5f" % mse.min())
            print("max is %.5f" % mse.max())
            print("std is %.5f" % mse.std())
            
            tr = mse.mean() + mse.std()
            with open(f'threshold_{top_n_features}', 'w') as t:
                t.write(str(tr))
            print(f"Calculated threshold is {tr}")

            x_test_predictions = sess.run(Y, feed_dict={x: x_test, y: x_test})
            print("Calculating MSE on test set...")
            mse_test = np.mean(np.power(x_test - x_test_predictions, 2), axis=1)
            over_tr = mse_test > tr
            false_positives = sum(over_tr)
            test_size = mse_test.shape[0]
            print(f"{false_positives} false positives on dataset without attacks with size {test_size}")

            #------------------Testing--------------------------------------
            with open(f'threshold_{top_n_features}') as t1:
                tr1 = np.float64(t1.read())
            print(f"Calculated threshold test is {tr1}")
            x_pred1 = sess.run(Y, feed_dict={x: X_test_scaled1, y: X_test_scaled1})
            mse1 = np.mean(np.power(X_test_scaled1 - x_pred1, 2), axis=1)
            y_pred1 = mse1 > tr1
            Y_pred1 = y_pred1.astype(int)

            print('Accuracy')
            print(accuracy_score(Y_test1, Y_pred1))
            print('Recall')
            print(recall_score(Y_test1, Y_pred1))
            print('Precision')
            print(precision_score(Y_test1, Y_pred1))
            print(confusion_matrix(Y_test1, Y_pred1))

            end_time = timeit.default_timer()
            print("Time {0} minutes".format((end_time- start_time)/ 60.))  


