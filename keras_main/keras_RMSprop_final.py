import os
import numpy as np
from sklearn import preprocessing
import csv 
import time
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from keras import Input,Model
from keras.models import Sequential,save_model
from keras.layers import Dense,Activation,Dropout
from keras.optimizers import Adam,TFOptimizer,RMSprop
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

time_start = time.time()

def mkdir(path):
    isExist = os.path.exists(path)
    if isExist:
        print("the path exist")
        return False
    else:
        os.makedirs(path)
        print("the path is created successfully!")
        return True

def read_csv(filename):
    data = []
    f = open(filename,'r')
    line_reader=csv.reader(f)

    for row in line_reader:
        data.append(row)
    data = np.array(data, dtype = float)

    f.close
    return data

def add_layer(inputs, L, N, act='None',dropout=0):
    x=Dense(N, activation=act)(inputs)
    for i in  range(L-3):
        x=Dense(N, activation=act)(x)
    x=Dense(N)(x)
    Dropout(dropout)
    predictions = Dense(3)(x)
    return predictions

def keras_debug(train_features,train_labels,test_features,test_labels,
        L,N,act='None',dropout=0,lr=1e-3, newpath = "./keras_debug/"):
    inputs = Input(shape=(6,))
    predictions = add_layer(inputs,L,N,act,dropout)
    model = Model(inputs=inputs, outputs = predictions)
    
    op = RMSprop(lr)
    model.compile(optimizer=op, loss='mse')
    
    title="L=%d,N=%d,act=%s,dropout=%.1f,lr=%.5f"%(L,N,act,dropout,lr)
    # print(title)
    # checkpoint = ModelCheckpoint(filepath = newpath + title + "_best.hdf5",
            # monitor = 'val_loss', verbose = 0, 
            # save_best_only = True, mode = 'min')

    # train_step = model.fit(train_features, train_labels, epochs = 3000 *128, 
            # validation_data=(validation_features, validation_labels),
            # callbacks = [checkpoint],
            # batch_size = train_labels.shape[0], verbose = 0)

    # train_loss = train_step.history['loss']
    # validation_loss = train_step.history['val_loss']

    # val_minloss = min(validation_loss)
    # os.rename(newpath + title + "_best.hdf5",newpath + title +"_" +str(val_minloss)+".hdf5")
    # dataframe = pd.DataFrame({'train_loss':train_loss,'validation_loss':validation_loss})
    # dataframe.to_csv(newpath+"/"+title+'.csv',sep=',')

    # plt.figure()
    # plt.title(title)
    # l1, = plt.plot(train_loss)
    # l2, = plt.plot(validation_loss)
    # plt.ylim(0,500)
    # plt.legend(handles=[l1,l2],labels=['train_loss','validation_loss'],loc = 'best')
    # plt.show()
    # plt.savefig(newpath+title+".png")

    train_loss = []
    validation_loss = []
    test_loss = []

    for i in range(15000 * 128):

        train_step = model.fit(train_features,train_labels,
                validation_data = (validation_features, validation_labels),
                epochs=1,batch_size=train_labels.shape[0],verbose=0)
         
        loss_value = train_step.history['loss'][0]
        val_loss_value = train_step.history['val_loss'][0]
        
        scope = model.evaluate(test_features,test_labels,batch_size=test_labels.shape[0],verbose=0)
        train_loss.append(loss_value)
        validation_loss.append(val_loss_value)
        test_loss.append(scope)
        
        if i==0:
            model.save(newpath + title + "_" + str(i) + "_"+ str(val_loss_value) +".hdf5")
            val_minloss = val_loss_value
            num = i
            
        if val_loss_value  < val_minloss:
            model.save(newpath + title + "_" + str(i) + "_"+ str(val_loss_value) +".hdf5")
            os.remove(newpath + title + "_" + str(num) + "_"+ str(val_minloss) + '.hdf5')
            val_minloss = val_loss_value 
            num=i
            
        if i %128 ==0:
            print("*"*50,'\n',title,"\n step: ",i)
            print('train_loss:', train_step.history['loss'][0]) 
            print('val_loss:', train_step.history['val_loss'][0])
            print('test_loss:',scope)



    plot_model(model ,to_file=newpath + title +'_model.png')


    dataframe = pd.DataFrame({'train_loss':train_loss,'validation_loss':validation_loss,'test_loss':test_loss})
    dataframe.to_csv(newpath+"/"+title+'.csv',sep=',')

    plt.figure()
    plt.title(title)
    l1, = plt.plot(train_loss)
    l2, = plt.plot(validation_loss)
    l3, = plt.plot(test_loss)
    plt.ylim(0,500)
    plt.legend(handles=[l1,l2,l3],labels=['train_loss','validation_loss','test_loss'],loc = 'best')
    plt.show()
    plt.savefig(newpath+title+".png")

train_features = read_csv('train_features.csv')
train_labels = read_csv('train_labels.csv')
t_features = read_csv('test_features.csv')
t_labels = read_csv('test_labels.csv')

test_features = t_features[np.arange(0,len(t_features),2)]
test_labels = t_labels[np.arange(0,len(t_labels),2)]
validation_features = t_features[np.arange(1, len(t_features), 2)]
validation_labels = t_labels[np.arange(1, len(t_labels), 2)]


newpath = "./keras_debug_RMSprop_final_4/"
mkdir(newpath)

for N in [215]:
    for dropout in [0,0.2,0.3,0.4,0.5]:
         keras_debug(train_features,train_labels,test_features,test_labels,
                3,N,act = 'elu',dropout=dropout,newpath = newpath,lr=1e-3)

time_end = time.time()

print((time_end - time_start)/3600)
