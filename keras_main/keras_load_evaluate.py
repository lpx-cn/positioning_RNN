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
from keras.optimizers import Adam,TFOptimizer
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint

from keras.models import load_model

def read_csv(filename):
    data = []
    f = open(filename,'r')
    line_reader=csv.reader(f)

    for row in line_reader:
        data.append(row)
    data = np.array(data, dtype = float)

    f.close
    return data

train_features = read_csv('train_features.csv')
train_labels = read_csv('train_labels.csv')
t_features = read_csv('test_features.csv')
t_labels = read_csv('test_labels.csv')

print(t_labels)

test_features = t_features[np.arange(0,len(t_features),2)]
test_labels = t_labels[np.arange(0,len(t_labels),2)]
validation_features = t_features[np.arange(1, len(t_features), 2)]
validation_labels = t_labels[np.arange(1, len(t_labels), 2)]



model = load_model('./keras_debug_adm_final_1/L=3,N=215,act=elu,dropout=0.2,lr=0.001_84.51871490478516.hdf5')
test_loss = model.evaluate(test_features,test_labels,batch_size = test_labels.shape[0],verbose =0)
val_loss = model.evaluate(validation_features,validation_labels,
                          batch_size = validation_features.shape[0],verbose = 0)


loss=[]
loss_square = []
prediction = []


data_value = test_features
label_value = test_labels
time_list = []
for i in range(len(data_value)):
    time_start = time.time()
    x = np.expand_dims(data_value[i], axis=0)
    predictation_i = model.predict(x)
    prediction.append(predictation_i)
    time_end = time.time()
    time_list.append(time_end - time_start)

    loss_i =  label_value[i] - predictation_i
    loss_i = np.linalg.norm(loss_i)

    loss_square.append(loss_i**2)
    loss.append(loss_i)

    print("*"*50,'\n step: %d'% i)
    print("the real value is: ", label_value[i])
    print("prediction value is : ", predictation_i)
    print("posisitoning error is : %f"%loss_i)
    
# print(loss)
print("the mean loss is :%f" % (np.mean(loss)))
print('test_loss:',test_loss)
print('validation_loss:',val_loss)

print(time_list)
print(np.mean(time_list))
