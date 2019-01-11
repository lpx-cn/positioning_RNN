# the relation between height and positioning error

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

def mkdir(newpath):
    isExist = os.path.exists(newpath)
    if isExist:
        print("the path exists!")
        return False
    else:
        print("the path is created successfully!")
        os.makedirs(newpath)
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

train_features = read_csv('train_features.csv')
train_labels = read_csv('train_labels.csv')
test_features = read_csv('test_features_height.csv')
test_labels = read_csv('test_labels_height.csv')

newpath = "./plot_loss_value/"
mkdir(newpath)

model = load_model('./keras_debug_adm_final_1/L=3,N=215,act=elu,dropout=0.2,lr=0.001_84.51871490478516.hdf5')


loss=[]
prediction = []
time_list=[]

data_value = test_features
label_value = test_labels
for i in range(len(data_value)):
    x = np.expand_dims(data_value[i], axis=0)

    time_start = time.time()
    predictation_i = model.predict(x)
    time_end=time.time()

    time_list.append(time_end-time_start)
    prediction.append(predictation_i)

    loss_i =  label_value[i] - predictation_i
    loss_i = np.linalg.norm(loss_i)

    loss.append(loss_i)
    print("*"*50,'\n step: %d'% i)
    print("the real value is: ", label_value[i])
    print("prediction value is : ", predictation_i)
    print("posisitoning error is : %f"%loss_i)
    
print("the mean loss is :%f" % (np.mean(loss)))
dataframe = pd.DataFrame({"test_loss":loss})
dataframe.to_csv(newpath + 'MPE.csv')


loss_height = []
meanloss_height = []
height = []
for i in range(9):
    height.append(test_labels[i*100,2])
    axis_i = np.arange(i*100, (i+1)*100)
    loss_height.append(np.array(loss)[axis_i])
    meanloss_height.append(np.mean(loss_height[i]))

dataframe = pd.DataFrame({"height":height,"test_loss_mean":meanloss_height})
dataframe.to_csv(newpath + 'MPE_height.csv')

print(np.mean(time_list))
