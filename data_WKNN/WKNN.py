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

def KNN(train_features, train_labels, K, feature):
    distance = []
    for i in range(len(train_features)):
        distance_i = train_features[i] - feature
        distance.append(np.linalg.norm(distance_i))
    distance_sort = np.sort(distance)
    wights = 1/ distance_sort
    index = np.argsort(distance)
    # for i in range(K):
        # print(train_labels[index[i]],distance[index[i]])

    wights_sum = np.sum(wights[0:K])
    position = np.zeros(shape= [3])
    for i in range(K):
        w = wights[i]/wights_sum
        position += np.array(train_labels[index[i]]) * w
    return position



train_features = read_csv('train_features.csv')
train_labels = read_csv('train_labels.csv')
t_features = read_csv('test_features_height.csv')
t_labels = read_csv('test_labels_height.csv')

K=8
for K in [5]:
    prediction = []
    loss = []
    for i in range(len(t_features)):
        # print("*"*50)
        prediction_i = KNN(train_features, train_labels, K, t_features[i])
        prediction.append(prediction_i)
        # print("Real position is:",t_labels[i],", prediction is :",prediction_i)
        loss_i =prediction_i - t_labels[i]
        loss_i = np.linalg.norm(loss_i)
        loss.append(np.linalg.norm(loss_i))
    dataframe = pd.DataFrame({"test_loss":loss})
    dataframe.to_csv("./MPE.csv")

    loss_mean = np.mean(loss)
    print("K=",K,",loss mean is:", loss_mean)
    
    
time_end = time.time()

print((time_end - time_start)/3600)
