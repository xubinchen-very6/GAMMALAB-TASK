#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 14:24:40 2018

@author: bobo
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import sys

np.random.seed(20180702)

"""
从N个target中选出任意两个，将一个作为anchor point，另外一个混入distractor中构成test set
将test set中的每一个embedding依次于anchor point计算欧式距离
遍历所有可能性，返回距离最近的Top 1是target的平均准确度
"""
def inference(target_embeddings,distractor_embeddings,plot=False):
    
    N = len(target_embeddings)
    dist1 = cdist(target_embeddings,target_embeddings,'euclidean')
    dist2 = cdist(target_embeddings,distractor_embeddings,'euclidean')
    dist2_min = np.min(dist2,axis=1)
    res = np.less(dist1,dist2_min)
    acc = (np.sum(res) - N)/(N*(N-1))

    return acc

def sampling(embeddings,labels,threshold=0.05):
    N = len(embeddings)
    index = np.random.random_sample((N))<threshold
    embeddings = embeddings[index,:]
    labels = labels[index]
    return embeddings,labels
     
if __name__=='__main__':
    if len(sys.argv) < 2:
        print('Two paths are needed: (1) path_to_your_embeddings.csv (2) path_to_task3_test_label.csv ')
        sys.exit()
        
    df = pd.read_csv(sys.argv[1],header=-1)
    embeddings = df.values
    df = pd.read_csv(sys.argv[2],header=-1)
    labels = df.values
    labels = labels[0:len(embeddings),0]
    
    embeddings,labels = sampling(embeddings,labels)
    
    label_set = list(set(labels))
    np.random.shuffle(label_set)
        
    target = label_set[0:5]
    distractor = label_set[5:10]
    
    distractor_index = np.in1d(labels,distractor) 
    distractor_embeddings = embeddings[distractor_index,:]
        
    accuracy = []
    for i in target:
        target_index = np.equal(labels,i)
        target_embeddings = embeddings[target_index,:]
        res = inference(target_embeddings,distractor_embeddings)
        print("Average of accuracy for Position",i,"is:",res)
        accuracy.append(res)
    
    print("1:N average accuracy is:",np.mean(accuracy))