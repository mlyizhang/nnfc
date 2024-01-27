from utils import *
import os
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import random
import numpy
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import time
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
import math
dataname=['balancescale','segmentation','ecoli','zoo','thyroid','wine','banknote','sonar','liver','ionosphere'
          ,'leaf','liver','spambase','vehicle','Yeast']
dataname=['movement_libras','compound','Aggregation','jain','Pathbased','R15']
dataname=['wine','breast','ecoli','zoo','thyroid','seeds','abalone'
          ,'gesture','liver','ionosphere','heart','balancescale','vehicle','banknote','sonar' ,'leaf'
          ,'Yeast']
dataname=['iris']
for i in dataname:
    data_path = '../dataset/' + i + 'fed.pkl'
    print(data_path)
    ari=[]
    nmi=[]

    for i in range(1000):
        a,n,p=nnfc(data_path)
        ari.append(a)
        nmi.append(n)
    print('ari',max(ari), 'nmi',max(nmi),p)
    # 保存结果。
    with open('result.txt','a') as f:
        f.write(data_path)
        f.write('ari'+str(max(ari))+'nmi'+str(max(nmi))+str(p)+'\n')


