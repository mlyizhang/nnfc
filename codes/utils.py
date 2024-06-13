#! /usr/bin/env python
# -*- coding: utf-8 -*-

#############################
# utils files.
#############################
from numpy import *
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import json
import pickle
import math
from utils import *
import os
from sklearn.cluster import DBSCAN
import random
import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score
from typing import List, Tuple
from numpy import arange, argsort, argwhere, empty, full, inf, intersect1d, max, ndarray, sort, sum, zeros
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
import math


def load_dataset(filepath):
    """
        Return:
            dataset: dict
    """
    with open(filepath, 'rb') as fr:
        dataset = pickle.load(fr)
    return dataset
# 按行的方式计算两个坐标点之间的距离
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

def SNN(k: int, nc: int, data: ndarray) -> Tuple[ndarray, ndarray]:
	unassigned = -1
	n, d = data.shape

	# Compute distance
	# --------------------------------------------------------------------------------

	distance = squareform(pdist(data))

	# Compute neighbor
	# --------------------------------------------------------------------------------

	indexDistanceAsc: ndarray = argsort(distance)
	indexNeighbor: ndarray = indexDistanceAsc[:, :k]

	# Compute shared neighbor
	# --------------------------------------------------------------------------------

	indexSharedNeighbor = empty([n, n, k], int)
	numSharedNeighbor = empty([n, n], int)
	for i in range(n):
		numSharedNeighbor[i, i] = 0
		for j in range(i):
			shared: ndarray = intersect1d(indexNeighbor[i], indexNeighbor[j], assume_unique=True)
			numSharedNeighbor[j, i] = numSharedNeighbor[i, j] = shared.size
			indexSharedNeighbor[j, i, :shared.size] = indexSharedNeighbor[i, j, :shared.size] = shared

	# Compute similarity
	# --------------------------------------------------------------------------------

	similarity = zeros([n, n])  # Diagonal and some elements are 0
	for i in range(n):
		for j in range(i):
			if i in indexSharedNeighbor[i, j] and j in indexSharedNeighbor[i, j]:
				indexShared = indexSharedNeighbor[i, j, :numSharedNeighbor[i, j]]
				distanceSum = sum(distance[i, indexShared] + distance[j, indexShared])
				similarity[i, j] = similarity[j, i] = numSharedNeighbor[i, j] ** 2 / distanceSum

	# Compute ρ
	# --------------------------------------------------------------------------------

	rho = sum(sort(similarity)[:, -k:], axis=1)

	# Compute δ
	# --------------------------------------------------------------------------------

	distanceNeighborSum = empty(n)
	for i in range(n):
		distanceNeighborSum[i] = sum(distance[i, indexNeighbor[i]])
	indexRhoDesc = argsort(rho)[::-1]
	delta = full(n, inf)
	for i, a in enumerate(indexRhoDesc[1:], 1):
		for b in indexRhoDesc[:i]:
			delta[a] = min(delta[a], distance[a, b] * (distanceNeighborSum[a] + distanceNeighborSum[b]))
	delta[indexRhoDesc[0]] = -inf
	delta[indexRhoDesc[0]] = max(delta)

	# Compute γ
	# --------------------------------------------------------------------------------
	#rho=rho+numpy.random.laplace(0,10,len(data))
	#delta=+numpy.random.laplace(0,10,len(data))

	gamma = rho * delta


	# Compute centroid
	# --------------------------------------------------------------------------------

	indexAssignment = full(n, unassigned)
	indexCentroid: ndarray = sort(argsort(gamma)[-nc:])
	indexAssignment[indexCentroid] = arange(nc)

	# Assign non-centroid step 1
	# --------------------------------------------------------------------------------

	queue: List[int] = indexCentroid.tolist()
	while queue:
		a = queue.pop(0)
		for b in indexNeighbor[a]:
			if indexAssignment[b] == unassigned and numSharedNeighbor[a, b] >= k / 2:
				indexAssignment[b] = indexAssignment[a]
				queue.append(b)

	# Assign non-centroid step 2
	# --------------------------------------------------------------------------------

	indexUnassigned = argwhere(indexAssignment == unassigned).flatten()
	while indexUnassigned.size:
		numNeighborAssignment = zeros([indexUnassigned.size, nc], int)
		for i, a in enumerate(indexUnassigned):
			for b in indexDistanceAsc[a, :k]:
				if indexAssignment[b] != unassigned:
					numNeighborAssignment[i, indexAssignment[b]] += 1
		if most := max(numNeighborAssignment):
			temp = argwhere(numNeighborAssignment == most)
			indexAssignment[indexUnassigned[temp[:, 0]]] = temp[:, 1]
			indexUnassigned = argwhere(indexAssignment == unassigned).flatten()
		else:
			k += 1

	return indexCentroid, indexAssignment






import numpy as np


# 定义拉普拉斯噪声的函数
def add_laplace_noise(data, epsilon, sensitivity):
    """
    添加拉普拉斯噪声
    :param data: 原始数据
    :param epsilon: 隐私预算
    :param sensitivity: 数据的敏感性
    :return: 添加噪声后的数据
    """
    # 计算拉普拉斯分布的scale参数
    scale = sensitivity / epsilon
    # 生成与数据相同大小的拉普拉斯噪声
    noise = np.random.laplace(0, scale, data.shape)

    return data + noise


# 假设我们有以下原始数据
#data = np.array([10, 20, 30, 40, 50])

# 隐私预算和数据的敏感性
#epsilon = 1.0
#sensitivity = 1.0

# 添加拉普拉斯噪声
#noisy_data = add_laplace_noise(data, epsilon, sensitivity)
#print("Original Data: ", data)
#print("Noisy Data: ", noisy_data)

def nnfc(data_path):
    parameters = []
    datapkl = load_dataset(data_path)  # dataset is a json file
    eachlable = datapkl['eachlable']
    order = datapkl['order']
    data = list(datapkl['full_data'])
    allgamme = np.zeros((1, len(data)))[0]
    corepoints = []
    client_results = []
    for i_client in range(len(list(order))):
        # add noise to local data (Differential privacy)

        lodata = datapkl["client_" + str(i_client)]

        noise = np.random.uniform(0, 1, size=lodata.shape)

        euc = euclidean_distances(lodata)
        row, col = np.diag_indices_from(euc)
        euc[row, col] = np.max(euc)
        minvalue = np.min(euc)
        ratio = []
        noisenew = noise
        for i in range(noisenew.shape[0]):
            for j in range(noisenew.shape[1]):
                if noisenew[i][j] >= 0.5:
                    noisenew[i][j] = 1 - noisenew[i][j]

        arr = noisenew.reshape(1, noise.shape[0] * noise.shape[1])[0]
        ratio = []
        for i in arr:
            for j in arr:
		num1,num2=float(i)/float(j),float(j )/float(i)
                ratio.append(max([num1,num2]))    
        maxratio = max(ratio)
        epsilon = 1.01 * maxratio / minvalue

        scale = 1 / epsilon
        # generate laplacian noises
        for i in range(noise.shape[0]):
            for j in range(noise.shape[1]):
                if noise[i][j] >= 0.5:
                    noise[i][j] = noise[i][j] - scale * math.log(2 * (1 - noise[i][j]))

                else:
                    noise[i][j] = noise[i][j] + scale * math.log(2 * (noise[i][j]))

        lodata = lodata + noise
        #corepoints.append(lodata)
        n_clusters = min([len(lodata) // 3, 50])
        parameters.append(n_clusters)
        cluster = KMeans(n_clusters).fit(lodata)

        corepoints.append(cluster.cluster_centers_)
    serverdata = np.concatenate(corepoints, axis=0)
        # 对加密的数据进行代表点提取


    #serverdata = []
    #for i in corepoints:
    #    for j in i:
    #        serverdata.append(j)
    #serverdata = np.array(serverdata)

    label = datapkl['true_label']
    # parameters.append(n_clusters)
    cnum = len(set(label))
    # print('最终的类中心个数，',cnum)
    parameters.append(cnum)
    k = 2
    parameters.append(k)
    # print('snndpc中的参数k',k )
    centroid, assignment = SNN(k, cnum, serverdata)
    # finalcenter = cluster2.cluster_centers_
    # finalcenter = cluster2.cluster_centers_
    finalcenter = []
    ii = 0
    for i in centroid:
        finalcenter.append(serverdata[i])
        ii = ii + 1
    # 根据finalcenter分配所有客户端的数据点。
    idx = []
    for i in data:
        simi = []
        for j in finalcenter:
            simi.append(np.linalg.norm(i - j))
        idx.append(simi.index(min(simi)) + 1)

    arr = np.array(idx)
    ari = round(adjusted_rand_score(label, arr), 4)
    nmi = round(normalized_mutual_info_score(label, arr), 4)
    return ari,nmi,parameters
