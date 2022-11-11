#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 17:16:15 2022

@author: carolinacaramelo
"""


import numpy as np
import pandas as pd

import scipy
from scipy.cluster.hierarchy import dendrogram,linkage

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA



# =============================================================================
# import sklearn
# from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
# import sklearn.metrics as sm
# from sklearn.preprocessing import scale
# =============================================================================
import pandarallel


import sgt
sgt.__version__
from sgt import SGT

# =============================================================================
# data = pd.DataFrame([["ch1",[2,3,4]],["ch2",[56,78,96]],["ch3",[90,312,56,390]],["ch4",[3,4,2]],["ch5",[90,23,12]],["ch6",[9,45,32,1000]]], columns=["id", "sequence"])
# data = list(zip(for l in data["sequence"]))
# 
# print(data)
# 
# 
# plt.figure(figsize=(10, 7))  
# plt.title("Dendrograms")  
# dend = shc.dendrogram(shc.linkage(data, method='ward'))
# =============================================================================




# A sample corpus of two sequences.
corpus = pd.DataFrame([["char_1", [2,3,4]], 
                       ["char_2", [56,78,96]], ["char_3", [90,312,56,390]], 
                       ["char_4",[3,4,2]], 
                       ["char_5",[90,23,12]],["char_6",[9,45,32,1000]],
                       ["char_7", [56,78,100,45]], ["char_8",[3,4]], 
                       ["char_9", [90,312,65,1212]], 
                       ["char_10",[90,312,90,23,9]]],
                      columns=['id', 'sequence'])

print(corpus)


sgt_ = SGT(kappa=1, 
           lengthsensitive=False, 
           mode='multiprocessing')
sgtembedding_df = sgt_.fit_transform(corpus)

print(sgtembedding_df)

sgtembedding_df = sgtembedding_df.set_index('id')
labels = sgtembedding_df.index
print(labels)

#performing PCA 
pca = PCA(n_components=2)
pca.fit(sgtembedding_df)

X=pca.transform(sgtembedding_df)

print(np.sum(pca.explained_variance_ratio_))
df = pd.DataFrame(data=X, columns=['x1', 'x2'])
print(df)

Z = linkage(sgtembedding_df, 'ward')
W =  linkage(df, 'ward')
 
print(Z)
print("Z shape", Z.shape)
  
fig, ax = plt.subplots(figsize=(12,8))
plt.title("Ward")
dendrogram(Z, labels=labels)      # Call dendrogram on Z
plt.show()

fig, ax = plt.subplots(figsize=(12,8))
plt.title("Ward")
dendrogram(W)      # Call dendrogram on Z
plt.show()


hierarchical_cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
labels = hierarchical_cluster.fit_predict(sgtembedding_df) 
print(labels)


plt.figure(figsize=(10, 7))  
plt.scatter(df['x1'], df['x2'], c=labels) 
plt.show()




