#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 12:32:11 2022

@author: carolinacaramelo
"""
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import sgt
from sgt import SGT
import pandarallel 
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc



corpus = pd.DataFrame([["character1_perturbed", [2,3,4]], 
                        ["character1_perturbed", [2,3,4]], 
                        ["character1_perturbed", [2,3,4]],
                        ["character2_perturbed", [222,332,41]],
                        ["character2_perturbed", [222,321,41]],
                        ["character3_perturbed", [89,103,456,678]],
                        ["character3_perturbed", [88,103,456,678]],
                        ["character1_omniglot", [5]],
                        ["character1_omniglot", [5]],
                        ["character2_omniglot", [1212,465,900,12,32]],
                        ["character2_omniglot", [1212,465,900,10,23]],
                        ["character2_omniglot", [1212,465,900,10,780]],
                        ["character3_omniglot", [222,332,432,100,176]],
                        ["character3_omniglot", [222,300,432,100,176]],
                        ["character4_omniglot", [2,10]]], columns=['id', 'sequence'])


corpus = corpus.drop('id', axis=1)
# =============================================================================
# sgt_ = SGT(kappa=1, 
#            lengthsensitive=False, 
#            mode='multiprocessing')
# sgtembedding_df = sgt_.fit_transform(corpus)
# 
# print(sgtembedding_df)
# 
# sgtembedding_df = sgtembedding_df.set_index('id')
# sgtembedding_df
# 
# pca = PCA(n_components=2)
# pca.fit(sgtembedding_df)
# 
# X=pca.transform(sgtembedding_df)
# 
# print(np.sum(pca.explained_variance_ratio_))
# df = pd.DataFrame(data=X, columns=['x1', 'x2'])
# print(df.head())
# 
# kmeans = KMeans(n_clusters=3, max_iter =300)
# kmeans.fit(df)
# 
# labels = kmeans.predict(df)
# centroids = kmeans.cluster_centers_
# 
# fig = plt.figure(figsize=(5, 5))
# colmap = {1: 'r', 2: 'g', 3: 'b'}
# colors = list(map(lambda x: colmap[x+1], labels))
# plt.scatter(df['x1'], df['x2'], color=colors, alpha=0.5, edgecolor=colors)
# =============================================================================

plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(corpus, method='ward'))