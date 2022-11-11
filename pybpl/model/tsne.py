#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 14:52:33 2022

@author: carolinacaramelo
"""

from __future__ import print_function
import time
import numpy as np 
import pandas as pd 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
import seaborn as sns 
from PIL import Image
from sklearn.datasets import fetch_openml
from pathlib import Path
import cv2
import glob
import os 
from sklearn.metrics import pairwise_distances
from scipy.stats import t, entropy
from sklearn.manifold import _t_sne
from sklearn.decomposition import PCA


def dataset():
    directory = "/Users/carolinacaramelo/Desktop/TSNE"
    #appending the pics to the training data list
    training_data = []
    labels = []
    
    for path in os.listdir(directory):
            path = os.path.join(directory, path)
            print(path)
            try:
         
                for path2 in os.listdir(path):
                    path2 = os.path.join(path, path2)
                    print(path2)
                    try:
                  
                        for path3 in os.listdir(path2):
                            path3 = os.path.join(path2, path3)
                            print(path3)
                            try:
                         
                                for images in os.listdir(path3):
                                    if (images.endswith(".png")):
                                        image = Image.open(os.path.join(path3,images))
                                        image = image.convert('RGBA')
                                        arr = np.array(image)
                                        arr = arr.reshape(44100)
                                        training_data.append(arr)
                                        label= os.path.split(path)[1]
                                        label = str(label)
                                        labels.append(label)  
                            except:
                                pass
                    except:
                        pass
            except:
                pass


    training_data = np.array(training_data)
    labels = np.array(labels)
    print(np.unique(labels))
            
    return training_data, labels

def tsne():
    X = dataset()[0]
    y = dataset()[1]
    n_components = 2
    tsne = TSNE(n_components, perplexity=50)
    tsne_result = tsne.fit_transform(X)
    tsne_result.shape
    tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': y})
    fig, ax = plt.subplots(1, figsize=(10,10))
    sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax,s=70)
    lim = (tsne_result.min()-5, tsne_result.max()+5)
    plt.title("t-SNE results", size=15)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    
    return tsne_result
    


def distance_matrix():
    X = dataset()[0]
    y = dataset()[1]
    y_sorted_idc = y.argsort()
    X_sorted = X[y_sorted_idc]
 
    distance_matrix = pairwise_distances(X,
                                     metric='euclidean')
 
    distance_matrix_sorted = pairwise_distances(X_sorted,
                                            metric='euclidean')
     
    fig, ax = plt.subplots(1,2, figsize=(10,10))
    ax[0].imshow(distance_matrix, 'Greys')
    ax[1].imshow(distance_matrix_sorted, 'Greys')
    ax[0].set_title("Unsorted")
    ax[1].set_title("Sorted by Label")
    
    return distance_matrix
        
def distribution():
    x = distance_matrix()[0,1:]
    t_dist_sigma01 = t(df=1.0, loc=0.0, scale=1.0)
    t_dist_sigma10 = t(df=1.0, loc=0.0, scale=10.0)
    P_01 = t_dist_sigma01.pdf(x)
    P_10 = t_dist_sigma10.pdf(x)
     
    perplexity_01 = 2**entropy(P_01)
    perplexity_10 = 2**entropy(P_10)
     
    dist_min = min(P_01.min(), P_10.min())
    dist_max = max(P_01.max(), P_10.max())
    bin_size = (dist_max - dist_min) / 100
    bins = np.arange(dist_min+bin_size/2, dist_max+bin_size/2, bin_size)
    fig, ax = plt.subplots(1, figsize=(10,10))
    ax.hist(P_01, bins=bins)
    ax.hist(P_10, bins=bins)
    ax.set_xlim((0, 0.4e-6))
    ax.legend((r'$\sigma = 01; Perplexity = $' + str(perplexity_01),
               r'$\sigma = 10; Perplexity = $' + str(perplexity_10)))
    
     
    perplexity = 30  # Same as the default perplexity
    p = _t_sne._joint_probabilities(distances=distance_matrix(),
                            desired_perplexity = perplexity,
                            verbose=False)

    return p

def tsne_optimized():
    X = dataset()[0]
    y = dataset()[1]
    p = distribution()
    
    # Create the initial embedding
    n_samples = X.shape[0]
    n_components = 2
    X_embedded = 1e-4 * np.random.randn(n_samples,
                                        n_components).astype(np.float32)
     
    embedding_init = X_embedded.ravel()  # Flatten the two dimensional array to 1D
     
    # kl_kwargs defines the arguments that are passed down to _kl_divergence
    kl_kwargs = {'P': p,
                 'degrees_of_freedom': 1,
                 'n_samples': n_samples,
                 'n_components':2}
     
    # Perform gradient descent
    embedding_done = _t_sne._gradient_descent(_t_sne._kl_divergence,
                                              embedding_init,
                                              0,
                                              1000,
                                              kwargs=kl_kwargs)
     
    # Get first and second TSNE components into a 2D array
    tsne_result = embedding_done[0].reshape(n_samples,2)
     
    # Convert do DataFrame and plot
    tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0],
                                   'tsne_2': tsne_result[:,1],
                                   'label': y})
    fig, ax = plt.subplots(1, figsize=(10,10))
    sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax,s=120)
    lim = (tsne_result.min()-5, tsne_result.max()+5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)


def pca_tsne():
    # get dataset
    X = dataset()[0]
    y = dataset()[1]
    
    # first reduce dimensionality before feeding to t-sne
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X)
    
    # randomly sample data to run quickly
    rows = np.arange(25000)
    np.random.shuffle(rows)
    n_select = 10000
    
    # reduce dimensionality with t-sne
    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000,
    learning_rate=200)
    tsne_result = tsne.fit_transform(X_pca[rows[:n_select],:])
    
    # visualize
    tsne_result.shape
    tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': y})
    fig, ax = plt.subplots(1, figsize=(10,10))
    sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax,s=30)
    lim = (tsne_result.min()-5, tsne_result.max()+5)
    plt.title("t-SNE results", size=15)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    
    return tsne_result

def tsne2 (X_pca):
    # get dataset
    X = dataset()[0]
    y = dataset()[1]
    X_pca = X_pca
    # randomly sample data to run quickly
    rows = np.arange(25000)
    np.random.shuffle(rows)
    n_select = 10000
    # reduce dimensionality with t-sne
    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000,
    learning_rate=200)
    tsne_result = tsne.fit_transform(X_pca[rows[:n_select],:])
    
    # visualize
    tsne_result.shape
    tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': y})
    fig, ax = plt.subplots(1, figsize=(10,10))
    sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax,s=30)
    lim = (tsne_result.min()-5, tsne_result.max()+5)
    plt.title("t-SNE results", size=15)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    
    return tsne_result
    