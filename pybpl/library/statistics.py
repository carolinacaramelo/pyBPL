#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 12:03:15 2022

@author: carolinacaramelo
"""

from pybpl.library import Library
import numpy as np 
import matplotlib.pyplot as plt
import torch
from scipy.stats import gaussian_kde
from plotnine import ggplot, aes, geom_bar, geom_density, geom_histogram
import pandas as pd
from scipy.interpolate import UnivariateSpline
import scipy.interpolate as inter
import seaborn as sns
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV


def flatten(l):
    return [item for sublist in l for item in sublist]


def scatter_start():
    lib = Library (use_hist= True)
    x = np.linspace(1, 1212, 1212)
    
    print ("x done")
    y= lib.logStart
    y1 = torch.exp(y)
    maxi = torch.max(y1)
    mini= torch.min(y1)
    mean = torch.mean(y1)
    median = torch.median(y1)
    y = y1.numpy()
    
    #median = torch.median(y)
    print ("y done")
    print ("starting plot..")
    
    # Calculate the point density
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    
    #scatter plot with density
    #devia fazer legenda
    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(x, y, c=z, s=50)
    ax.set_xlabel('Transitions')
    ax.set_ylabel('Probability')
    
    plt.show()
    
    #scatter plot 

    plt.scatter (x,y, s =1)
    plt.plot(x,y)
    
    plt.xlabel('Transitions')
    plt.ylabel('Probability')
    plt.show()
    
    np.savetxt("./logstart",y)
    file = open("./statistics", 'w') 
    file.write("logStart statistics \n")
  
    file.write("Maximum prob" + str(maxi)+ "\n")
    file.write("Minimum prob" + str(mini)+ "\n")
    file.write("Mean" + str(mean)+ "\n")
    file.write("Median" + str(median)+ "\n")
    file.write("logStart" + str(y1)+ "\n")
    file.close()
    

 
def statistics_pT():
    lib = Library (use_hist= True)
    

    x = np.linspace(1, 1212*1212, 1212*1212)
    
    y= lib.logT
    y = torch.exp(y)
    y1 =(y/torch.sum(y))
    maxi = torch.max(y1)
    mini= torch.min(y1)
    mean = torch.mean(y1)
    median = torch.median(y1)
    
    y = y1.numpy()
    y = y.flatten()
    
    
    df = pd.DataFrame({'x':x, 'y':y})
    print(df)
    print(df.quantile([0.25,0.5,0.75]))
    
    quart_1 = df[df['x']<= int(df.quantile(0.25)[0])]
    print(quart_1)
    
    quart_2 = df[df['x']<= int(df.quantile(0.5)[0])]
    print(quart_2)
    
    quart_3 = df[df['x']<= int(df.quantile(0.75)[0])]
    print(quart_3)
    
    quart_4 = df[df['x']<= int(df.quantile(0.1)[0])]
    print(quart_4)
    
    x1 = int(df.quantile(0.25)[0])
    x2 = int(df.quantile(0.5)[0])
    x3= int(df.quantile(0.75)[0])

    
    
    # Calculate the point density
    #xy = np.vstack([x,y])
    #z = gaussian_kde(xy)(xy)
    
    # Sort the points by density, so that the densest points are plotted last
    #idx = z.argsort()
    #x, y, z = x[idx], y[idx], z[idx]
    
    #scatter plot with density
    #devia fazer legenda
    #fig, ax = plt.subplots(figsize=(40,40))
    #ax.scatter(x, y, c=z, s=50)
    #plt.xlabel('Transitions')
    #plt.ylabel('Probability')
    #plt.show()
    f, ax = plt.subplots(figsize=(20,20))
    plt.scatter (x,y, s =1)
    plt.plot(x,y)
    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)
    plt.axvline(x = x1, color = 'k', label = 'Q1')
    plt.axvline(x = x2, color = 'k', label = 'Q2')
    plt.axvline(x = x3, color = 'k', label = 'Q3')
    plt.xlabel('Transitions')
    plt.ylabel('Probability')
    plt.show()
    
    file = open("./statistics", 'w') 
    file.write("pT statistics \n")
    file.write("First quartile" + str(quart_1)+ "\n")
    file.write("Second quartile" + str(quart_2) + "\n")
    file.write("Third quartile" + str(quart_3)+ "\n")
    file.write("Fourth quartile" + str(quart_4)+ "\n")
    file.write("Maximum prob" + str(maxi)+ "\n")
    file.write("Minimum prob" + str(mini)+ "\n")
    file.write("Mean" + str(mean)+ "\n")
    file.write("Median" + str(median)+ "\n")
    file.write("pT" + str(y1)+ "\n")
    file.close()
    np.savetxt("./pT", y1)
   
        
    
    
def hist_start():
    lib = Library (use_hist= True)
    y= lib.logStart
    y1 = torch.exp(y)
    #y1 =(y/torch.sum(y))
    y = y1.numpy()
    y = y.flatten()
     
    
    bins =[20,500,1212]
    
    for i in bins:
        # Creating histogram
        fig, axs = plt.subplots(1, 1, figsize =(10, 10), tight_layout = True)
        axs.hist(y, bins= i, density= True)
        plt.xlabel("Magnitude")
        plt.ylabel("Counts")
        plt.title('Probability distribution')
        
         
        # Show plot
        plt.show()
 
def hist_pT():
    lib = Library (use_hist= True)
    y = lib.logT
    y = torch.exp(y)
    y1 = (y/torch.sum(y))
    y = y1.numpy()
    y = y.flatten()
    bins = int(np.sqrt(y.size))
    
    
    xlim = [0.00002, 0.000002, 0.000001]
    for i in range(len(xlim)):
        # all values counts
        fig, ax = plt.subplots(figsize=(10,10))
        plt.hist(y, bins= bins)
        plt.xlabel("Magnitude")
        plt.ylabel("Counts")
        plt.title('Histogram pT matrix magnitudes')
        plt.xlim(0, xlim[i])
        
        # Show plot
        plt.show()
    
    bins = [10000, 100000, 500000]
    for i in range(len(bins)):
         # all values counts
        fig, ax = plt.subplots(figsize=(10,10))
        plt.hist(y, bins= bins[i])
        plt.xlabel("Magnitude")
        plt.ylabel("Counts")
        plt.title('Histogram pT matrix magnitudes')
        plt.xlim(0, 0.000001)
        
        # Show plot
        plt.show()
        
def hist_norm_pT():
    lib = Library (use_hist= True)
    y = lib.logT
    y = torch.exp(y)
    y1 = (y/torch.sum(y))
    y = y1.numpy()
    y = y.flatten()
    bins = [10000, 100000, 500000]
    for i in range(len(bins)):
        fig, ax = plt.subplots(figsize=(10, 10))
        
        sns.histplot(y, stat='probability', ax=ax, bins=bins[i])
        plt.xlim(0,0.000001)
        plt.title('Histogram pT matrix magnitudes')
        
        
        quant_5, quant_25, quant_50, quant_75, quant_95 = np.quantile(y,0.05), np.quantile(y,0.25), np.quantile(y,0.5), np.quantile(y,0.75), np.quantile(y,0.95)
        quants = [[quant_5,  0.46], [quant_25,  0.56], [quant_50,  0.56],  [quant_75,  0.76], [quant_95, 0.86]]
        
        for i in quants:
            ax.axvline(i[0], ymax = i[1], linestyle = ":", color="red")
    
    print(quant_5, quant_25, quant_50, quant_75, quant_95) 
    
    
def perturb_quartile1():
    lib = Library (use_hist= True)
    y = lib.logT
    y = torch.exp(y)
    y1 = (y/torch.sum(y))
    y = y1.numpy()
    y = y.flatten()
    q1 = np.quantile(y,0.25)
    q2 = np.quantile(y, 0.5)
    q3 = np.quantile (y, 0.75)
    q1_values = y <= q1
    q1_values = q1_values.nonzero()[0]
    q2_values = y <= q2
    q2_values = q2_values.nonzero()[0]
    q3_values = y <= q3
    q3_values = q3_values.nonzero()[0]
    prob_count = 0
    for  i in q1_values:
        prob_count += y[i]
    print("prob count", prob_count)
    
    size = q1_values.size #number of samples to draw 
    print(size)
    
    new_values = np.random.normal(loc=0, scale =1, size=size)
    new_values= np.exp(new_values)
    new_values = new_values/new_values.sum()
    print ("new_values matrix" , new_values)
    print ("new_values size" , new_values.size)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(new_values, stat='probability', ax=ax, bins=20000)
    plt.title('Histogram new_values')
    plt.xlim(0,0.00004)
    
    new_pT = np.copy(y)
    
    #create dictionary for array replacement 
    
    dic = {}
    for i in range(len(q1_values)):
        dic[q1_values[i]]= new_values[i]
    
    #change quantile 1 values in the new pT matrix 
    for k, v in dic.items(): new_pT[k] = v
    
    #normalize new pT matrix 
    new_pT= new_pT/new_pT.sum()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(new_pT, stat='probability', ax=ax, bins=10000)
    plt.title('Histogram new_pT')
    
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(new_pT, stat='probability', ax=ax, bins=100000)
    plt.xlim(0,0.000001)
    
    plt.title('Histogram new_pT')
    
    print (" new_pT matrix" , new_pT)
    print (" new_pT matrix sum" , new_pT.sum())
    print (" new_pT matrix size" , new_pT.size)

def perturb_quartile2():
    lib = Library (use_hist= True)
    y = lib.logT
    y = torch.exp(y)
    y1 = (y/torch.sum(y))
    y = y1.numpy()
    y = y.flatten()
    q2 = np.quantile(y, 0.5)
    q2_values = y <= q2
    q2_values = q2_values.nonzero()[0]
    prob_count = 0
    for  i in q2_values:
        prob_count += y[i]
    print("prob count", prob_count)
    
    size = q2_values.size #number of samples to draw 
    print(size)
    
    new_values = np.random.normal(loc=0, scale =1, size=size)
    new_values= np.exp(new_values)
    new_values = new_values/new_values.sum()
    print ("new_values matrix" , new_values)
    print ("new_values size" , new_values.size)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(new_values, stat='probability', ax=ax, bins=20000)
    plt.title('Histogram new_values')
    plt.xlim(0,0.00004)
    
    new_pT = np.copy(y)
    
    #create dictionary for array replacement 
    
    dic = {}
    for i in range(len(q2_values)):
        dic[q2_values[i]]= new_values[i]
    
    #change quantile 1 values in the new pT matrix 
    for k, v in dic.items(): new_pT[k] = v
    
    #normalize new pT matrix 
    new_pT= new_pT/new_pT.sum()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(new_pT, stat='probability', ax=ax, bins=10000)
    plt.title('Histogram new_pT')
    
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(new_pT, stat='probability', ax=ax, bins=100000)
    plt.xlim(0,0.000001)
    
    plt.title('Histogram new_pT')
    
    print (" new_pT matrix" , new_pT)
    print (" new_pT matrix sum" , new_pT.sum())
    print (" new_pT matrix size" , new_pT.size)

def perturb_flattening():
    #logT matrix
    lib = Library (use_hist= True)
    y = lib.logT
    y = torch.exp(y)
    y1 = (y/torch.sum(y))
    y = y1.numpy()
    y = y.flatten()
    print(torch.max(y1))
    print(torch.mean(y1))
   
    #quantiles 
    q1 = np.quantile(y,0.25)
    q2 = np.quantile(y, 0.5)
    q4 = np.quantile (y, 0.95)
    q5 = np.quantile (y, 1)
    q3 =  np.quantile (y, 0.75)
    
    #probs from q1 and between q4 and q5
    probs1 = y[(y>=q4)&(y<=q5)]
    probs2 = y[(y<=q1)]
    
    
    #values between q1 and q2
    q_values = (y >= q1) & (y <= q3)
    q_values = q_values.nonzero()[0]
    q_values = q_values [: int(q_values.size*0.8)]
    print(q_values.size)
    
    new_values = np.random.choice(probs1, size= int(q_values.size))
    print(new_values.size)
    
    #new_pT matrix
    new_pT = np.copy(y)
    
    dic = {}
    for i in range(len(q_values)):
        dic[q_values[i]]= new_values[i]
    
    #change quantile 1 values in the new pT matrix 
    for k, v in dic.items(): new_pT[k] = v
    
    
    #normalize new pT matrix 
    new_pT= new_pT/new_pT.sum()
    
    print(new_pT.sum())
    print(new_pT.size)
    print(np.mean(new_pT))
    print(np.max(new_pT))
    
    
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(new_pT, stat='probability', ax=ax, bins=100000)
    plt.title('Histogram new_pT')
    plt.xlim(0,0.000002)
    
     
def perturb_mean():
    #logT matrix
    lib = Library (use_hist= True)
    y = lib.logT
    y = torch.exp(y)
    y1 = (y/torch.sum(y))
    y = y1.numpy()
    y = y.flatten()
    
    maxi = torch.max(y1).item()
    mean = torch.mean(y1).item()
    mini = torch.min(y1).item()
    
    less_mean = y < mean 
    less_mean = less_mean.nonzero()[0]
    
    values = np.random.uniform(low=mean, high=maxi, size=(less_mean.size,))
    new_values = np.random.choice(values, size= less_mean.size)
    print()
    
    new_pT = np.copy(y)
    
    dic = {}
    for i in range(len(less_mean)):
        dic[less_mean[i]]= new_values[i]
    
    #
    for k, v in dic.items(): new_pT[k] = v 
    
    print(new_pT.sum())
    print(new_pT)
    
    #normalize new pT matrix 
    new_pT= new_pT/new_pT.sum()
    
    print(new_pT.sum())
    print(new_pT)
    
    
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(new_pT, stat='probability', ax=ax, bins=10000)
    plt.title('Histogram new_pT')
    plt.ylim(0,0.002)
    
    
    more_mean = y > mean 
    more_mean = more_mean.nonzero()[0]
    
    values = np.random.uniform(low=mini, high=mean, size=(more_mean.size,))
    new_values = np.random.choice(values, size= more_mean.size)
    print()
    
    new_pT2 = np.copy(y)
    
    dic = {}
    for i in range(len(more_mean)):
        dic[more_mean[i]]= new_values[i]
    
    #
    for k, v in dic.items(): new_pT2[k] = v 
    
    print(new_pT2.sum())
    print(new_pT2)
    
    #normalize new pT matrix 
    new_pT2= new_pT2/new_pT2.sum()
    
    print(new_pT2.sum())
    print(new_pT2)
    
    
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(new_pT2, stat='probability', ax=ax, bins=100)
    plt.title('Histogram new_pT')
    #plt.ylim(0,0.002)
   

def perturb_zero():
    lib = Library (use_hist= True)
    y = lib.logT
    y = torch.exp(y)
    y1 = (y/torch.sum(y))
    y = y1.numpy()
    y = y.flatten()
    q1 = np.quantile(y,0.25)
    
    q1_values = y <= q1
    q1_values = q1_values.nonzero()[0]
    
    new_pT = np.copy(y)
    
    for i in q1_values:
        new_pT[i] = 0
    
    #normalize new pT matrix 
    new_pT= new_pT/new_pT.sum()
    
    print(new_pT.sum())
    print(new_pT)
    
    
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(new_pT, stat='probability', ax=ax, bins=10000)
    plt.title('Histogram new_pT')
    plt.ylim(0,0.003)
    plt.xlim(0,  0.00002)
        
    
    
    
    
    
    
    
    
    
    
    
   
    
    
    
    
    
