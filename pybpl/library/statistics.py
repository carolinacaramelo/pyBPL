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

#flatten a list
def flatten(l):
    return [item for sublist in l for item in sublist]


def scatter_start():
    #try to visualize the log start matrix
    #transition one x=1 corresponds to prim1 being sampled first, transition two x=2 corresponds to prim2 being sampled first, etc. 
    lib = Library (use_hist= True)
    x = np.linspace(1, 1212, 1212)
    
    
    #get the original log start matrix from the library 
    #log start has the probabilities of each primitive being sampled first 
    y= lib.logStart
    y1 = torch.exp(y)
    
    #getting the mx, min, mean and median from log start matrix 
    maxi = torch.max(y1)
    mini= torch.min(y1)
    mean = torch.mean(y1)
    median = torch.median(y1)
    y = y1.numpy()

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
    ax.set_title('Probability distribution of start primitives')
    plt.show()
    
    #scatter plot of the log start matrix, probabilities of each primitive being sampled first
    #by doing this we get the perspective of when generating characters what are the primitives that are most probable to 
    #be sampled first
    plt.scatter (x,y, s =1)
    plt.plot(x,y)
    
    plt.xlabel('Transitions')
    plt.ylabel('Probability')
    plt.show()
    
    #saving the statistics of log start matrix
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
    #try to visualize the log T (pT) matrix, try to have a better understanding of which transitions happen the most
    #transition one x=1 corresponds to prim1-prim1, transition two x=2 corresponds to prim1-prim2, etc. 
    #there are 1212*1212 transitions, corresponding to every transition between the 1212 primitives
    lib = Library (use_hist= True)
    x = np.linspace(1, 1212*1212, 1212*1212)
    
    #getting the logT matrix and transforming it to pT matrix, the same way the authors do it in the library
    y= lib.logT
    y = torch.exp(y)
    y1 =(y/torch.sum(y))
    
    #getting the pT max, min, mean, and median 
    maxi = torch.max(y1)
    mini= torch.min(y1)
    mean = torch.mean(y1)
    median = torch.median(y1)
    
    y = y1.numpy()
    y = y.flatten()
    
    #getting the quantiles of the pT matrix
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

    #getting the scatter plot of the pT matrix 
    #having a better view of which transitions are the msot probable to happen when generating new characters 
    f, ax = plt.subplots(figsize=(20,20))
    plt.scatter (x,y, s =1)
    plt.plot(x,y)
    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)
    plt.xlabel('Transitions')
    plt.ylabel('Probability')
    plt.title('Probability distribution of transitions between primitives')
    plt.show()
    
    #plotting the same scatter plot but with quantiles 
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
    plt.title('Probability distribution of transitions between primitives')
    plt.show()
    
    #saving the statistics of the pT matrix 
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
   
#the previous functions allow us to have a better understanding of what are the primitives that are most sampled, as well as 
#what transitions happen the most when generating a new character, however, now we want to understand where the mass of these matrices
#is concentrated, what probabilities are most representated, understand the statistics of the magnitudes (probabilities) that are
#represented in log start and pT. Understanding logstart and pT distributions we can perturb these matrices and consequently perturb 
#the way in which alphabet characters are generated. 
    
    
def hist_start():
    #hist_start will allow us to get the histogram of log start matrix 
    
    #getting the log start matrix from the library 
    lib = Library (use_hist= True)
    y = lib.logStart
    y1 = torch.exp(y)
    y = y1.numpy()
    y = y.flatten()
     
    #histograms with different number of bins 
    bins =[20,500,1212]
    
    for i in bins:
        # Creating histogram
        fig, axs = plt.subplots(1, 1, figsize =(10, 10), tight_layout = True)
        axs.hist(y, bins= i, density= True)
        plt.xlabel("Magnitude")
        plt.ylabel("Counts")
        plt.title('Histogram of logStart matrix magnitudes')
        # Show plot
        plt.show()
 

def hist_pT():
    #hist_start will allow us to get the histogram of the pT matrix 
    
    #getting the pT matrix from the library 
    lib = Library (use_hist= True)
    y = lib.logT
    y = torch.exp(y)
    y1 = (y/torch.sum(y))
    y = y1.numpy()
    y = y.flatten()
    bins = int(np.sqrt(y.size))

    #once we understand the the magnitudes that happen/appear the most in the matrix are the smallest ones
    #we do a "zoom in" in the histogram in order to have a better visualization of the probability distribution
    #zoom in with different x limits
    xlim = [0.00002, 0.000002, 0.000001]
    for i in range(len(xlim)):
        # all values' counts
        fig, ax = plt.subplots(figsize=(10,10))
        plt.hist(y, bins= bins)
        plt.xlabel("Magnitude")
        plt.ylabel("Counts")
        plt.title('Histogram of pT matrix magnitudes')
        plt.xlim(0, xlim[i])
        
        # Show plot
        plt.show()
    
    #histogram for different bins, already with zoom in of xlim=0.000001
    bins = [10000, 100000, 500000]
    for i in range(len(bins)):
        #all values' counts
        fig, ax = plt.subplots(figsize=(10,10))
        plt.hist(y, bins= bins[i])
        plt.xlabel("Magnitude")
        plt.ylabel("Counts")
        plt.title('Histogram of pT matrix magnitudes')
        plt.xlim(0, 0.000001)
        # Show plot
        plt.show()
        

def hist_norm_pT():
    #histogram of the pT matrix, but normalized (instead of getting counts we get probabilities)
    
    #getting the pT matrix from the library
    lib = Library (use_hist= True)
    y = lib.logT
    y = torch.exp(y)
    y1 = (y/torch.sum(y))
    y = y1.numpy()
    y = y.flatten()
    
    #normalized histograms for different bins and xlim=0.000001
    bins = [10000, 100000, 500000]
    for i in range(len(bins)):
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.histplot(y, stat='probability', ax=ax, bins=bins[i])
        plt.xlim(0,0.000001)
        plt.title('Histogram of pT matrix magnitudes')
        
        #getting the quantiles of the pT matrix
        quant_5, quant_25, quant_50, quant_75, quant_95 = np.quantile(y,0.05), np.quantile(y,0.25), np.quantile(y,0.5), np.quantile(y,0.75), np.quantile(y,0.95)
        quants = [[quant_5,  0.46], [quant_25,  0.56], [quant_50,  0.56],  [quant_75,  0.76], [quant_95, 0.86]]
        for i in quants:
            ax.axvline(i[0], ymax = i[1], linestyle = ":", color="red")

    print(quant_5, quant_25, quant_50, quant_75, quant_95) 
    
    
def hist_norm_start():
    #histogram of the logstart matrix, but normalized (instead of getting counts we get probabilities)
    
    #getting the log start matrix from the library
    lib = Library (use_hist= True)
    y = lib.logStart
    y1 = torch.exp(y)
    y = y1.numpy()
    y = y.flatten()
    
    #normalized histograms for different bins 
    bins = [10000, 100000, 500000]
    for i in range(len(bins)):
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.histplot(y, stat='probability', ax=ax, bins=bins[i])
        plt.title('Histogram of logStart matrix magnitudes')
        
        #getting the quantiles of the logStart matrix
        quant_5, quant_25, quant_50, quant_75, quant_95 = np.quantile(y,0.05), np.quantile(y,0.25), np.quantile(y,0.5), np.quantile(y,0.75), np.quantile(y,0.95)
        quants = [[quant_5,  0.46], [quant_25,  0.56], [quant_50,  0.56],  [quant_75,  0.76], [quant_95, 0.86]]
        for i in quants:
            ax.axvline(i[0], ymax = i[1], linestyle = ":", color="red")

    print(quant_5, quant_25, quant_50, quant_75, quant_95) 




#after getting a better view of the matrices' distributions we can start perturbing these distributions 
def perturb_quartile1():
    #perturbing the first quartile of the pT matrix
    #getting pT matrix
    lib = Library (use_hist= True)
    y = lib.logT
    y = torch.exp(y)
    y1 = (y/torch.sum(y))
    y = y1.numpy()
    y = y.flatten()
    
    #getting first quartile values
    q1 = np.quantile(y,0.25)
    q1_values = y <= q1
    q1_values = q1_values.nonzero()[0]
    
    #probability sum of the first quartile values
    prob_count = 0
    for  i in q1_values:
        prob_count += y[i]
    print("prob count", prob_count)
    
    #number of samples to draw is equal to the size of q1_values
    size = q1_values.size 
    print(size)
    
    #sample new values randomly from a normal distribution with mean 0.5 and standar deviation 1 (this was a random choice..any other idea?)
    #the number of values that is sampled is equal to the size of q1_values, because the new values will replace the q1_values in the pT matrix 
    new_values = np.random.normal(loc=0.5, scale =1, size=size)
    #normalization of the samples values - is this needed?
    new_values= np.exp(new_values)
    new_values = new_values/new_values.sum()
    print ("new_values matrix" , new_values)
    print ("new_values size" , new_values.size)
    
    #plot and visualize the distribution of the new values that were sampled
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(new_values, stat='probability', ax=ax, bins=20000)
    plt.title('Histogram new_values')
   
    #start creating the new perturbed pT matrix
    new_pT = np.copy(y)
    
    #create dictionary for array replacement 
    dic = {}
    for i in range(len(q1_values)):
        dic[q1_values[i]]= new_values[i]
    
    #change quartile 1 values for the new values in the new perturbed pT matrix 
    for k, v in dic.items(): new_pT[k] = v
    
    #normalize new pT matrix 
    new_pT= new_pT/new_pT.sum()
    
    #plot histogram of the new perturbed pT matrix, compare it with the original pT matrix histogram
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(new_pT, stat='probability', ax=ax, bins=10000)
    plt.title('Histogram new perturbed pT matrix')
    
    #zoom in
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(new_pT, stat='probability', ax=ax, bins=100000)
    plt.xlim(0,0.000001)
    plt.title('Histogram new perturbed pT matrix')
    
    print ("new_pT matrix",new_pT)
    print ("new_pT matrix sum",new_pT.sum())
    print ("new_pT matrix size",new_pT.size)
    
    #repeat the process for logstart
    print("STARTING FOR LOGSTART")
    
    #perturbing the first quartile of the log start matrix
    lib = Library (use_hist= True)
    y = lib.logstart
    y1 = torch.exp(y)
    y = y1.numpy()
    y = y.flatten()
    
    #getting first quartile values
    q1 = np.quantile(y,0.25)
    q1_values = y <= q1
    q1_values = q1_values.nonzero()[0]
    
    #probability sum of the first quartile values
    prob_count = 0
    for  i in q1_values:
        prob_count += y[i]
    print("prob count", prob_count)
    
    #number of samples to draw is equal to the size of q1_values
    size = q1_values.size 
    print(size)
    
    #sample new values randomly from a normal distribution with mean 0.5 and standar deviation 1 (this was a random choice..any other idea?)
    #the number of values that is sampled is equal to the size of q1_values, because the new values will replace the q1_values in the pT matrix 
    new_values = np.random.normal(loc=0.5, scale =1, size=size)
    #normalization of the samples values - is this needed?
    new_values= np.exp(new_values)
    new_values = new_values/new_values.sum()
    print ("new_values matrix" , new_values)
    print ("new_values size" , new_values.size)
    
    #plot and visualize the distribution of the new values that were sampled
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(new_values, stat='probability', ax=ax, bins=20000)
    plt.title('Histogram new_values')
   
    #start creating the new perturbed logstart matrix
    new_start = np.copy(y)
    
    #create dictionary for array replacement 
    dic = {}
    for i in range(len(q1_values)):
        dic[q1_values[i]]= new_values[i]
    
    #change quartile 1 values for the new values in the new perturbed logstart matrix 
    for k, v in dic.items(): new_start[k] = v
    
    #normalize new logstart matrix 
    new_start= new_start/new_start.sum()
    
    #plot histogram of the new perturbed logstart matrix, compare it with the original logstart matrix histogram
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(new_pT, stat='probability', ax=ax, bins=10000)
    plt.title('Histogram new perturbed pT matrix')
    
    #zoom in
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(new_pT, stat='probability', ax=ax, bins=100000)
    plt.title('Histogram new perturbed logStart matrix')
    
    print ("new_start matrix",new_pT)
    print ("new_start matrix sum",new_pT.sum())
    print ("new_start matrix size",new_pT.size)


def perturb_quartile2():
    #perturbing the second quartile of the pT matrix 
    
    #getting logT from the library
    lib = Library (use_hist= True)
    y = lib.logT
    y = torch.exp(y)
    y1 = (y/torch.sum(y))
    y = y1.numpy()
    y = y.flatten()
    
    #getting the second quartile values
    q2 = np.quantile(y, 0.5)
    q2_values = y <= q2
    q2_values = q2_values.nonzero()[0]
    
    #probability sum of the values in q2
    prob_count = 0
    for  i in q2_values:
        prob_count += y[i]
    print("prob count", prob_count)
    
    #number of values in the second quartile, values that will be replaced in the pT matrix 
    #number of samples to draw from the normal distribution to get the new_values
    size = q2_values.size  
    print(size)
    
    #sample new values randomly from a normal distribution with mean 0.5 and standar deviation 1 (this was a random choice... any other idea?)
    #the number of values that is sampled is equal to the size of q2_values, because the new values will replace the q1_values in the pT matrix 
    new_values = np.random.normal(loc=0.5, scale =1, size=size)
    new_values= np.exp(new_values)
    new_values = new_values/new_values.sum()
    print ("new_values matrix" , new_values)
    print ("new_values size" , new_values.size)
    
    #plot and visualize the distribution of the new values that were sampled
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(new_values, stat='probability', ax=ax, bins=20000)
    plt.title('Histogram new_values')
    
    
    #normalize new pT matrix 
    new_pT = np.copy(y)
    
    #create dictionary for array replacement 
    dic = {}
    for i in range(len(q2_values)):
        dic[q2_values[i]]= new_values[i]
    
    #change quantile 2 values in the new pT matrix 
    for k, v in dic.items(): new_pT[k] = v
    
    #normalize new pT matrix 
    new_pT= new_pT/new_pT.sum()
    
    #plot histogram of the new perturbed pT matrix, compare it with the original pT matrix histogram
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(new_pT, stat='probability', ax=ax, bins=10000)
    plt.title('Histogram perturbed new_pT')
    
    #zoom in
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(new_pT, stat='probability', ax=ax, bins=100000)
    plt.xlim(0,0.000001)
    
    plt.title('Histogram perturbed new_pT')
    
    print (" new_pT matrix" , new_pT)
    print (" new_pT matrix sum" , new_pT.sum())
    print (" new_pT matrix size" , new_pT.size)
    
    #repeat the process for the logStart matrix 
    print("STARTING FOR LOGSTART")
    

    #getting logStart from the library
    lib = Library (use_hist= True)
    y = lib.logStart
    y1 = torch.exp(y)
    y = y1.numpy()
    y = y.flatten()
    
    #getting the second quartile values
    q2 = np.quantile(y, 0.5)
    q2_values = y <= q2
    q2_values = q2_values.nonzero()[0]
    
    #probability sum of the values in q2
    prob_count = 0
    for  i in q2_values:
        prob_count += y[i]
    print("prob count", prob_count)
    
    #number of values in the second quartile, values that will be replaced in the pT matrix 
    #number of samples to draw from the normal distribution to get the new_values
    size = q2_values.size  
    print(size)
    
    #sample new values randomly from a normal distribution with mean 0.5 and standar deviation 1 (this was a random choice... any other idea?)
    #the number of values that is sampled is equal to the size of q2_values, because the new values will replace the q1_values in the logStart matrix 
    new_values = np.random.normal(loc=0.5, scale =1, size=size)
    new_values= np.exp(new_values)
    new_values = new_values/new_values.sum()
    print ("new_values matrix" , new_values)
    print ("new_values size" , new_values.size)
    
    #plot and visualize the distribution of the new values that were sampled
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(new_values, stat='probability', ax=ax, bins=20000)
    plt.title('Histogram new_values')
    
    
    #normalize new start matrix 
    new_start = np.copy(y)
    
    #create dictionary for array replacement 
    dic = {}
    for i in range(len(q2_values)):
        dic[q2_values[i]]= new_values[i]
    
    #change quantile 2 values in the new pT matrix 
    for k, v in dic.items(): new_start[k] = v
    
    #normalize new start matrix 
    new_start= new_start/new_start.sum()
    
    #plot histogram of the new perturbed start matrix, compare it with the original start matrix histogram
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(new_start, stat='probability', ax=ax, bins=10000)
    plt.title('Histogram perturbed new_start')
    
    #zoom in
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(new_pT, stat='probability', ax=ax, bins=100000)
    
    
    plt.title('Histogram perturbed new_start')
    
    print (" new_start matrix" , new_pT)
    print (" new_start matrix sum" , new_pT.sum())
    print (" new_start matrix size" , new_pT.size)

def perturb_flattening():
    #perturb_flattening will reeplace the first quartile values with the magnitues that are represented in the 4th quartile
    #the goal is to grow the number of entries in the pT matrix with higher probabilities (which are the probs in the 4th quartile)
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
    q3 =  np.quantile (y, 0.75)
    q4 = np.quantile (y, 0.95)
    q5 = np.quantile (y, 1)
    
    #probs from q1 and between q4 and q5
    probs1 = y[(y>=q4)&(y<=q5)]
    probs2 = y[(y<=q1)]
    print(probs1)
    print(probs2)
    
    
    #0.8 of the values between q1 and q3, number of values we want to change
    q_values = (y >= q1) & (y <= q3)
    q_values = q_values.nonzero()[0]
    q_values = q_values [:int(q_values.size*0.8)]
    print(q_values.size)
    
    new_values = np.random.choice(probs1, size= int(q_values.size))
    print(new_values.size)
    
    #plot and visualize the distribution of the new values that were sampled
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(new_values, stat='probability', ax=ax, bins=20000)
    plt.title('Histogram new_values')
    
    
    #new_pT matrix
    new_pT = np.copy(y)
    
    dic = {}
    for i in range(len(q_values)):
        dic[q_values[i]]= new_values[i]
    
    #change 80% of the values between quantile 1 and 3 in the new pT matrix 
    for k, v in dic.items(): new_pT[k] = v
    
    
    #normalize new pT matrix 
    new_pT= new_pT/new_pT.sum()
    
    print(new_pT.sum())
    print(new_pT.size)
    print(np.mean(new_pT))
    print(np.max(new_pT))
    
    #plot histogram of new pT_matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(new_pT, stat='probability', ax=ax, bins=100000)
    plt.title('Histogram new_pT')
   
    print("STARTING FOR LOGSTART")
    
    #logstart matrix
    lib = Library (use_hist= True)
    y = lib.logStart
    y1 = torch.exp(y)
    y = y1.numpy()
    y = y.flatten()
    print(torch.max(y1))
    print(torch.mean(y1))
   
    #quantiles 
    q1 = np.quantile(y,0.25)
    q3 =  np.quantile (y, 0.75)
    q4 = np.quantile (y, 0.95)
    q5 = np.quantile (y, 1)
    
    #probs from q1 and between q4 and q5
    probs1 = y[(y>=q4)&(y<=q5)]
    probs2 = y[(y<=q1)]
    print(probs1)
    print(probs2)
    
    
    #0.8 of the values between q1 and q3, number of values we want to change
    q_values = (y >= q1) & (y <= q3)
    q_values = q_values.nonzero()[0]
    q_values = q_values [:int(q_values.size*0.8)]
    print(q_values.size)
    
    new_values = np.random.choice(probs1, size= int(q_values.size))
    print(new_values.size)
    
    #plot and visualize the distribution of the new values that were sampled
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(new_values, stat='probability', ax=ax, bins=20000)
    plt.title('Histogram new_values')
    
    
    #new_start matrix
    new_start = np.copy(y)
    
    dic = {}
    for i in range(len(q_values)):
        dic[q_values[i]]= new_values[i]
    
    #change 80% of the values between quantile 1 and 3 in the new start matrix 
    for k, v in dic.items(): new_start[k] = v
    
    
    #normalize new start matrix 
    new_start= new_start/new_start.sum()
    
    print(new_start.sum())
    print(new_start.size)
    print(np.mean(new_start))
    print(np.max(new_start))
    
    #plot histogram of new start_matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(new_start, stat='probability', ax=ax, bins=100000)
    plt.title('Histogram new_start')
    
     
def perturb_less_mean():
    #perturb_less_mean gets the entries that have values that are lower than the mean and replaces these values
    #by values that are randomly sampled froma uniform distribution between the mean and the max of pT
    
    #logT matrix
    lib = Library (use_hist= True)
    y = lib.logT
    y = torch.exp(y)
    y1 = (y/torch.sum(y))
    y = y1.numpy()
    y = y.flatten()
    
    #getting the max and min of pT
    maxi = torch.max(y1).item()
    mean = torch.mean(y1).item()
    print(mean)
    
    #entries of pT matrix that have values lower than the mean
    less_mean = y < mean 
    less_mean = less_mean.nonzero()[0]
    print(less_mean.size)
    
    #new values sampled from a uniform distribution 
    values = np.random.uniform(low=mean, high=maxi, size=(less_mean.size,))
    new_values = np.random.choice(values, size= less_mean.size)
    print(new_values)
    
    #new values plot
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(new_values, stat='probability', ax=ax, bins=20000)
    plt.title('Histogram new_values')
    
    #perturbing pT matrix and getting a new pT matrix
    new_pT = np.copy(y)
    
    dic = {}
    for i in range(len(less_mean)):
        dic[less_mean[i]]= new_values[i]
    
    #replace values
    for k, v in dic.items(): new_pT[k] = v 
    
    print(new_pT.sum())
    print(new_pT)
    
    #normalize new pT matrix 
    new_pT= new_pT/new_pT.sum()
    
    print(new_pT.sum())
    print(new_pT)
    
    #plotting new_pT 
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(new_pT, stat='probability', ax=ax, bins=10000)
    plt.title('Histogram new_pT')
    
    print("STARTING FOR LOGSTART")
    
    #logStart matrix
    lib = Library (use_hist= True)
    y = lib.logStart
    y = torch.exp(y)
    y = y1.numpy()
    y = y.flatten()
    
    #getting the max and min of Start
    maxi = torch.max(y1).item()
    mean = torch.mean(y1).item()
    print(mean)
    
    #entries of start matrix that have values lower than the mean
    less_mean = y < mean 
    less_mean = less_mean.nonzero()[0]
    print(less_mean.size)
    
    #new values sampled from a uniform distribution 
    values = np.random.uniform(low=mean, high=maxi, size=(less_mean.size,))
    new_values = np.random.choice(values, size= less_mean.size)
    print(new_values)
    
    #new values plot
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(new_values, stat='probability', ax=ax, bins=20000)
    plt.title('Histogram new_values')
    
    #perturbing start matrix and getting a new pT matrix
    new_start = np.copy(y)
    
    dic = {}
    for i in range(len(less_mean)):
        dic[less_mean[i]]= new_values[i]
    
    #replace values
    for k, v in dic.items(): new_start[k] = v 
    
    print(new_start.sum())
    print(new_start)
    
    #normalize new start matrix 
    new_start= new_start/new_start.sum()
    
    print(new_start.sum())
    print(new_start)
    
    #plotting new_start
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(new_start, stat='probability', ax=ax, bins=10000)
    plt.title('Histogram new_start')
    
   
    
def perturb_more_mean():  
    #perturb_more_mean gets the entries of the pT matrix that have higher values that the mean, and replaces these values
    #by values sampled from a uniform distribution with min value the minimum of pT and maximum value the mean of pT
    #higher probability entries will disapear 
    
    #logT matrix
    lib = Library (use_hist= True)
    y = lib.logT
    y = torch.exp(y)
    y1 = (y/torch.sum(y))
    y = y1.numpy()
    y = y.flatten()
    
    #mean and minimum of pT
    mean = torch.mean(y1).item()
    mini = torch.min(y1).item()
    
    #entries of pT with values higher than the mean
    more_mean = y > mean 
    more_mean = more_mean.nonzero()[0]
    print(more_mean.size)
    
    #new values sampled from uniform distribution
    values = np.random.uniform(low=mini, high=mean, size=(more_mean.size,))
    new_values = np.random.choice(values, size= more_mean.size)
    print(new_values)
    
    #plotting new values
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(new_values, stat='probability', ax=ax, bins=20000)
    plt.title('Histogram new_values')
    
    #perturbing pT
    new_pT = np.copy(y)
    
    dic = {}
    for i in range(len(more_mean)):
        dic[more_mean[i]]= new_values[i]
    
    #replacing values in new pT
    for k, v in dic.items(): new_pT[k] = v 
    
    print(new_pT.sum())
    print(new_pT)
    
    #normalize new pT matrix 
    new_pT= new_pT/new_pT.sum()
    
    print(new_pT.sum())
    print(new_pT)
    
    #plotting new pT
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(new_pT, stat='probability', ax=ax, bins=100)
    plt.title('Histogram new_pT')
    #plt.ylim(0,0.002)
    
    print("STARTING FOR LOGSTART")
    #logstart matrix
    lib = Library (use_hist= True)
    y = lib.logT
    y = torch.exp(y)
    y = y1.numpy()
    y = y.flatten()
    
    #mean and minimum of start
    mean = torch.mean(y1).item()
    mini = torch.min(y1).item()
    
    #entries of logstart with values higher than the mean
    more_mean = y > mean 
    more_mean = more_mean.nonzero()[0]
    print(more_mean.size)
    
    #new values sampled from uniform distribution
    values = np.random.uniform(low=mini, high=mean, size=(more_mean.size,))
    new_values = np.random.choice(values, size= more_mean.size)
    print(new_values)
    
    #plotting new values
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(new_values, stat='probability', ax=ax, bins=20000)
    plt.title('Histogram new_values')
    
    #perturbing logstart
    new_start = np.copy(y)
    
    dic = {}
    for i in range(len(more_mean)):
        dic[more_mean[i]]= new_values[i]
    
    #replacing values in new start
    for k, v in dic.items(): new_start[k] = v 
    
    print(new_start.sum())
    print(new_start)
    
    #normalize new start matrix 
    new_start= new_start/new_start.sum()
    
    print(new_start.sum())
    print(new_start)
    
    #plotting new start
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(new_start, stat='probability', ax=ax, bins=100)
    plt.title('Histogram new_start')
   

def perturb_zero():
    #perturb_zero will get the lowest probabilities (magnitudes) of the pT matrix (the ones in q2) and get them to zero
    #then we normalize the pT matrix, increasing the magnitude values in the non zero entries
    
    lib = Library (use_hist= True)
    y = lib.logT
    y = torch.exp(y)
    y1 = (y/torch.sum(y))
    y = y1.numpy()
    y = y.flatten()
    q2 = np.quantile(y,0.5)
    q2_values = y <= q2
    q2_values = q2_values.nonzero()[0]
    
    new_pT = np.copy(y)
    
    for i in q2_values:
        new_pT[i] = 0
    
    #normalize new pT matrix 
    new_pT= new_pT/new_pT.sum()
    
    print(new_pT.sum())
    print(new_pT)
    
    #plotting new pt
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(new_pT, stat='probability', ax=ax, bins=10000)
    plt.title('Histogram new_pT')
    plt.ylim(0,0.003)
    plt.xlim(0,  0.00002)
    
    print("STARTING FOR LOGSTART")
    
    lib = Library (use_hist= True)
    y = lib.logT
    y1 = torch.exp(y)
    y = y1.numpy()
    y = y.flatten()
    q2 = np.quantile(y,0.5)
    q2_values = y <= q2
    q2_values = q2_values.nonzero()[0]
    
    new_start = np.copy(y)
    
    for i in q2_values:
        new_start[i] = 0
    
    #normalize new start matrix 
    new_start= new_start/new_start.sum()
    
    print(new_start.sum())
    print(new_start)
    
    #plotting new start
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(new_start, stat='probability', ax=ax, bins=10000)
    plt.title('Histogram new_start')
   
    
        
    
    
    
    
    
    
    
    
    
    
    
   
    
    
    
    
    
