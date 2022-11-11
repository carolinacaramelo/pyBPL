#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 20:30:44 2022

@author: carolinacaramelo
"""

def define_alpha(a,b):
    equals = len(set(a) & set(b))  # & is intersection - elements common to both
    result = equals/len(a)
    return result 

from collections import Counter

def counts(MyList):
    a = dict(Counter(MyList))
    return a 
    

def flatten(l):
    return [item for sublist in l for item in sublist]
      
#number of strokes 
alph1 = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
alph2 = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
alph3 = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
alph4 = [3,2,2,2,2,2,2,3,2,2,2,2,2,2,3,2,2,2,3,1,2,2,2,1,1]
alph5 = [3,3,1,1,3,3,3,3,2,3,3,3,3,3,1,3,3,1,3,1,3,3,3,3,1]
alph6 = [2,4,1,3,3,4,1,2,3,2,5,3,5,3,3,3,3,3,2,3,2,3,3,3,2]
alph7 = [5,3,3,3,1,5,5,3,2,5,3,3,3,3,2,3,3,3,3,3,1,5,3,2,2]
alph8 = [2,1,2,3,2,3,1,4,6,4,4,2,6,6,4,1,6,2,2,2,6,4,2,6,4]

alph9 = [1,4,1,1,4,3,1,1,4,1,1,4,4,4,4,1,3,1,4,4,1,4,4,1,1]
alph10 = [1,1,1,2,1,1,1,1,2,2,3,3,1,1,4,1,3,2,4,2,4,2,2,1,3]
alph11 = [2,3,3,3,2,3,2,3,3,1,3,3,2,3,3,3,2,2,2,3,3,1,3,2,2]


import numpy as np
import matplotlib.pyplot as plt
def plot_strokes():
    plt.figure(figsize=(10,10))
    N = 5
    ind = np.arange(N) 
    width = 0.15
      
    xvals = [0.08,0.48,0.44,0.08,0.08]
    bar1 = plt.bar(ind-width, xvals, width)
      
    yvals = [0.24,0,0.28,0.36,0.16]
    bar2 = plt.bar(ind, yvals, width)
      
    zvals = [0.52,0.08,0.16,0.56,0.56]
    bar3 = plt.bar(ind+width, zvals, width)
    
    uvals = [0.08,0.44,0.12,0,0]
    bar4 = plt.bar(ind+width*2, uvals, width)
    
    wvals = [0.08,0,0,0,0.2]
    bar5 = plt.bar(ind+width*3, wvals, width)
    
    tvals = [0,0,0,0,0]
    bar6 = plt.bar(ind+width*4, tvals, width)
      
    plt.xlabel("Concentration parameter", size=12)
    plt.ylabel('Probability', size=12)
    plt.title("Sampled number of strokes", size=15)
      
    plt.xticks(ind+width,[ '3', '3.5', '4', '4.5', '5'])
    plt.legend( (bar1, bar2, bar3,bar4,bar5,bar6), ('ns=1', 'ns=2', 'ns=3', 'ns=4', 'ns=5', 'ns=6') )
    plt.show()
    
def plot_same_el():
    N = 5
    ind = np.arange(N) 
    width = 0.35
    
    #plt.xticks(ind+width,['3', '3.5', '3', '4.5', '5'])
    y = [25,60,50,25.7,54.6]
    plt.figure (figsize=(10,10))
    plt.bar(ind, y, width)
    plt.title('Percentage of repeated sampled primitives', size=15)
    plt.xticks(ind,['3', '3.5', '4', '4.5', '5'])
    plt.ylabel("Percentage", size=12)
    plt.xlabel("Concentration parameter", size=12)
    plt.show()
    
def plot_same_el2():
  
   
    y = [25,60,50,25.7,54.6]
    x= [3,3.5,4,4.5,5]
    plt.figure (figsize=(10,10))
    plt.plot(x,y)
    plt.title('Percentage of repeated sampled primitives', size=15)
   
    plt.ylabel("Percentage", size=12)
    plt.xlabel("Concentration parameter", size=12)
    plt.show()
    

