#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 11:30:11 2022

@author: carolinacaramelo
"""

from pybpl.library import Library 
import matplotlib.pyplot as plt 
import numpy as np 
import torch 

def flatten(l):
    return [item for sublist in l for item in sublist]

#run this for different values of alpha - time constant 
def dif_perturbations(alpha):
    
    #getting the pT matrix 
    lib = Library(use_hist= True)
    logR = lib.logT
    R = torch.exp(logR)
    pT = R / torch.sum(R)
    np.savetxt("./pT_original", pT)
    #original pT matrix graph 
    x = np.linspace(0, 1212*1212, 1212*1212)
    plt.figure (figsize=(10,10))
    plt.scatter(x, pT)
    plt.title("Transition probabilities between primitives")
    plt.ylabel("Probabilities")
    plt.xlabel("Primitive pairs")
    plt.show()
    
    
    #getting logstart
    logstart = torch.exp(lib.logStart)
    np.savetxt("./logstart_original", logstart)
    #original logstart matrix graph 
    x = np.linspace(0, 1212, 1212)
    plt.figure (figsize=(10,10))
    plt.scatter(x, logstart)
    plt.title("Primitive's probabilities of starting a stroke")
    plt.ylabel("Probabilities")
    plt.xlabel("Primitives")
    plt.show()

    #defining a distance function according to the diffusion process 
    d = -1 * torch.log(pT)
    d_start = torch.log(logstart)
    
    
    d = alpha * d
    d_start = alpha * d_start
    np.savetxt("./d", d)
    np.savetxt("./d_start", d_start)
    print(d_start)
    
    #graph of diffusion distance
    x = np.linspace(0, 1212*1212, 1212*1212)
    plt.figure (figsize=(10,10))
    plt.scatter(x, d)
    plt.title("Distance function in primitive space")
    plt.ylabel("Distance")
    plt.xlabel("Primitive pairs")
    plt.show()
    
    x = np.linspace(0, 1212, 1212)
    plt.figure (figsize=(10,10))
    plt.scatter(x, d_start)
    plt.title("Distance function in primitive space")
    plt.ylabel("Distance")
    plt.xlabel("Primitives")
    plt.show()
  
    
    exp_d = torch.exp(d)
    ro_pT = exp_d / torch.sum(exp_d)
    #ro_pT = ro_pT.numpy().flatten()
    np.savetxt("./ro_pT", ro_pT)
    
    #graph of ro_pT
    x = np.linspace(0, 1212*1212, 1212*1212)
    plt.figure (figsize=(10,10))
    plt.scatter(x, ro_pT)
    plt.title("Ro pT matrix")
    #plt.ylabel("Distance")
    #plt.xlabel("Primitive pairs")
    plt.show()
    
    #line graph of ro_pT 
    #plt.figure (figsize=(10,10))
    #plt.plot(x, ro_pT)
    #plt.title("Ro pT matrix")
    #plt.show()
    
    
    exp_dstart = torch.exp(d_start)
    np.savetxt("./exp_dstart", exp_dstart)
    
    
    ro_start = exp_dstart/ torch.sum(exp_dstart)
    print(ro_start.sum())
    np.savetxt("./ro_start", ro_start)
    
    #graph of ro_start
    x = np.linspace(0, 1212, 1212)
    plt.figure (figsize=(10,10))
    plt.scatter(x, ro_start)
    plt.title("Ro logStart matrix")
    #plt.ylabel("Distance")
    #plt.xlabel("Primitives")
    plt.show()

        
    return ro_pT, ro_start

def dif_perturbations_alpha_viz():
    #getting the pT matrix 
    lib = Library(use_hist= True)
    logR = lib.logT
    R = torch.exp(logR)
    pT = R / torch.sum(R)
    #original pT matrix graph 
    x = np.linspace(0, 1212*1212, 1212*1212)
    plt.figure (figsize=(10,10))
    plt.scatter(x, pT)
    plt.title("Transition probabilities between primitives")
    plt.ylabel("Probabilities")
    plt.xlabel("Primitive pairs")
    plt.show()
    
    alpha = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    plt.figure (figsize=(10,10))
    plt.title("Ro pT matrix")
    for alpha in alpha:
        #defining a distance function according to the diffusion process 
        d = -alpha * torch.log(pT)
        #ro_pT matrix 
        exp_d = torch.exp(d)
        ro_pT = exp_d / torch.sum(exp_d)
        #graph of ro_pT
        x = np.linspace(0, 1212*1212, 1212*1212)
        plt.scatter(x, ro_pT, label = 'Alpha=%s' %alpha)
        #plt.yticks(np.arange(0,5e-6, step=0.2))
        #plt.ylabel("Distance")
        #plt.xlabel("Primitive pairs")
    plt.legend()
    plt.show()
        
        
def diffusion_perturbing(alpha, threshold, constant):
    ro_pT = dif_perturbations(alpha)[0]
    
    #graph of ro_pT with a certain alpha and a certain threshold
    x = np.linspace(0, 1212*1212, 1212*1212)
    plt.figure (figsize=(10,10))
    plt.scatter(x, ro_pT)
    plt.title("Ro pT matrix")
    plt.axhline(y=threshold, xmin=0, xmax=0.95, linestyle ="--", color= "#FF7F50", label="Threshold")
    #plt.ylabel("Distance")
    #plt.xlabel("Primitive pairs")
    plt.legend()
    plt.show()
    
    #perturbing ro_pT above threshold values
    changed_values = ro_pT >= threshold
    changed_values = changed_values.nonzero()
    
    #creating the new perturbed ro_pT matrix >> new pT matrix 
    new_pT = np.copy(ro_pT)
     
    #create dictionary for array replacement 
    dic = {}
    for i in range(changed_values.shape[0]):
        dic[changed_values[i]]= constant
    
    #change quartile 1 values for the new values in the new perturbed pT matrix 
    for k, v in dic.items(): new_pT[k[0],k[1]] = v
    
    #normalize new pT matrix 
    new_pT= new_pT/new_pT.sum()
    print(new_pT.sum())
    #graph of ro_pT with a certain alpha and a certain threshold
    plt.figure (figsize=(10,10))
    plt.scatter(x, new_pT)
    plt.title("New pT matrix")
    #plt.ylabel("Distance")
    #plt.xlabel("Primitive pairs")
    plt.show()
    
    
    return new_pT
   
        
    
    
 
        
        
        
        