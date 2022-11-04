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
import scipy.linalg
from sklearn.model_selection import GridSearchCV
from scipy.optimize import brute, fmin, minimize
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.linalg import eigh
import plotly.io as io
io.renderers.default='browser'
import seaborn as sns
import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy.special import rel_entr
import scipy.stats
from scipy.stats import ks_2samp
from torch.autograd import Variable


def flatten(l):
    return [item for sublist in l for item in sublist]

#compute pT matrix, correspondent distance matrix and ro_pT matrix in the fastest way 
#use this function to calculate entropy
def dif_perturbations(beta):
    #getting the pT matrix 
    lib = Library(use_hist= True)
    logR = lib.logT
    R = torch.exp(logR)
    pT = R / torch.sum(R)
    d = -1 * torch.log(pT)
    ro_pT = torch.exp(-beta*d)/torch.sum(torch.exp(-beta*d))
    
    return pT,ro_pT 

#compute the logstart matrix, correspondent distance matrix, and ro_start matrix in the fastest way
#use this function to calculate the entropy 
def dif_perturbations_start(beta):
    lib = Library(use_hist= True)
    logstart = torch.exp(lib.logStart)
    logstart = torch.where(logstart==0, torch.tensor(1e-20, dtype=logstart.dtype), logstart)
    d_start = -1 *torch.log(logstart)
    ro_start = torch.exp(-beta*d_start)/torch.sum(torch.exp(-beta*d_start))

    return logstart,ro_start

######################## PERTURBING - NEW MATRICES ##############################################
#for now we are not using these function - but I will leave them here 
def diffusion_perturbing(beta, threshold, constant):
    #computing ro_pT
    results = dif_perturbations(beta)
    ro_pT = results[1]
    
    #perturbing ro_pT above threshold values
    indexes_change = ro_pT >= threshold
    indexes_change = indexes_change.nonzero()
    
    #creating the new perturbed pT matrix matrix >> new pT matrix 
    new_pT = np.copy(results[0])
     
    #create dictionary for array replacement 
    dic = {}
    for i in range(indexes_change.shape[0]):
        dic[indexes_change[i]]= constant
    
    #change the new constant value in the indexes, selected from the ro_pT matrix, in the pT matrix
    #this way we obtain the new_pT matrix 
    for k, v in dic.items(): new_pT[k[0],k[1]] = v
    
    #normalize new pT matrix 
    new_pT= new_pT/new_pT.sum()
    new_pT = torch.tensor(new_pT)
    
    return new_pT

def diffusion_perturbing_start(beta, threshold, constant):
    #computing ro_start
    results = dif_perturbations_start(beta)
    ro_start = results[1]
    
    #perturbing ro_start above threshold values
    indexes_change = ro_start >= threshold
    indexes_change = indexes_change.nonzero()
    
    #creating the new perturbed logstart matrix >> new logstart matrix 
    new_start = np.copy(torch.exp(results[0]))
     
    #create dictionary for array replacement 
    dic = {}
    for i in range(indexes_change.shape[0]):
        dic[indexes_change[i]]= constant
    
    #change the new constant value in the indexes, selected from the ro_start matrix, in the logstart matrix
    #this way we obtain the new_start matrix 
    for k, v in dic.items(): new_start[k[0]] = v
    
    #normalize new pT matrix 
    new_start= new_start/new_start.sum()
    new_start = torch.tensor(new_start)
    
    return new_start
    


    
################################## DIFFUSION PERTURBATION GRAPHS ####################################
    
#run this for different values of beta - inverse time constant - get all the results for
#different alpha values
#try to have a understanding of which alpha, threshold and constant values to use
#in the grid search optimization 
def dif_perturbations_graph(beta):
    #pT MATRIX
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
    
    #defining a distance function according to the diffusion process 
    d = -1 * torch.log(pT)
    np.savetxt("./d", d)
    
    #graph of diffusion distance
    x = np.linspace(0, 1212*1212, 1212*1212)
    plt.figure (figsize=(10,10))
    plt.scatter(x, d)
    plt.title("Distance function in primitive space")
    plt.ylabel("Distance")
    plt.xlabel("Primitive pairs")
    plt.show()
    
    #computing ro_pT matrix
    exp_d = torch.exp(-beta*d)
    ro_pT = exp_d/torch.sum(exp_d)
    np.savetxt("./ro_pT", ro_pT)
    
    #graph of ro_pT
    x = np.linspace(0, 1212*1212, 1212*1212)
    plt.figure (figsize=(10,10))
    plt.scatter(x, ro_pT)
    plt.title("Ro pT matrix")
    plt.ylabel("Probability") #não tenho a certeza
    plt.xlabel("Primitive pairs")
    plt.show()
    
    #line graph of ro_pT - ro_pT is the new perturbed matrix 
    plt.figure (figsize=(10,10))
    plt.plot(x, ro_pT)
    plt.title("Ro pT matrix")
    plt.ylabel("Probability") #não tenho a certeza
    plt.xlabel("Primitive pairs")
    plt.show()
    
    #LOGSTART MATRIX
    #getting logstart
    start = torch.exp(lib.logStart) #its not logstart anymore, but it is still de original used matrix
    #replacing zero values in logstart matrix for approximate zero values
    start = torch.where(start==0, torch.tensor(1e-20, dtype=start.dtype), start)
    np.savetxt("./start_original", start)
    
    #original logstart matrix graph 
    x = np.linspace(0, 1212, 1212)
    plt.figure (figsize=(10,10))
    plt.scatter(x, start)
    plt.title("Primitive's probabilities of starting a stroke")
    plt.ylabel("Probabilities")
    plt.xlabel("Primitives")
    plt.show()

    #defining a distance function according to the diffusion process 
    d_start = -1*torch.log(start)
    np.savetxt("./d_start", d_start)
    
    #graph of diffusion distance
    x = np.linspace(0, 1212, 1212)
    plt.figure (figsize=(10,10))
    plt.scatter(x, d_start)
    plt.title("Distance function in primitive space")
    plt.ylabel("Distance")
    plt.xlabel("Primitives")
    plt.show()

    #computing ro_start matrix 
    exp_dstart = torch.exp(-beta*d_start)
    np.savetxt("./exp_dstart", exp_dstart)
    ro_start = exp_dstart/torch.sum(exp_dstart)
    np.savetxt("./ro_start", ro_start)
    
    #graph of ro_start
    x = np.linspace(0, 1212, 1212)
    plt.figure (figsize=(10,10))
    plt.scatter(x, ro_start)
    plt.title("Ro logStart matrix")
    plt.ylabel("Distance")
    plt.xlabel("Primitives")
    plt.show()
   
    return ro_pT, ro_start #new_perturbed matrices 

#visualizing the ro_pT and ro_start matrix for every alpha value in an overlapping graph
def dif_perturbations_beta_viz():
    #pT MATRIX
    #getting the pT matrix 
    lib = Library(use_hist= True)
    logR = lib.logT
    R = torch.exp(logR)
    pT = R / torch.sum(R)
    
    beta = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    plt.figure (figsize=(10,10))
    plt.title("Ro pT matrix")
    plt.ylabel("Probability") #não tenho a certeza
    plt.xlabel("Primitive pairs")
    for beta in beta:
        #defining a distance function according to the diffusion process 
        d = -1 * torch.log(pT)
        #ro_pT matrix 
        exp_d = torch.exp(-beta*d)
        ro_pT = exp_d / torch.sum(exp_d)
        #graph of ro_pT
        x = np.linspace(0, 1212*1212, 1212*1212)
        plt.scatter(x, ro_pT, label = 'Beta=%s' %beta)
        #plt.yticks(np.arange(0,5e-6, step=0.2))
    plt.legend()
    plt.show()
    
    #LOGSTART MATRIX
    #getting logstart matrix
    start = torch.exp(lib.logStart) 
    start = torch.where(start==0, torch.tensor(1e-20, dtype=start.dtype), start)
    
    beta = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    plt.figure (figsize=(10,10))
    plt.title("Ro logStart matrix")
    plt.ylabel("Probability") #não tenho a certeza
    plt.xlabel("Primitives")
    for beta in beta:
        #defining a distance function according to the diffusion process 
        d_start = -1*torch.log(start)
        #ro_start matrix 
        exp_dstart = torch.exp(-beta*d_start)
        ro_start = exp_dstart/torch.sum(exp_dstart)
        #graph of ro_start
        x = np.linspace(0, 1212*1212, 1212*1212)
        plt.scatter(x, ro_start, label = 'Beta=%s' %beta)
        #plt.yticks(np.arange(0,5e-6, step=0.2))
    plt.legend()
    plt.show()
    
        
#function to plot the new perturbed matrices, includes threshold and constant     
def diffusion_perturbing_graph(beta, threshold, constant):
    #computing ro_pT
    ro_pT = dif_perturbations(beta)[1]
    pT =  dif_perturbations(beta)[0]

    #graph of ro_pT with a certain alpha and a certain threshold
    x = np.linspace(0, 1212*1212, 1212*1212)
    plt.figure (figsize=(10,10))
    plt.scatter(x, ro_pT)
    plt.title("Ro pT matrix")
    plt.axhline(y=threshold, xmin=0, xmax=0.95, linestyle ="--", color= "#FF7F50", label="Threshold")
    plt.ylabel("Probability")
    plt.xlabel("Primitive pairs")
    plt.legend()
    plt.show()
    
    #perturbing ro_pT above threshold values
    indexes_change = ro_pT >= threshold
    indexes_change = indexes_change.nonzero()
    #creating the new perturbed ro_pT matrix >> new pT matrix 
    new_pT = np.copy(pT) 
    #create dictionary for array replacement 
    dic = {}
    for i in range(indexes_change.shape[0]):
        dic[indexes_change[i]]= constant

    for k, v in dic.items(): new_pT[k[0],k[1]] = v
    
    #normalize new pT matrix 
    new_pT= new_pT/new_pT.sum()
    np.savetxt("./new_pt", new_pT)
    
    #graph of new_pT with a certain beta and a certain threshold,
    plt.figure (figsize=(10,10))
    plt.scatter(x, new_pT, label = "Constant=%d"%constant)
    plt.title("New pT matrix")
    plt.ylabel("Transition probabilities")
    plt.xlabel("Primitive pairs")
    plt.legend()
    plt.show()
    
    #Repeat the process for the logstart matrix
    ro_start = dif_perturbations_start(beta)[1]
    start =  dif_perturbations_start(beta)[0]

    #graph of ro_start with a certain alpha and a certain threshold
    x = np.linspace(0, 1212, 1212)
    plt.figure (figsize=(10,10))
    plt.scatter(x, ro_start)
    plt.title("Ro logStart matrix")
    plt.axhline(y=threshold, xmin=0, xmax=0.95, linestyle ="--", color= "#FF7F50", label="Threshold")
    plt.ylabel("Probability")
    plt.xlabel("Primitives")
    plt.legend()
    plt.show()
    
    #perturbing ro_start above threshold values
    indexes_change = ro_start >= threshold
    indexes_change = indexes_change.nonzero()
    
    #creating the new perturbed ro_start matrix >> new start matrix 
    new_start = np.copy(start)
    #create dictionary for array replacement 
    dic = {}
    for i in range(indexes_change.shape[0]):
        dic[indexes_change[i]]= constant
    
    #change in the correspondent indexes the new values in the new perturbed start matrix 
    for k, v in dic.items(): new_start[k[0]] = v
    
    #normalize new start matrix 
    new_start= new_start/new_start.sum()
    np.savetxt("./new_pt", new_start)
   
    #graph of new_start with a certain alpha a certain threshold and constant
    plt.figure (figsize=(10,10))
    plt.scatter(x, new_start, label = "Constant=%d"%constant)
    plt.title("New logStart matrix")
    plt.ylabel("Transition probabilities")
    plt.xlabel("Primitives")
    plt.legend()
    plt.show()

    return new_pT, new_start

################## TRAJECTORY ENTROPY #########################################################
#reference the code - Mohamed Kafsi - The entropy of consitional markov trajectories
#shannon entropy H= - sum [p(x)logp(x)]
#  input is the irreducible finite state markov chain - in our case  
def local_entropy(P, graph):
    """Computes the local entropy at each state of the MC defined by the transition
    probabilities P"""
    # TODO: Not optimal memory wise !!
    L = np.copy(P)
    L[P > 0] = np.log2(P[P > 0]) #log of matrix entries
    K = np.dot(P, np.transpose(L)) #dot product of the matrix entries and the log of matrix entries
    entropy_out = -1*np.diagonal(K) #negative 
    
    if graph == True:
        y = entropy_out.reshape((P.shape[0], 1))
        x= np.linspace(0,1212,1212)
        plt.figure (figsize=(10,10))
        plt.scatter(x, y)
        plt.title("Local entropy of transition matrix")
        plt.ylabel("Entropy")
        plt.xlabel("Matrix rows")
        plt.show()
    
    return entropy_out.reshape((P.shape[0], 1))


def stationary_distribution(P, graph):
    """Computes the stationary distribution mu associated with the MC whose
    transition probabilities are given by the numpy array P
    IMPORTANT: the MC must be irreducile and aperiodic to admit a sttinary
    distribution
    """
    v = np.real(scipy.linalg.eig(P, left=True, right=False)[1][:, 0])
    mu = np.abs(v)/np.sum(np.abs(v))
    
    if graph == True:
        x= np.linspace(0,1212,1212)
        plt.figure (figsize=(10,10))
        plt.scatter(x, mu)
        plt.title("Stationary distribution of transition matrix")
        plt.ylabel("Probability")
        plt.xlabel("Matrix rows")
        plt.show()
    return mu


def trajectory_entropy(P, graph):
    """Returns the matrix of trajectories entropy H associed to MC whose transition
    probabilities are given by the numpy array P.
    IMPORTANT: the MC is irreducile and aperiodic"""
    n = P.shape[0]
    mu = stationary_distribution(P, graph)
    A = np.tile(mu, (n, 1))
    # local entropies of the MC
    l_entropy = local_entropy(P, graph)
    H_star = np.tile(l_entropy, (1, n))
    # entropy rate
    entropy_rate = np.dot(mu.transpose(), l_entropy)
    H_delta = np.diagflat(entropy_rate/mu)
    K = np.dot(np.linalg.inv(np.identity(n) - P + A), H_star-H_delta)
    K_tilda = np.tile(np.diag(K).transpose(), (n, 1))
    H = K - K_tilda + H_delta
    H = H.sum() #final step of summing the entries of the entropy matrix 
    return H

#trajectory entropy logstart + pT original 
def trajectory_entropy_original_total(graph):
    #getting the pT matrix 
    lib = Library(use_hist= True)
    logR = lib.logT
    R = torch.exp(logR)
    pT = R / torch.sum(R)
    
    #logStart matrix 
    start = torch.exp(lib.logStart) #its not logstart anymore, but it is still de original used matrix
    start = torch.where(start==0, torch.tensor(1e-20, dtype=start.dtype), start)
    
    #joining the two matrices, adding the state 0 (primitive -1 - doesnt exist)
    pad = torch.nn.ZeroPad2d((1,0))
    pT = pad(pT)
    start = pad(start)
    start = torch.tensor([start.numpy()])
    pT = torch.cat((start,pT),0)

    pT = pT.numpy()
    np.set_printoptions(precision=3, suppress=True)
    #print("Transition probability matrix \n {}".format(pT))
    H = trajectory_entropy(pT, graph)
    print("Trajectory entropies matrix original matrix\n {}".format(H))
    return H
    
def trajectory_entropy_total(new_pT, new_start, graph): 
    pad = torch.nn.ZeroPad2d((1,0))
    new_pT = pad(new_pT)
    new_start = pad(new_start)
    new_start = torch.tensor([new_start.numpy()])
    new_pT = torch.cat((new_start,new_pT),0)

    new_pT = new_pT.numpy()
    np.set_printoptions(precision=3, suppress=True)
    #print("Transition probability matrix \n {}".format(new_pT))
    H = trajectory_entropy(new_pT, graph)
    print("Trajectory entropies perturbed matrix\n {}".format(H))
    return H
    
##################### PARAMETER OPTIMIZATION ######################################################

#the function we want to minimize - loss function 
#we want to maximize the difference of entropies between the new matrices and the original ones
#which is the same as minimizing the negative of that difference difference 
# =============================================================================
# def loss_function(param):
#     beta, threshold, constant = param
#     new_pT = diffusion_perturbing(beta, threshold, constant)
#     new_start = diffusion_perturbing_start(beta, threshold, constant)
#     loss =  - (trajectory_entropy_total(new_pT, new_start, False) - trajectory_entropy_original_total(False))
#     print("Loss function value\n {}".format(loss))
#     return loss
# =============================================================================

def loss_function(param):
    beta = param
    new_pT = dif_perturbations(beta)[1]
    new_start = dif_perturbations_start(beta)[1]
    loss =  - (trajectory_entropy_total(new_pT, new_start, False) - trajectory_entropy_original_total(False))
    print("Loss function value\n {}".format(loss))
    return loss

#in order to optimize the beta parameter - the beta trendline will gravitate towards this value 
def grid_search():
    #beta values to take into consideration
    beta = [5, 7,9,10] 
    sample = list()
 
    for x in beta:
        #list with combination of parameter values
         sample.append(x) 
     
    #loss function values for every combination of parameter values              
    sample_eval = [loss_function(beta) for beta in sample]
    best = min(sample_eval)
    index = sample_eval.index(best)
    best_parameters = sample[index]
    
    return best, best_parameters, sample_eval







# =============================================================================
# #in order to optimize the alpha, threshold and constant parameters 
# def grid_search():
#     #alpha values to take into consideration
#     alpha = [0.5,1] 
#     #constant values to take into consideration 
#     constant = [0.25e-7, 0.5e-7] 
#     sample = list()
#     alpha_list=[]
#     threshold_list=[]
#     constant_list=[]
#     for x in alpha:
#         #calculate the ro_pT depending on the alpha value 
#         ro_pT = dif_perturbations(x)[0] 
#         #threshold interval of values depends on ro matrix
#         threshold = [ro_pT.mean(), torch.quantile(ro_pT, 0.25)] 
#         for y in threshold:
#             for z in constant:
#                 #list with combination of parameter values
#                  sample.append([x,y,z]) 
#                  alpha_list.append(x)
#                  threshold_list.append(y)
#                  constant_list.append(z)
#     #loss function values for every combination of parameter values              
#     sample_eval = [loss_function(alpha, threshold,constant) for alpha, threshold, constant in sample]
#     best = min(sample_eval)
#     index = sample_eval.index(best)
#     best_parameters = sample[index]
#     
#     return best, best_parameters, alpha_list, threshold_list, constant_list, sample_eval
# =============================================================================

def plot_hyperparams_grid(alpha_list,threshold_list,constant_list, loss_list):
    dic = dict({"alpha": alpha_list, "threshold": threshold_list, "constant": constant_list ,"loss": loss_list})
    data = pd.DataFrame(dic)
    fig, ax = plt.subplots(figsize=(10, 10))
    rel = sns.relplot(data=data, x='alpha', y='threshold', size='loss', hue="constant")
    rel.fig.suptitle('Parameter grid')
    
def plot_hyperparams_3d(alpha_list,threshold_list,constant_list, loss_list):
    fig = plt.figure(figsize=(10,10))
    
    ax = plt.axes(projection ='3d')
    ax.scatter(alpha_list, threshold_list, constant_list,  c = loss_list)
    ax.set_title('Parameter grid')
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Threshold")
    ax.set_zlabel("Constant")
    
    plt.show()
   
 


#other function to do grid search, trying to optimize the loss function by minimizing it 
#limit of iterations 
#don't know which one is the most efficient 
def optimize_min():
    r = ((1e-6,1.0), (0.1e-7, 1e-7), (1e-5,10e-5))
    res = minimize(loss_function,[0.1,0.8e-7, 0.5e-7], bounds=r)
    
    return res



#other way to do it 
def optimize_pert():
    graph_x1 = []
    graph_x2 = []
    graph_x3 = []
    graph_x4 = []
    for alpha_value in np.arange(0,0.8,0.2):
        alpha_value = alpha_value
        graph_x1_row = []
        graph_x2_row = []
        graph_x3_row = []
        graph_x4_row = []
        ro_pT= dif_perturbations(alpha_value)[0]
        for threshold_value in np.arange(ro_pT.min(),ro_pT.max(),0.1):
            for constant_value in np.arange(1e-5,0.1,1e-5):
                hyperparams = (alpha_value,threshold_value,constant_value)
                loss = loss_function(hyperparams)
                graph_x1_row.append(alpha_value)
                graph_x2_row.append(threshold_value)
                graph_x3_row.append(constant_value)
                graph_x4_row.append(loss)
        graph_x1.append(graph_x1_row)
        graph_x2.append(graph_x2_row)
        graph_x3.append(graph_x3_row)
        graph_x4.append(graph_x4_row)
        print('')
    graph_x1=np.array(graph_x1)
    graph_x2=np.array(graph_x2)
    graph_x3=np.array(graph_x3)
    graph_x4=np.array(graph_x4)
    max_loss = np.max(graph_x4)
    pos_max_loss = np.argwhere(graph_x4 == np.max(graph_x4))[0]
    print('Maximum entropy difference: %.4f' %(max_loss))
    print('Optimum alpha: %f' %(graph_x1[pos_max_loss[0],pos_max_loss[1]]))
    print('Optimum threshold: %f' %(graph_x2[pos_max_loss[0],pos_max_loss[1]]))
    print('Optimum constant: %f' %(graph_x3[pos_max_loss[0],pos_max_loss[1]]))
    
    return graph_x1,graph_x2,graph_x3, graph_x4





######################### DIFFUSION MAP - VISUALIZE PRIMITIVE SPACE IN 2D ###################

def find_diffusion_matrix(pT, alpha=0.15):
    """Function to find the diffusion matrix P
        
        >Parameters:
        alpha - to be used for gaussian kernel function
        X - feature matrix as numpy array
        
        >Returns:
        P_prime, P, Di, K, D_left
    """
    alpha = alpha
        
    dists = - alpha * np.log(pT) 
    K = (torch.exp(dists)).numpy()
    
    r = np.sum(K, axis=0)
    Di = np.diag(1/r)
    P = np.matmul(Di, K)
    
    D_right = np.diag((r)**0.5)
    D_left = np.diag((r)**-0.5)
    P_prime = np.matmul(D_right, np.matmul(P,D_left))

    return P_prime, P, Di, K, D_left


def find_diffusion_map(P_prime, D_left, n_eign=3):
    """Function to find the diffusion coordinates in the diffusion space
        
        >Parameters:
        P_prime - Symmetrized version of Diffusion Matrix P
        D_left - D^{-1/2} matrix
        n_eigen - Number of eigen vectors to return. This is effectively 
                    the dimensions to keep in diffusion space.
        
        >Returns:
        Diffusion_map as np.array object
    """   
    n_eign = n_eign
    
    eigenValues, eigenVectors = eigh(P_prime)
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    
    diffusion_coordinates = np.matmul(D_left, eigenVectors)
    
    return diffusion_coordinates[:,:n_eign]

def plot_2Dsub_figures(d_map, alpha_values, title='Diffused points'):
    subplot_titles=[f'α={round(a,4)}' for a in alpha_values]
    fig = make_subplots(rows=2, cols=5,subplot_titles=subplot_titles)
    for i in range(1,3):
        for j in range(1,6):
            dmap_idx = i+j-1
            fig.add_trace(
                go.Scatter(x=d_map[dmap_idx][:,0], y=d_map[dmap_idx][:,1], mode='markers', marker=dict(
                size=3,color=d_map[dmap_idx][:,1],opacity=0.8,colorscale='Viridis')),row=i, col=j)

    fig.update_layout(title_text=title, title_x=0.5)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(height=500, width=1000, showlegend=False)
    fig.show()
   
    
def apply_diffusions(alpha_start=0.001, alpha_end= 0.009, title='Diffused points'):
    lib = Library(use_hist= True)
    logR = lib.logT
    R = torch.exp(logR)
    pT = R / torch.sum(R)
    d_maps = []
    alpha_values = np.linspace(alpha_start, alpha_end, 10)
    for alpha in alpha_values:
        P_prime, P, Di, K, D_left = find_diffusion_matrix(pT, alpha=alpha)
        d_maps.append(find_diffusion_map(P_prime, D_left, n_eign=2))
    return d_maps, alpha_values


##################### OTHER MEASURES TO COMPARE MATRICES #######################################

#kl divergence is a distance metric that quantifies the difference between two probbaility distributions.
#This is not a symmetric metric, so if we calculate the kl divergence of q from p the value will be different 
#than if we do it vice versa 
def kl_divergence(p,q):
    kl= sum(rel_entr(p, q)) #the unit is nats, but here it is refered in terms of bits 
    return kl 

def jensen_shannon_distance(p, q):
    """
    method to compute the Jenson-Shannon Distance 
    between two probability distributions
    """

    # convert the vectors into numpy arrays in case that they aren't
    p = np.array(p)
    q = np.array(q)

    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2

    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)

    return distance   

def ks_test(p,q): #im not sure if it makes sense to do this 
    result = ks_2samp(p,q)
    return result 