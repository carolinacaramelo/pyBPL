#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 11:35:44 2022

@author: carolinacaramelo
"""

# import the modules
import os
import warnings
import matlab.engine
from processing_image import process_image
from Perturbing import Perturbing
from pybpl.library import Library
from PIL import Image
import torch
import numpy as np

#start matlab engine - run inference code 

# start matlab engine
eng = matlab.engine.start_matlab()

try:
    # add BPL code to matlab path
    bpl_path = os.environ['BPL_PATH']
    eng.addpath(eng.genpath(bpl_path), nargout=0)
except:
    warnings.warn('BPL_PATH environment variable not set... therefore you'
                  'must have BPL matlab repository already added to your matlab '
                  'path')

# add current directory to matlab path
eng.addpath(os.path.dirname(__file__), nargout=0)

dir = '/Users/carolinacaramelo/Desktop/treinos/alphabet_test1'

def list_files(dir):
    
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))
    return r

def inference():
    model = Perturbing()
    dir = '/Users/carolinacaramelo/Desktop/inf_test' #put here the directory that we are supposed to use - where tha alphabets are stored
    r = list_files(dir)
    num = 0
    list_ids = []
    path = "./processed"
    os.makedirs(path)
    list_ch_omniglot =[]
    list_ch_perturbed =[]
    list_dir_omniglot = []
    list_dir_perturbed = []

    for root in os.listdir(dir):
        root = os.path.join(dir,root)
        for dirs in os.listdir(root):
            dirs = os.path.join(root, dirs)
            for files in os.listdir(dirs):
                files = os.path.join(dirs, files)
                for name in os.listdir(files):
                    name = os.path.join(files, name)
                    try:
                        print(name)
                        #im = Image.open(r[num])
                        im =  Image.open(name)
                        process_image(im, num)
                        eng.demo_fit_perturbing(num,nargout=0)
                        print("done")
                        model.load()
                        print(model.ids)
                        list_ids += model.ids
                        num +=1
                        
                        if root == "/Users/carolinacaramelo/Desktop/inf_test/perturbed":
                            list_ch_perturbed +=[model.ids]
                            list_dir_perturbed += [name]
                            print(root)
                            print(list_ch_perturbed)
                        if root ==  "/Users/carolinacaramelo/Desktop/inf_test/omniglot":
                            list_ch_omniglot +=[model.ids]
                            list_dir_omniglot += [name]
                            print(root)
                            print(list_ch_omniglot)
                            
                    except:pass
                    
    list_ch_omniglot = np.array(list_ch_omniglot)
    list_ch_perturbed = np.array(list_ch_perturbed)
    list_dir_omniglot = np.array(list_dir_omniglot)
    list_dir_perturbed = np.array(list_dir_perturbed)
    np.savetxt("./list_ids_perturbed", list_ch_perturbed,fmt='%s')
    np.savetxt("./list_ids_omniglot", list_ch_omniglot,fmt='%s')
    np.savetxt("./list_dir_perturbed", list_dir_perturbed,fmt='%s')
    np.savetxt("./list_dir_omniglot", list_dir_omniglot,fmt='%s')
   
    return list_ids
    
     

def empirical_countings(): 
    list_ids = inference()
    logstart = torch.zeros(1212)
    pT = torch.zeros(1212,1212)
    total_trans = 0
    for l in list_ids:
        start = 0
        ids = l.tolist()
        total_trans += len(ids)-1
        for i in range(len(ids)):
                   try:
                       pT[ids[i], ids[i+1]] +=1
                       
                   except:
                       pass    
        for k in l:
            
            if l.size() == 1:
                logstart[k.item()] += 1
            else:
                if start==0:
                    logstart[k.item()] += 1
            start +=1 
    print(pT)
    print(logstart)
    print(total_trans)
    logstart = logstart/ len(list_ids) #new logstart
    pT = pT / total_trans #new pT
    print(pT)
    print(logstart)
    list_ids = np.array(list_ids)
    
    np.savetxt("./list_ids_empirical_countings", list_ids,fmt='%s')
    
    return logstart, pT, list_ids
    
                
def main(): #provavelmente vou ter de fazer um except aqui porque existe uma entrada da logstart que é infinita 
# =============================================================================
#     lib = Library (use_hist=True)
#     #getting original logstart matrix 
#     original_logstart = lib.logStart
#     #getting original pT matrix 
#     logT = lib.logT
#     R = torch.exp(logT)
#     original_pT = R / torch.sum(R)
# =============================================================================
    
    #doing inference and estimating new matrices 
    results = empirical_countings() #empirical counting does the inference inside
    final_start = results[0]
    final_start = np.array(final_start)
    final_pT = results[1]
    final_pT = np.array(final_pT)
    
    #final matrices 
    #final_logstart = (original_logstart + new_logstart)/2 #still have to decide how to do this 
    #final_pT = (original_pT + new_pT)/2
    
    np.savetxt("./final_start",final_start)
    np.savetxt("./final_pT", final_pT)
    
    return final_start, final_pT #talvez faça sentido guardar as matrizes no bloco de notas e depois fazer upload onde for necessário
    
                 
        
    