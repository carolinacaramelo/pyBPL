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
    dir = '/Users/carolinacaramelo/Desktop/alphabet' #put here the directory that we are supposed to use - where tha alphabets are stored
    r = list_files(dir)
    num = 1
    list_ids = []
    path = "./processed"
    os.makedirs(path)
    for root, dirs, files in os.walk(dir):
        for name in files:
            try:
                print(num)
                im = Image.open(r[num])
                process_image(im, num)
                eng.demo_fit_perturbing(num,nargout=0)
                print("done")
                model.load()
                print(model.ids)
                list_ids += model.ids
                num +=1
            except:
                pass
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
    np.savetxt("./list_ids_empirical_countings", list_ids)
    
    return logstart, pT, list_ids
    
                
def main(): #provavelmente vou ter de fazer um except aqui porque existe uma entrada da logstart que é infinita 
    lib = Library (use_hist=True)
    #getting original logstart matrix 
    original_logstart = lib.logStart
    #getting original pT matrix 
    logT = lib.logT
    R = torch.exp(logT)
    original_pT = R / torch.sum(R)
    
    #doing inference and estimating new matrices 
    results = empirical_countings() #empirical counting does the inference inside
    new_logstart = results[0]
    new_pT = results[1]
    
    #final matrices 
    final_logstart = (original_logstart + new_logstart)/2 #still have to decide how to do this 
    final_pT = (original_pT + new_pT)/2
    
    np.savetxt("./final_logstart", final_logstart)
    np.savetxt("./final_pT", final_pT)
    
    return final_logstart, final_pT #talvez faça sentido guardar as matrizes no bloco de notas e depois fazer upload onde for necessário
    
                 
        
    