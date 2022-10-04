#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 11:35:44 2022

@author: carolinacaramelo
"""

# import the modules
import os
from processing_image import process_image
from Perturbing import Perturbing
from pybpl.library import Library
from PIL import Image
import warnings
import matlab.engine
import torch

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
    dir = '/Users/carolinacaramelo/Desktop/treinos/alphabet_test1' #put here the directory that we are supposed to use - where tha alphabets are stored
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
    
    return logstart, pT
    
                
                
        
    