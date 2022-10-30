#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 11:34:52 2022

@author: carolinacaramelo
"""

import os
import warnings
import matlab.engine
import numpy as np 
import matplotlib as plt

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

def run_classification():
    results = eng.run_classification_perturbing()
    
    x = np.linspace(0,20)       
    
    plt.figure(figsize=(10,10))   
    plt.bar(x, results)
    plt.title('One-shot classification results') #acrescentar depois o nome das perturbações 
    plt.xticks(np.arange(0, 20, 1))
    plt.yticks(np.arange(0, 1, 0.05))
    plt.xlabel('Run')
    plt.ylabel('One-shot classification error (%)')
    plt.show()
    
    