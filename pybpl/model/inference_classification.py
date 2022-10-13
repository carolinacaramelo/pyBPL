#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 15:01:24 2022

@author: carolinacaramelo
"""

import os
from processing_image import process_image
from Perturbing import Perturbing
from pybpl.library import Library
from PIL import Image
import warnings
import matlab.engine
import torch
import re

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



def list_files(dir):

    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))
    r = sorted(r)
    return r

def inference(): #this has to be run one classification run at a time 
    dir = "/Users/carolinacaramelo/Desktop/Thesis_Results/PHASE_4_CLASSIFICATION/evaluation_set/runs_classification/run02" #add the run
    r = list_files(dir)
    
    for num in range(len(r)):
            try:
                print(num)
                im = Image.open(r[num])
                process_image(im, num)
                m = re.search('run02/(.+?).png', r[num]) #change the name of the run
                eng.demo_fit_classification(1,-(("_"+m.group(1).replace("/", "")).replace("0","")).replace("ingclass",""),num,nargout=0) #run - substituir pelo número da run
                
                print("done")
            except:
                pass

def main():
    inference()
    

if __name__ == "__main__":
    main()      
       
        