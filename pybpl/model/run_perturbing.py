#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 16:11:47 2022

@author: carolinacaramelo
"""
from Perturbing import Perturbing
from pybpl.library import Library
from pybpl.model.type_dist import ConceptTypeDist, CharacterTypeDist, PartTypeDist, StrokeTypeDist
import torch
import matplotlib.pyplot as plt
import numpy as np

# load the hyperparameters of the BPL graphical model (i.e. the "library")
lib = Library (use_hist = True)

#pp = Perturbing ()
#pp.truncateT ([0,1,2])
#ids = torch.tensor ([1,2])
#pp.pT_truncate ([0,1,2], ids[0]
#nsub= torch.tensor ([4])
#pp.sample_ids([80,45,210], nsub, True, 0)
#pp.run_example()
#pp.n_strokes = 2 # number of strokes
#subpart_sample = np.linspace(0,1212,1212, dtype = int)
#subp_list = [np.random.choice(subpart_sample),np.random.choice(subpart_sample)] 
#print(subp_list)
#pp.subparts = subp_list #possible ids to use when sampling the id list 
#pp.subparts = [100,305] #possible ids to use when sampling the id list 
#pp.alpha = 0 
#pp.nsub = torch.tensor ([2])

pp = Perturbing ()
pp.run_example(2,3,0.1,3)



#pp.run_n_examples(2, False)

#pp.run_n_examples(2, True)






#aa = StrokeTypeDist (lib)
#print (aa.sample_subIDs(nsub[0]))
#s= lib.logStart
#print(s[0])

#subparts = [1,2,3]
#pp.truncate_start(subparts)
