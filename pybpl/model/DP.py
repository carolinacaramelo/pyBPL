#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 10:21:27 2022

@author: carolinacaramelo
"""

from numpy.random import choice
import matplotlib.pyplot as plt
from scipy.stats import beta, norm
import numpy as np
from scipy.stats import dirichlet
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sn
from IPython.display import clear_output
from IPython.core.display import display, HTML
from pybpl.library import Library
import torch 
import torch.distributions as dist

class DirichletProcessSample():
    def __init__(self, base_measure, alpha):
        self.base_measure = base_measure
        self.alpha = alpha
        
        self.cache = []
        self.weights = []
        self.total_stick_used = 0.

    def __call__(self):
        remaining = 1.0 - self.total_stick_used
        i = DirichletProcessSample.roll_die(self.weights + [remaining])
        if i is not None and i < len(self.weights) :
            return self.cache[i]
        else:
            stick_piece = beta(1, self.alpha).rvs() * remaining
            self.total_stick_used += stick_piece
            self.weights.append(stick_piece)
            new_value = self.base_measure()
            self.cache.append(new_value)
            return new_value
        
    @staticmethod 
    def roll_die(weights):
        if weights:
            return choice(range(len(weights)), p=weights)
        else:
            return None
    
    
def example():
    lib = Library (use_hist = True)
    base_measure =  lambda: dist.Categorical(probs=lib.pkappa).sample() + 1
    n_samples = 1
    samples = {}
    alpha = 0.1
    dirichlet_norm = DirichletProcessSample(base_measure=base_measure, alpha=alpha)
    print(dirichlet_norm)
  

    for i in range (100):
        samples["Alpha: %s" % alpha] = [dirichlet_norm()]
        print(samples)
        
        