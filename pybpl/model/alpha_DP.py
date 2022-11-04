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