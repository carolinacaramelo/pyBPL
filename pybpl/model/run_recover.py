#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 14:39:24 2022

@author: carolinacaramelo
"""

from Perturbing import Perturbing
from pybpl.library import Library
import torch


import matplotlib.pyplot as plt
import numpy as np
from processing_image import process_image 

from PIL import Image


def pre_recover():
    model = Perturbing ()
    model.test_dataset(4,1,0,False)
    im = Image.open("./original.png")
    process_image(im)
    
    
    
    


def recover():

    lib = Library (use_hist = True)
    model = Perturbing ()
    #Get needed parameters extracted from inference 
    model.load()
    with open("./test2", 'w') as f:
        f.write("PARAMETERS FROM FIRST INFERENCE")
        f.write("ids_1 {}".format(model.ids))
        f.write("invscales_1 {}".format(model.inv_type))
        f.write("nsub_1 {}".format(model.nsub))
        f.write("relation category_1{}".format(model.r))
        f.write("gpos_1 {}".format(model.gpos))
        f.write("subid_spot_1 {}".format(model.subid_spot))
        f.write("attach_spot_1 {}".format(model.attach_spot))
        f.write("ncpt_1 {}".format(model.ncpt))
        f.write("nprev_1{}".format(model.nprev))
        f.write("shapes_token_1{}".format(model.shapes_token))
        f.write("invscales_token_1{}".format(model.inv_token))
        f.write("pos_token_1{}".format(model.pos_token))
        f.write("eval_spot_token_1{}".format(model.eval_spot_token))
    
    
    #generate image
    c_type = model.known_stype_recover(False)
    c_token = model.model_sample_token_recover(c_type)
    c_image = model.CM.sample_image(c_token)
    plt.rcParams["figure.figsize"] = [105, 105]
    plt.imsave('./original_after_inf.png', c_image, cmap='Greys')
   
    #image processing 
    im = Image.open("./original_after_inf.png")
    process_image(im)
  
def recover_2():
    lib = Library (use_hist = True)
    model = Perturbing ()
    model.load2()
    
    with open("./test2_after", 'w') as f:
       f.write("PARAMETERS FROM SECOND INFERENCE")
       f.write("ids_2 {}".format(model.ids))
       f.write("invscales_2 {}".format(model.inv_type))
       f.write("nsub_2 {}".format(model.nsub))
       f.write("relation category_2{}".format(model.r))
       f.write("gpos_2 {}".format(model.gpos))
       f.write("subid_spot_2 {}".format(model.subid_spot))
       f.write("attach_spot_2 {}".format(model.attach_spot))
       f.write("ncpt_2 {}".format(model.ncpt))
       f.write("nprev_2{}".format(model.nprev))
       f.write("shapes_token_2{}".format(model.shapes_token))
       f.write("invscales_token_2{}".format(model.inv_token))
       f.write("pos_token_2{}".format(model.pos_token))
       f.write("eval_spot_token_2{}".format(model.eval_spot_token))
        
    
  
   