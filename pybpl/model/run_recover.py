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
from skimage.metrics import structural_similarity as ssim
import cv2
import matlab.engine
import os
import warnings
from sewar.full_ref import mse, rmse


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


def pre_recover():
    model = Perturbing ()
    model.test_dataset(3,1,0,False)
    im = Image.open("./original.png")
    process_image(im, 0)
    
    
    
def recover():

    lib = Library (use_hist = True)
    model = Perturbing()
    #Get needed parameters extracted from inference 
    model.load()
    with open("./test6", 'w') as f:
        f.write("PARAMETERS FROM FIRST INFERENCE")
        f.write("ids_1 {}".format(model.ids))
        f.write("invscales_1 {}".format(model.inv_type))
        f.write("nsub_1 {}".format(model.nsub))
        f.write("relation category_1{}".format(model.r))
        f.write("gpos_1 {}".format(model.gpos))
        f.write("subid_spot_1 {}".format(model.subid_spot))
        f.write("attach_spot_1 {}".format(model.attach_spot))
        f.write("ncpt_1 {}".format(model.ncpt_r))
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
    process_image(im, 1)
  
def recover_2():
    lib = Library (use_hist = True)
    model = Perturbing ()
    model.load2()
    
    with open("./test6_after", 'w') as f:
       f.write("PARAMETERS FROM SECOND INFERENCE")
       f.write("ids_2 {}".format(model.ids))
       f.write("invscales_2 {}".format(model.inv_type))
       f.write("nsub_2 {}".format(model.nsub))
       f.write("relation category_2{}".format(model.r))
       f.write("gpos_2 {}".format(model.gpos))
       f.write("subid_spot_2 {}".format(model.subid_spot))
       f.write("attach_spot_2 {}".format(model.attach_spot))
       f.write("ncpt_2 {}".format(model.ncpt_r ))
       f.write("nprev_2{}".format(model.nprev))
       f.write("shapes_token_2{}".format(model.shapes_token))
       f.write("invscales_token_2{}".format(model.inv_token))
       f.write("pos_token_2{}".format(model.pos_token))
       f.write("eval_spot_token_2{}".format(model.eval_spot_token))
        


def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err



def main():
    pre_recover()
    err_list1= []
    err_list2= []
    err_list3= []
    
       
    try:
        for i in range (0,7):
            #try:
            model = Perturbing()
            # call matlab fn, do inference
            eng.demo_fit_perturbing(i,nargout=0)
            #load image parameters
            model.load()
            print("done")
            with open("./iteration%d"%i, 'w') as f:
                f.write("PARAMETERS FROM %d" %i + "INFERENCE")
                f.write("ids_1 {}".format(model.ids))
                f.write("invscales_1 {}".format(model.inv_type))
                f.write("nsub_1 {}".format(model.nsub))
                f.write("relation category_1{}".format(model.r))
                f.write("gpos_1 {}".format(model.gpos))
                f.write("subid_spot_1 {}".format(model.subid_spot))
                f.write("attach_spot_1 {}".format(model.attach_spot))
                f.write("ncpt_1 {}".format(model.ncpt_r))
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
            plt.imsave("./image_generated%d"%(i+1)+".png", c_image, cmap='Greys')
            plt.imsave("/Users/carolinacaramelo/Desktop/Thesis_results/image_generated%d"%(i+1)+".png", c_image, cmap='Greys')
            
            print("done2")
            
            imageB = Image.open("./image_generated%d"%(i+1)+".png")
            process_image(imageB,(i+1))
            print("done3")
            
            #structural similarity index
            #image processing 
            imageA= cv2.imread("./image0.png")
            imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
            imageB = cv2.imread("./image%d"%(i+1)+".png")
            imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
            print("done4")
            
         
            err1 = ssim(imageA, imageB)
            err_list1 += [err1]
            print(err_list1)
        
            err2 = mse(imageB, imageA)
            
            err_list2 += [err2]
            print(err_list2)
            
            err3 = rmse(imageB, imageA)
            err_list3 += [err3]
            print(err_list3)
            
            iteration = i
            

        x = np.linspace(0,iteration,iteration+1).tolist()       
           
        plt.bar(x, err_list1, align="center")
        plt.title('SSIM Original image VS. images recovered with inference')
        plt.xticks(np.arange(0, iteration, 1))
        plt.yticks(np.arange(0, 1, 0.05))
        plt.xlabel('Iteration')
        plt.ylabel('SSIM')
        plt.show()
        
        plt.bar(x, err_list2, align = "center")
        plt.title('MSE Original image VS. images recovered with inference')
        plt.xticks(np.arange(0, iteration, 1))
        plt.yticks(np.arange(0, 1000, 100))
        plt.xlabel('Iteration')
        plt.ylabel('MSE')
        plt.show()
        
        plt.bar(x, err_list3, align = "center")
        plt.title('RMSE Original image VS. images recovered with inference')
        plt.xticks(np.arange(0, iteration, 1))
        plt.yticks(np.arange(0, 50, 10))
        plt.xlabel('Iteration')
        plt.ylabel('RMSE')
        plt.show()
    
    except:
        StopIteration
        




if __name__ == "__main__":
    main()      
       
        
    