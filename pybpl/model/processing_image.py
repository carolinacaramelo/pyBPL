#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 14:16:17 2022

@author: carolinacaramelo
"""

import numpy as np
from PIL import Image
#import imutils
import cv2
import skimage.exposure


def process_image(im, num):
    #Process images after generative phase in order to do inference 
    #Posterior to this, this will have to be transformed in a loop in order to process every generated new image
    
    ##1.Convert to white all of the non black pixels of an image 
    
    # Separate RGB arrays
    
    R, G, B = im.convert('RGB').split()
    r = R.load()
    g = G.load()
    b = B.load()
    w, h = im.size
    
    # Convert non-black pixels to white
    for i in range(w):
        for j in range(h):
            if(r[i, j] != 0 or g[i, j] != 0 or b[i, j] != 0):
                r[i, j] = 255 # Just change R channel
    
    # Merge just the R channel as all channels
    im = Image.merge('RGB', (R, R, R))
    im.save("./original_final.png")
    
    ##2. CONVERT TO GREYSCALE 
    img = cv2.imread("./original_final.png")
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ##3. CONVERT TO BINARY IMAGE 
    
    ret, thresh = cv2.threshold(imgray, 127, 255, 0, cv2.THRESH_BINARY)
    
    cv2.imwrite("./binary.png", thresh)
    
    
    ##4.NEGATIVE IMAGE 
    
    
    img = cv2.bitwise_not(thresh)
    cv2.imwrite("./binary_neg.png", img)
    
    
    ##5.CLEAN NOISE 
    
    
    img = np.uint8(img)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE) # Use cv2.CCOMP for two level hierarchy
    
    # create an empty mask
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    # loop through the contours
    for i, cnt in enumerate(contours):
    
        if hierarchy[0][i][3] != -1: # basically look for holes
            # if the size of the contour is less than a threshold (noise)
            if cv2.contourArea(cnt) < 70:
                # Fill the holes in the original image
                cv2.drawContours(img, [cnt], 0, (255), -1)
    
    
    image = cv2.bitwise_not(img, img, mask=mask)
    cv2.imwrite("./clean.png", image)
    
    
    
    ##6.REMOVE ISOLATED PIXELS 
    
    
    
    input_image_comp = cv2.bitwise_not(image)  # could just use 255-img
    
    kernel1 = np.array([[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]], np.uint8)
    kernel2 = np.array([[1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]], np.uint8)
    
    hitormiss1 = cv2.morphologyEx(image, cv2.MORPH_ERODE, kernel1)
    hitormiss2 = cv2.morphologyEx(input_image_comp, cv2.MORPH_ERODE, kernel2)
    hitormiss = cv2.bitwise_and(hitormiss1, hitormiss2)
    
    hitormiss_comp = cv2.bitwise_not(hitormiss)  # could just use 255-img
    del_isolated = cv2.bitwise_and(image,image, mask=hitormiss_comp)
    cv2.imwrite("./clean_final7.png", del_isolated)
    
    
    
    
    ##7.SMOOTH EDGES
    
    # blur threshold image
    blur = cv2.GaussianBlur(img, (0,0), sigmaX=3, sigmaY=3, borderType = cv2.BORDER_DEFAULT)
    
    result = skimage.exposure.rescale_intensity(blur, in_range=(127.5,255), out_range=(0,255))
    
    cv2.imwrite("./image%d"%num+".png", result)
    
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    
    
    
    
    
    
    ##2.convert generative phase image to grey scale and binary image
    
    #Convert image to greeyscale
    #im_gray = np.array(Image.open("./original_final.png").convert('L'))
    #print (type(im_gray))
    
    #Cinvert image to binary image 
    #maxval = 255
    #thresh = 128
    
    
    #im_bin = (im_gray > thresh) * maxval
    #print(im_bin.shape)
    #print(im_bin)
    
    #Image.fromarray(np.uint8(im_bin)).save('./binary.png')
    
    #im = Image.open("./binary.png")
    #im_bin = im_bin.astype('uint8')
    #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    #closing = cv2.morphologyEx(im_bin, cv2.MORPH_CLOSE, kernel)
    
    #cv2.imwrite("./clean.png", closing)
    
    
    
    
    
    #Convert png to jpg
    #im = Image.open('./numpy_binarization.png')
    #rgb_im = im.convert('RGB')
    #rgb_im.save('./numpy_binarization.jpg')
    
    
    
