#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 00:37:05 2018

@author: pratik
"""

def dilation(structuring_element,img):
    import numpy as np;
    strucArray = structuring_element.copy().reshape(1,9).flatten()
    skeleton=[[0 for i in range(0,np.shape(img)[1])] for i in range(0,np.shape(img)[0])]
    for h in range(np.shape(img)[0]-2):
        for w in range((np.shape(img)[1])-2):
            sliced_img = img[h:h+3,w:w+3].copy().reshape(1,9).flatten()
            if(not (sliced_img == 0).all()):
                if((sliced_img == strucArray).any()):
                    skeleton[h+1][w+1] = 1
                    
    return np.asarray(skeleton,dtype='int8')

def erosion(structuring_element,img):
    import numpy as np;
    strucArray = structuring_element.copy().reshape(1,9).flatten()
    keyStruct=[]
    for i,boolVal in enumerate(strucArray == 1): 
        if(boolVal == True):
            keyStruct.append(i)
    skeleton=[[0 for i in range(0,np.shape(img)[1])] for i in range(0,np.shape(img)[0])]
    for h in range(np.shape(img)[0]-2):
        for w in range((np.shape(img)[1])-2):
            sliced_img = img[h:h+3,w:w+3].copy().reshape(1,9).flatten()
            if(not (sliced_img == 0).all()):
                for i in keyStruct:
                    if((sliced_img[keyStruct] == 1).all()):
                        skeleton[h+1][w+1] = 1
    return np.asarray(skeleton,dtype='int8')