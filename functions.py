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


def threshold(thresholdVal,image,thresholdLimit=255):
    import numpy as np;
    resultImage = image.copy()
    for h in range(resultImage.shape[0]):
        for w in range(resultImage.shape[1]):
            if(resultImage[h][w] > thresholdVal and resultImage[h][w] <= thresholdLimit):
                resultImage[h][w] = 1
            else:
                resultImage[h][w] = 0
    return np.array(resultImage,dtype='int8')

def normImage(matA):
    import numpy as np
    skeleton=[[] for i in range(0,np.shape(matA)[0])]
    maxValue = 0
    absValue = 0
    for window_h in range(0,np.shape(matA)[0]):
        for window_w in range(0,np.shape(matA)[1]):
            absValue = abs(matA[window_h][window_w])
            skeleton[window_h].append(absValue)     
            if(maxValue < absValue) : maxValue = absValue
    returnMat=[[] for i in range(0,np.shape(matA)[0])]
    for window_h in range(0,np.shape(matA)[0]):
        for window_w in range(0,np.shape(matA)[1]):
            returnMat[window_h].append(skeleton[window_h][window_w] / maxValue)
    return returnMat