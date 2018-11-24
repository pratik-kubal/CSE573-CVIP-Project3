#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 15:16:30 2018

@author: pratik
"""
UBIT = 'pkubal';
import numpy as np;
import cv2
np.random.seed(sum([ord(c) for c in UBIT]))
from matplotlib import pyplot as plt

seg_image = cv2.imread("./original_imgs/segment.jpg",0)
def threshold(thresholdVal,image):
    resultImage = image.copy()
    for h in range(resultImage.shape[0]):
        for w in range(resultImage.shape[1]):
            if(resultImage[h][w] > thresholdVal):
                resultImage[h][w] = 1
            else:
                resultImage[h][w] = 0

plt.hist(seg_image.ravel(),256,[1,256])
plt.savefig('./2/res_segment.jpg')