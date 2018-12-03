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
from functions import threshold,histogram,pointDetection

seg_image = cv2.imread("./original_imgs/segment.jpg",0)

optimal_thresholdingMethod = histogram(image=seg_image)
optimalImage = optimal_thresholdingMethod.optimalThresholding()

adptHistMethod = histogram(image=seg_image,tileSize=[15,15])
image=adptHistMethod.adaptiveHist()

image = seg_image.copy()
vect_image = image.reshape((image.shape[0]*image.shape[1],1))
vect_image = np.array(vect_image,dtype='float32')
a,b = np.unique(vect_image,return_counts=True)
# Zero vals are too much so to see the actual distribution need to remove it
plt.bar(a[1:],b[1:])
'''
CHANGE TRUE IF GAUSSIAN FILTERING
'''
specificThresholdingMethod =histogram(image=seg_image,NoiseReduction=False,kernelSize=(3,3))
image = specificThresholdingMethod.specificThresholding([200,256],True)

# Point Detection
del point_img
point_img = cv2.imread("./original_imgs/point.jpg",0)
porosity = pointDetection(image=point_img,kernelSize=(3,3))
porosityPoint = porosity.pointDet(thresholdVal=[230,266])



