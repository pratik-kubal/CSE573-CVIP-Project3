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

hist_norm_image = cv2.equalizeHist(seg_image.copy())
hist_norm_image = cdfNorm(seg_image.copy())
vect_image = hist_norm_image.reshape((hist_norm_image.shape[0]*hist_norm_image.shape[1],1))
vect_image = np.array(vect_image,dtype='float32')


def cdfNorm(image):
    vect_image = image.reshape((image.shape[0]*image.shape[1],1))
    vect_image = np.array(vect_image,dtype='float32')
    # Histogram Equilization
    a,b = np.unique(vect_image,return_counts=True)
    L = 256
    p = np.divide(b,np.sum(b))
    # https://www.math.uci.edu/icamp/courses/math77c/demos/hist_eq.pdf
    cdf = np.floor(np.multiply(L-1,np.cumsum(p)))

    resultImage = image.copy()
    cdf_dict = {}
    for i,item in enumerate(list(a)):
        cdf_dict.update({item:cdf[i]})
    for h in range(resultImage.shape[0]):
        for w in range(resultImage.shape[1]):
            if(resultImage[h,w]) ==0:
                result_img[h,w] = result_img[h,w]
            else:
                result_img[h,w] = cdf_dict.get(resultImage[h,w])
    return np.array(result_img)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,100, 1.0)

_,_,centers = cv2.kmeans(vect_image,4,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

thresholdVal = np.sum(centers)/2

def threshold(thresholdVal,image):
    resultImage = image.copy()
    for h in range(resultImage.shape[0]):
        for w in range(resultImage.shape[1]):
            if(resultImage[h][w] > thresholdVal):
                resultImage[h][w] = 1
            else:
                resultImage[h][w] = 0
    return np.array(resultImage,dtype='uint8')

result_img = threshold(thresholdVal,hist_norm_image)
plt.hist(hist_norm_image.ravel(),256,[81,256])
plt.savefig('./2/histo_norm.jpg')

cv2.imwrite('./2/res_segment.jpg',np.multiply(result_img,255))

cv2.imwrite('./2/res_segment.jpg',hist_norm_image)

cv2.imwrite('./2/res_segment2.jpg',hist_norm_image)