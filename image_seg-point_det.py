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
from functions import threshold


seg_image = cv2.imread("./original_imgs/segment.jpg",0)
image = seg_image.copy()
vect_image = image.reshape((image.shape[0]*image.shape[1],1))
vect_image = np.array(vect_image,dtype='float32')
a,b = np.unique(vect_image,return_counts=True)
# Zero vals are too much so to see the actual distribution need to remove it
plt.bar(a[1:],b[1:])

naive_thres_bin_image = threshold(200,image,240)

cv2.imwrite('./2/naivvve.jpg',np.multiply(naive_thres_bin_image,255))

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
            resultImage[h,w] = cdf_dict.get(resultImage[h,w])
    return np.array(resultImage)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,100, 1.0)

_,_,centers = cv2.kmeans(vect_image,5,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

thresholdVal = np.sum(centers)

variance = a


result_img = threshold(thresholdVal,hist_norm_image)
plt.hist(hist_norm_image.ravel(),256,[81,256])
plt.savefig('./2/histo_norm.jpg')

cv2.imwrite('./2/res_segment.jpg',np.multiply(result_img,255))

cv2.imwrite('./2/res_segment.jpg',hist_norm_image)

cv2.imwrite('./2/res_segment2.jpg',hist_norm_image)




# Point Detection
point_img = cv2.imread("./original_imgs/point.jpg",0)
mask = [[-1,-1,-1,-1,-1],
        [-1,-16,-16,-16,-1],
        [-1,-16,128,-16,-1],
        [-1,-16,-16,-16,-1],
        [-1,-1,-1,-1,-1]]
mask = np.array(mask,dtype='float32')
det_img = pointDet(point_img,mask)
# NORMALIZE IMAGE
det_img = np.array(det_img,dtype='float32')
np.max(det_img)
test = normImage(det_img)
test = np.multiply(test,255)
test = np.array(test,dtype='uint8')
bin_det_img = threshold(1260,det_img,thresholdLimit=1312)
cv2.imwrite('./2/res_point1.jpg',np.multiply(bin_det_img,255))
cv2.imwrite('./2/res_point.jpg',test)

def pointDet(img,mask):
    img = point_img.copy()
    vect_mask = mask.reshape(1,mask.shape[0]*mask.shape[1])
    skeleton=[[0 for i in range(0,np.shape(img)[1])] for i in range(0,np.shape(img)[0])]
    for h in range(np.shape(img)[0]-mask.shape[0]):
        for w in range((np.shape(img)[1])-mask.shape[1]):
            sliced_img = img[h:h+mask.shape[0],w:w+mask.shape[1]].reshape(
                    mask.shape[0]*mask.shape[1],1)
            r=np.asscalar(np.dot(vect_mask,sliced_img))
            # Ignore neg points because we will anyways threshold for point
            skeleton[h+2][w+2] = r
    return skeleton

