#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 21:28:54 2018

@author: pratik
"""
UBIT = 'pkubal';
import numpy as np;
import cv2
np.random.seed(sum([ord(c) for c in UBIT]))

class houghTransform:
    def __init__(self,image):
        self.image = image.copy()
        self.rho = np.min(self.image.shape)
        self.theta=180
        self.accumulator=np.zeros((self.rho,self.theta))
    
    @staticmethod
    def __threshold(image,thresholdVal):
        for h in range(image.shape[0]):
            for w in range(image.shape[1]):
                if(image[h][w] > thresholdVal):
                    image[h][w] = 1
                else:
                    image[h][w] = 0
        return np.array(image,dtype='int8')
    
    def sobel(self,mask,thresholdVal,write=False):
        mask = np.transpose(mask)
        space = (int)(mask.shape[0]/2)
        pad_image = np.pad(self.image,pad_width=space,mode='edge')
        vec_mask = mask.reshape(1,mask.shape[0]*mask.shape[1])
        self.sobelImage=[[0 for i in range(0,np.shape(self.image)[1])] for i in range(0,np.shape(self.image)[0])]
        for h in range(np.shape(pad_image)[0]-(space+2)):
            for w in range((np.shape(pad_image)[1])-(space+2)):
                sliced_img = pad_image[h:h+mask.shape[0],w:w+mask.shape[1]].reshape(mask.shape[0]*mask.shape[1],1)
                self.sobelImage[h][w]=np.asscalar(np.dot(vec_mask,sliced_img))
        self.sobelImage = np.abs(self.sobelImage)/np.max(np.abs(self.sobelImage))
        #self.sobelImage = np.multiply(self.sobelImage,255)
        self.sobelImage = np.array(self.sobelImage)
        if(thresholdVal>0):
            self.sobelImage = self.__threshold(image=self.sobelImage,thresholdVal=thresholdVal/255)
        if(write):
            cv2.imwrite('./temp/hough.jpg',np.multiply(self.sobelImage,255))
            return None
        else:
            return self.sobelImage
    
    def transform(self):
        a
    
hough_image = cv2.imread("./original_imgs/hough.jpg",0)
red_lines = houghTransform(hough_image)
mask_x=[[1,2,1],
       [0,0,0],
       [-1,-2,-1]]
mask = np.array(mask_x)
test = red_lines.sobel(mask=mask,thresholdVal=100,write=False)

#http://collaboration.cmc.ec.gc.ca/science/rpn/biblio/ddj/Website/articles/CUJ/2003/0312/0312queisser/cuj0312queisser.htm
angle_range=[0,180]
num_angles=180
accumulator=np.zeros(((int)(np.ceil(np.sqrt(np.square(image.shape[0])+np.square(image.shape[1])))*2),np.abs(angle_range[0])+np.abs(angle_range[1])*2+2))

for h in range(image.shape[0]):
    for w in range(image.shape[1]):
        if(image[h,w] == 1):
            for angle in np.linspace(angle_range[0],angle_range[1],num=num_angles):
                rhoVal = np.add(np.multiply(h,np.cos(np.deg2rad(angle))),np.multiply(w,np.sin(np.deg2rad(angle))))
                if(rhoVal<0):
                    if(angle < 0):
                        accumulator[(int)(accumulator.shape[0]/2 - np.round(rhoVal*-1)),int(accumulator.shape[1]/2 - np.round(angle*-1))]+=1
                    else:
                        accumulator[(int)(accumulator.shape[0]/2 - np.round(rhoVal*-1)),int(np.round(angle)+accumulator.shape[1]/2)]+=1
                else:
                    if(angle < 0):
                        accumulator[(int)(np.round(rhoVal)+accumulator.shape[0]/2),int(accumulator.shape[1]/2 - np.round(angle*-1))]+=1
                    else:
                        accumulator[(int)(np.round(rhoVal)+accumulator.shape[0]/2),int(np.round(angle)+accumulator.shape[1]/2)]+=1
                    

cv2.imwrite('./temp/accumulator.jpg',accumulator)    
        