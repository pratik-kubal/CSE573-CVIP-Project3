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
from functions import houghTransform

hough_image = cv2.imread("./original_imgs/hough.jpg",0)
del hough_col
hough_col = cv2.imread("./original_imgs/hough.jpg")

blue_lines = houghTransform(hough_image)
sobel_image,edge_directions = blue_lines.sobel(magnitudeThreshold=[20,255],angleThreshold=[30,50],write=True)
angle_range=[-90,270]
num_angles=360
accumulator = blue_lines.transform(angle_range=angle_range,num_angles=num_angles)
points = blue_lines.accioLocalMax(thresholdVal =np.max(accumulator)*0.8 ,kernelSize = 10)

red_lines = houghTransform(hough_image)
sobel_image,edge_directions = red_lines.sobel(magnitudeThreshold=[20,110],angleThreshold=[70,90],write=True)
angle_range=[40,270]
accumulator = red_lines.transform(angle_range=angle_range,num_angles=num_angles)
points = red_lines.accioLocalMax(thresholdVal =np.max(accumulator)*0.5 ,kernelSize = 60)
points
red_lines.plotHoughLines(image=hough_col)