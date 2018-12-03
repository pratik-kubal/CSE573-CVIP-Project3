#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 00:37:05 2018

@author: pratik
"""
import numpy as np;
import cv2

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


class hough:
    import numpy as np;
    import cv2
    
    def __init__(self,image,noiseReduction=True,kernelSize=(7,7)):
        self.image = image.copy()
        if(noiseReduction):
            self.image = cv2.GaussianBlur(self.image,kernelSize,0)
        self.rho = np.min(self.image.shape)
        self.theta=180
        self.accumulator=np.zeros((self.rho,self.theta))
        mask_x = [[1,0,-1],
          [2,0,-2],
          [1,0,-1]]
        self.mask_x = np.array(mask_x)
        mask_y = [[-1,-2,-1],
                  [0,0,0],
                  [1,2,1]]
        self.mask_y = np.array(mask_y)
    
    @staticmethod
    def __threshold(image,thresholdVal):
        for h in range(image.shape[0]):
            for w in range(image.shape[1]):
                if(image[h][w] > thresholdVal[0] and image[h][w] <= thresholdVal[1]):
                    image[h][w] = 1
                else:
                    image[h][w] = 0
        return np.array(image)
    
    def sobel(self,magnitudeThreshold,angleThreshold,write=False):
        masks = [self.mask_x,self.mask_y]
        edges=[]
        for mask in masks:
            mask = np.transpose(mask)
            space = (int)(mask.shape[0]/2)
            pad_image = np.pad(self.image,pad_width=space,mode='edge')
            vec_mask = mask.reshape(1,mask.shape[0]*mask.shape[1])
            sobelImage=[[0 for i in range(0,np.shape(self.image)[1])] for i in range(0,np.shape(self.image)[0])]
            for h in range(np.shape(pad_image)[0]-(space+2)):
                for w in range((np.shape(pad_image)[1])-(space+2)):
                    sliced_img = pad_image[h:h+mask.shape[0],w:w+mask.shape[1]].reshape(mask.shape[0]*mask.shape[1],1)
                    sobelImage[h][w]=np.asscalar(np.dot(vec_mask,sliced_img))
            sobelImage = np.abs(sobelImage)/np.max(np.abs(sobelImage))
            edges.append(sobelImage)
        cv2.imwrite('./temp/edge_x.jpg',np.multiply(edges[0],255))
        cv2.imwrite('./temp/edge_y.jpg',np.multiply(edges[1],255))
        edge_magnitude = np.sqrt(edges[0] ** 2 + edges[1] ** 2)
        edge_magnitude /= np.max(edge_magnitude)
        self.edge_magnitude = np.multiply(edge_magnitude,255)
        self.edge_magnitude_bin= self.__threshold(image=self.edge_magnitude.copy(),thresholdVal=magnitudeThreshold)
        cv2.imwrite('./temp/edge_mag.jpg',np.multiply(edge_magnitude,255))
        edge_direction = np.arctan(edges[1] / (edges[0] + 1e-3))
        edge_direction = edge_direction * 180. / np.pi
        #edge_direction /= np.max(edge_direction)
        self.sobelImage = self.__threshold(image=edge_direction.copy(),thresholdVal=angleThreshold)
        self.edge_direction = np.zeros(edge_direction.shape)
        for h in range(self.sobelImage.shape[0]):
            for w in range(self.sobelImage.shape[1]):
                if(self.edge_magnitude_bin[h][w]==0):
                    self.sobelImage[h,w] = 0
        for h in range(self.sobelImage.shape[0]):
            for w in range(self.sobelImage.shape[1]):
                if(self.sobelImage[h,w] == 1):
                    self.edge_direction[h,w]=edge_direction[h,w]
        cv2.imwrite('./temp/sobel_angle_thresholded.jpg',np.multiply(self.sobelImage,255))
        #self.sobelImage = np.multiply(self.sobelImage,255)
        self.sobelImage = np.array(self.sobelImage)
        return self.sobelImage,self.edge_direction
    
    def transformLines(self,angle_range,num_angles,write=False):
        # https://arxiv.org/pdf/1510.04863.pdf
        # Calculates Gradient of the image
        self.accumulator=np.zeros(((int)(np.ceil(np.sqrt(np.square(self.sobelImage.shape[0])+np.square(self.sobelImage.shape[1])))*2),np.abs(angle_range[0])+np.abs(angle_range[1])*2+2))
#        a = int(np.ceil(np.max(self.edge_direction)-np.min(self.edge_direction)))+1
#        self.accumulator=np.zeros((int(np.ceil(np.sqrt(np.square(self.sobelImage.shape[0])+np.square(self.sobelImage.shape[1])))),a))
#        for x in range(self.sobelImage.shape[1]):
#            for y in range(self.sobelImage.shape[0]):
#                if(self.sobelImage[y,x] == 1):
#                    angle = np.deg2rad(self.edge_direction[y,x])
#                    rhoVal = np.add(np.multiply(x,np.cos((angle))),np.multiply(y,np.sin((angle))))
#                    #print(int(np.round(rhoVal)),int(np.round(np.rad2deg(angle))))
#                    self.accumulator[int(np.round(rhoVal)),int(np.round(self.edge_direction[y,x]))]+=self.edge_magnitude[y,x]*255
#                    #self.accumulator[int(np.round(rhoVal+self.accumulator.shape[0]/2)),int(self.accumulator.shape[1]/2 + angle)]+=1
#        # Refine Accumulator
        for x in range(self.sobelImage.shape[1]):
            for y in range(self.sobelImage.shape[0]):
                if(self.sobelImage[y,x] == 1):
                    for angle in np.arange(angle_range[0],angle_range[1]):
                        rhoVal = np.add(np.multiply(x,np.cos((np.deg2rad(angle)))),np.multiply(y,np.sin((np.deg2rad(angle)))))
                        #print(rhoVal)
                        self.accumulator[int(self.accumulator.shape[0]/2+np.round(rhoVal)),int(self.accumulator.shape[1]/2+np.round(angle))]+=self.edge_magnitude[y,x]*255
        cv2.imwrite('./temp/accumulato2r.jpg',np.multiply(self.accumulator/np.max(self.accumulator),255))
        return self.accumulator
    
    def accioLocalMax(self,thresholdVal,kernelSize=3):
        self.houghPoints=[]
        bin_accumulator = self.__threshold(self.accumulator.copy(),[thresholdVal,np.max(self.accumulator.copy())+1])
        for h in range(bin_accumulator.shape[0]):
            for w in range(bin_accumulator.shape[1]):
                if(h%kernelSize==0 and w%kernelSize==0):
                    if((bin_accumulator[h:h+kernelSize,w:w+kernelSize]==1).any()):
                        #binFrame = bin_accumulator[h:h+kernelSize,w:w+kernelSize]
                        localMax = np.unravel_index(self.accumulator[h:h+kernelSize,w:w+kernelSize].argmax(),self.accumulator[h:h+kernelSize,w:w+kernelSize].shape)
                        self.houghPoints.append((h+localMax[0],w+localMax[1]))
        return self.houghPoints
    
    def plotHoughLines(self,image,accuracy=0):
        houghImage = image.copy()
        for i,(rho,theta) in enumerate(self.houghPoints):
            if(rho < self.accumulator.shape[0]/2):
                rho = self.accumulator.shape[0]/2-rho
                rho = rho*-1
            if(rho > self.accumulator.shape[0]/2):
                rho = rho - self.accumulator.shape[0]/2
            if(theta < self.accumulator.shape[1]/2):
                theta = self.accumulator.shape[1]/2 - theta
                theta = (theta*-1)
            if(theta >= self.accumulator.shape[1]/2):
                theta = theta - self.accumulator.shape[1]/2
            self.houghPoints[i]=[rho,theta]
        if(accuracy>0):
            # Sanity Check
            houghPoints = np.array(self.houghPoints)
            median = np.median(houghPoints[0:,1:])
            for rho,theta in houghPoints:
                if(np.allclose(np.abs(theta),np.abs(median),atol=median,rtol=accuracy)):
                    a = np.cos(np.deg2rad(theta))
                    b = np.sin(np.deg2rad(theta))
                    x0 = a*(rho)
                    y0 = b*(rho)
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 - 1000*(a))
                    cv2.line(houghImage,(x1,y1),(x2,y2),(255,0,0),2)
        else:
            for rho,theta in self.houghPoints:
                a = np.cos(np.deg2rad(theta))
                b = np.sin(np.deg2rad(theta))
                x0 = a*(rho)
                y0 = b*(rho)
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                cv2.line(houghImage,(x1,y1),(x2,y2),(255,0,0),3)
        cv2.imwrite('./temp/houghOt.jpg',houghImage)
        
    def line45refine(self,magnitudeThreshold):
        from functions import dilation
        mask=[[2,-1,-1],
        [-1,2,-1],
        [-1,-1,2]]
        mask = np.array(mask)
        space = (int)(mask.shape[0]/2)
        pad_image = np.pad(self.image,pad_width=space,mode='edge')
        vec_mask = mask.reshape(1,mask.shape[0]*mask.shape[1])
        linesImage=[[0 for i in range(0,np.shape(self.image)[1])] for i in range(0,np.shape(self.image)[0])]
        for h in range(np.shape(pad_image)[0]-(space+2)):
            for w in range((np.shape(pad_image)[1])-(space+2)):
                sliced_img = pad_image[h:h+mask.shape[0],w:w+mask.shape[1]].reshape(mask.shape[0]*mask.shape[1],1)
                linesImage[h][w]=np.asscalar(np.dot(vec_mask,sliced_img))
        linesImage = np.abs(linesImage)/np.max(np.abs(linesImage))
        linesImage = np.multiply(linesImage,255)
        self.linesImage = self.__threshold(image=linesImage.copy(),thresholdVal=magnitudeThreshold)
        struc_elem = [[1,0,1],
              [1,0,1],
              [1,0,1]]
        struc_elem=np.asarray(struc_elem)
        dilation_img = dilation(structuring_element=struc_elem,img=self.linesImage)
        #erosion_img = erosion(structuring_element=struc_elem2,img=erosion_img)
        cv2.imwrite('./temp/dilated45.jpg',np.multiply(dilation_img,255))
        #self.dilation_img=dilation_img
        #self.sobelImage=self.dilation_img
        self.sobelImage = self.linesImage
        cv2.imwrite('./temp/lines_45.jpg',np.multiply(self.linesImage,255))
        return self.sobelImage
    
    def boundaryExtraction(self):
        from functions import erosion
        struc_elem = [[0,1,1],
              [1,0,1],
              [1,1,0]]
        #struc_elem2 = [[0,0,0],
        #      [1,0,1],
        #      [0,0,0]]
        struc_elem=np.asarray(struc_elem)
        #struc_elem2=np.asarray(struc_elem2)
        erosion_img = erosion(structuring_element=struc_elem,img=self.sobelImage)
        #erosion_img = erosion(structuring_element=struc_elem2,img=erosion_img)
        cv2.imwrite('./temp/erodedCircles.jpg',np.multiply(erosion_img,255))
        self.erosion_img=erosion_img
        self.sobelImage=self.erosion_img
        return self.erosion_img
    
    def transformCircle(self,r,angle_range):
        self.r = r
        maxA=0
        maxB=0
        for theta in np.arange(angle_range[0],angle_range[1]):
            a1=0-np.multiply(self.r,np.cos(np.deg2rad(theta)))
            b1=0-np.multiply(self.r,np.sin(np.deg2rad(theta)))
            a2=0-np.multiply(self.r,np.cos(np.deg2rad(theta)))
            b2=self.sobelImage.shape[0]-np.multiply(self.r,np.sin(np.deg2rad(theta)))
            a3=self.sobelImage.shape[1]-np.multiply(self.r,np.cos(np.deg2rad(theta)))
            b3=0-np.multiply(r,np.sin(np.deg2rad(theta)))
            a4=self.sobelImage.shape[1]-np.multiply(self.r,np.cos(np.deg2rad(theta)))
            b4=self.sobelImage.shape[0]-np.multiply(self.r,np.sin(np.deg2rad(theta)))
            if(maxA<np.max([a1,a2,a3,a4])):
                maxA=np.max([a1,a2,a3,a4])
            if(maxB<np.max([b1,b2,b3,b4])):
                maxB=np.max([b1,b2,b3,b4])
        #print(maxB,maxA)
        self.accumulator=np.zeros((int(np.ceil(maxB*2+2*self.r)+1),int(np.ceil(maxA*2+2*self.r)+1)))
        for x in range(self.sobelImage.shape[1]):
            for y in range(self.sobelImage.shape[0]):
                if(self.sobelImage[y,x]==1):
                    for theta in np.arange(angle_range[0],angle_range[1]):
                        a=x-np.multiply(self.r,np.cos(np.deg2rad(theta)))
                        b=y-np.multiply(self.r,np.sin(np.deg2rad(theta)))
                        self.accumulator[int(self.accumulator.shape[0]/2+np.round(b)),int(self.accumulator.shape[1]/2+np.round(a))] +=self.edge_magnitude[y,x]*255
        cv2.imwrite('./temp/circleAccumulator.jpg',np.multiply(self.accumulator/np.max(self.accumulator),255))
        return self.accumulator
    
    def plotHoughCircles(self,image):
        houghImage = image.copy()
        for b,a in self.houghPoints:
            if(b < self.accumulator.shape[0]/2):
                b = self.accumulator.shape[0]/2-b
                b = b*-1
            if(b > self.accumulator.shape[0]/2):
                b = b - self.accumulator.shape[0]/2
            if(a < self.accumulator.shape[1]/2):
                a = self.accumulator.shape[1]/2 - a
                a = (a*-1)
            if(a >= self.accumulator.shape[1]/2):
                a = a - self.accumulator.shape[1]/2
            cv2.circle(houghImage,(int(np.round(a)),int(np.round(b))),self.r,(255,0,0),3,8,0)
        cv2.imwrite('./temp/houghCircle.jpg',houghImage)
    

class histogram:
    import numpy as np;
    import cv2
    from matplotlib import pyplot as plt
    
    def __init__(self,image,tileSize=[8,8],NoiseReduction=False,kernelSize=(7,7)):
        self.image=image.copy()
        if(NoiseReduction):
            self.image = cv2.GaussianBlur(self.image,kernelSize,0)
        self.tileSize=tileSize
        
    @staticmethod
    def __threshold(image,thresholdVal,binarize=True):
        for h in range(image.shape[0]):
            for w in range(image.shape[1]):
                if(image[h][w] > thresholdVal[0] and image[h][w] <= thresholdVal[1]):
                    if(binarize):
                        image[h][w] = 1
                else:
                    image[h][w] = 0
        return np.array(image)
        
    def adaptiveHist(self):
        image = self.image.copy()
        for h in range(image.shape[0]):
            for w in range(image.shape[1]):
                if(h%self.tileSize[0]==0 and w%self.tileSize[1]==0):
                        #image[h1:h1+self.tileSize[0],w1:w1+self.tileSize[1]]=self.optimalThresholding(image=image[h1:h1+self.tileSize[0],w1:w1+self.tileSize[1]])
                        image[h:h+self.tileSize[0],w:w+self.tileSize[1]]=self.optimalThresholding(image=image[h:h+self.tileSize[0],w:w+self.tileSize[1]])
        self.adaptiveHistImage=image
        cv2.imwrite('./2/adaptiveThresholding.jpg',np.multiply(self.adaptiveHistImage,255))
        return self.adaptiveHistImage
    
    def __cdfNorm(self):
        image = self.image.copy()
        vect_image = image.reshape((image.shape[0]*image.shape[1],1))
        vect_image = np.array(vect_image,dtype='float32')
        # Histogram Equilization
        a,b = np.unique(vect_image,return_counts=True)
        L = 256
        p = np.divide(b,np.sum(b))
        # https://www.math.uci.edu/icamp/courses/math77c/demos/hist_eq.pdf
        cdf = np.floor(np.multiply(L-1,np.cumsum(p)))
        cdf_dict = {}
        for i,item in enumerate(list(a)):
            cdf_dict.update({item:cdf[i]})
        for h in range(image.shape[0]):
            for w in range(image.shape[1]):
                image[h,w] = cdf_dict.get(image[h,w])
        self.histNormalized = image
        return np.array(self.histNormalized)

    def optimalThresholding(self,image=None):
        #self.__cdfNorm()
        if(image is not None):
            self.histNormalized=image.copy()
        else:
            self.histNormalized=self.image.copy()
        vect_image = self.histNormalized.reshape((self.histNormalized.shape[0]*self.histNormalized.shape[1],1))
        vect_image = np.array(vect_image,dtype='float32')
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,100, 1.0)
        
        _,_,centers = cv2.kmeans(vect_image,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        
        thresholdVal = np.mean(centers)
        
        self.optimalThresImage = self.__threshold(image=self.histNormalized.copy(),thresholdVal=[thresholdVal,255])
        cv2.imwrite('./2/OptimalThresholding.jpg',np.multiply(self.optimalThresImage,255))
        return self.optimalThresImage
    
    def specificThresholding(self,thresholdVal,binarize=False):
        self.specificThresImage = self.__threshold(image=self.image.copy(),thresholdVal=thresholdVal,binarize=binarize)
        cv2.imwrite('./2/manualThresholding.jpg',np.multiply(self.specificThresImage,255))
        return self.specificThresImage
        
class pointDetection:
    import numpy as np;
    import cv2
    def __init__(self,image,kernelSize=(7,7)):
        self.image=image.copy()
        #self.image=cv2.GaussianBlur(self.image,kernelSize,0)
        self.image = np.array(self.image,dtype='float64')
        mask = [[-1,-1,-1],
        [-1,8,-1],
        [-1,-1,-1]]
        self.mask = np.array(mask)
        
    @staticmethod
    def __threshold(image,thresholdVal,binarize=True):
        for h in range(image.shape[0]):
            for w in range(image.shape[1]):
                if(image[h][w] > thresholdVal[0] and image[h][w] <= thresholdVal[1]):
                    if(binarize):
                        image[h][w] = 1
                else:
                    image[h][w] = 0
        return np.array(image)
    
    def pointDet(self,thresholdVal):
        from functions import normImage
        img = self.image.copy()
        vect_mask = self.mask.reshape(1,self.mask.shape[0]*self.mask.shape[1])
        pointDetectedImage=[[0 for i in range(0,np.shape(img)[1])] for i in range(0,np.shape(img)[0])]
        for h in range(np.shape(img)[0]-self.mask.shape[0]):
            for w in range((np.shape(img)[1])-self.mask.shape[1]):
                sliced_img = img[h:h+self.mask.shape[0],w:w+self.mask.shape[1]].reshape(self.mask.shape[0]*self.mask.shape[1],1)
                r=np.asscalar(np.dot(vect_mask,sliced_img))
                pointDetectedImage[h][w] = np.abs(r)
        pointDetectedImage = np.array(pointDetectedImage)
        print(np.max(pointDetectedImage))
        self.pointDetectedImage_bin = self.__threshold(image=pointDetectedImage.copy(),thresholdVal=thresholdVal)
        #pointDetectedImage = np.multiply(np.array(normImage(pointDetectedImage)),255)
        #pointDetectedImage = np.multiply(pointDetectedImage,255)
        cv2.imwrite('./temp/pointDetRaw.jpg',np.multiply(self.pointDetectedImage_bin,255))
        #self.pointDetectedImage = pointDetectedImage
        #cv2.imwrite('./temp/pointDetBin.jpg',np.multiply(self.pointDetectedImage_bin,255))
        return pointDetectedImage
        

        
         