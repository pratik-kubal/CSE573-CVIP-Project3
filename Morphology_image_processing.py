UBIT = 'pkubal';
import numpy as np;
import cv2
np.random.seed(sum([ord(c) for c in UBIT]))

from functions import dilation,erosion

bin_image = cv2.imread("./")
bin_image = cv2.imread("./original_imgs/noise.jpg",0)

bin_image = np.asarray(np.divide(bin_image,255),dtype='int8')
# [0, 1, 0, 1, 1, 1, 0, 1, 0]
struc_elem = [[1,1,1],
              [1,1,1],
              [1,1,1]]
struc_elem=np.asarray(struc_elem)

erosion_img = erosion(structuring_element=struc_elem,img=bin_image)

dil_img = dilation(structuring_element=struc_elem,img=erosion_img)

cv2.imwrite('./1/res_noise1.jpg',np.multiply(dil_img,255))

''' The above method cannot clear the noise inside the object. For the square 
at the top left still has noise. However it has denoised it from outside.'''

open_img = dilation(structuring_element=struc_elem,
                    img=erosion(structuring_element=struc_elem,
                                img=bin_image.copy()))

close_img = erosion(structuring_element=struc_elem,
                    img=dilation(structuring_element=struc_elem,
                                 img=open_img))

cv2.imwrite('./1/res_noise2.jpg',np.multiply(close_img,255))


# Boundary Detectiom

left = erosion(structuring_element=struc_elem,img=dil_img.copy())
boundary_img = np.subtract(dil_img,left)

cv2.imwrite('./1/res_bound1.jpg',np.multiply(boundary_img,255))


left = erosion(structuring_element=struc_elem,img=close_img.copy())
boundary_img = np.subtract(close_img,left)

cv2.imwrite('./1/res_bound2.jpg',np.multiply(boundary_img,255))