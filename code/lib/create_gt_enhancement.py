#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:10:29 2021

In this code we perform a Ground Truth Enhancement - GTE - process on based on ground truth images


@author: alfonso
"""

import os
import cv2
from os import walk
import matplotlib.pyplot as plt
import numpy as np

""" 
    To generate GTE images for different datasets change 'src_path' and 'gt_path' acordingly
"""
src_path = '../../data/train/src/'
gt_path = '../../data/train/gt/'

""" Location where the new GTE images will be saved """
gt_enhancement = src_path.replace("/src/","/gte/")

_, _, filenames = next(walk(src_path))

if not os.path.exists(gt_enhancement):
    os.makedirs(gt_enhancement)

"""
   To generate GTE images for a different disk size change 'diameter_size' to
   one of the following values: 
   5, 10, 15
"""

diameter_size = 5

def proces_file(img, img_original):
    
    count, labels = cv2.connectedComponents(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    #print(np.unique(img))
    for c in range(count):
        if c ==0:
            continue #skip background
        cmp = labels==c
        cmp = np.uint8(cmp*1)
        
        contours, _ = cv2.findContours(cmp,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        area = cv2.contourArea(contours[0])
        
        # global cell_area
        # global cell_count
        
        # cell_count = cell_count + 1
        # cell_area = cell_area + area
        
        M = cv2.moments(contours[0]);
        d = float(M['m00'])
        x = int(M['m10']/d if d !=0 else 1)
        y = int(M['m01']/d if d !=0 else 1);
        
        # Mark the center
        diameter = diameter_size if area >190 else 3
        
        cv2.circle(img_original,(x,y), diameter, (0,182,0), -1);
        cv2.circle(img,(x,y), diameter, (255,0,0), -1);
        
        # print(f"count={c}")
        
    plt.imshow(img_original)
    plt.title('marked')
    plt.show()
    
    
    # print(np.unique(img))
    # plt.imshow(img, cmap='gray')
    # plt.title('eroded')
    # plt.show()
    return img, img_original

count = 0
for filename in filenames:
    
    # if (filename != "HI_AL_18_7_DG1_1.png" and filename != "CONTROL_3_7_DG-2_1.png"):
    #      continue;
        
    fig = plt.figure()
    
    #print(filename)
    img_original = cv2.imread(gt_path+filename)
    
    gt_file_name = filename#.replace('.png', '_label_3d.png')
    img = cv2.imread(gt_path+gt_file_name, cv2.IMREAD_GRAYSCALE)
    img[img<60] = 0
    idx = (img>59) * (img<150)
    img[np.where(idx)]=0
    img[img>149] = 1
    colors = np.unique(img)
    #print(colors)
    plt.imshow(img_original)
    plt.title(filename)
    plt.show()
    
    plt.imshow(img, cmap='gray')
    plt.title(gt_file_name)
    plt.show()
    
    w,h = img.shape;
    cleaned = np.zeros((w,h,3));
    cleaned[:,:,0]=img*255
    cleaned[:,:,1]=img*255
    cleaned[:,:,2]=img*255
    
    # un_proc = gt_file_name.replace('.png', f'_un_proc.png')
    # cv2.imwrite(gt_path.replace("/gt/","/gte/")+un_proc, cleaned)
    
    img2, marked = proces_file(img, cleaned)
    # print(np.unique(img2))
    plt.imshow(img2, cmap='gray')
    plt.title('processed')
    plt.show()
    
    
    cv2.imwrite(gt_enhancement+gt_file_name, img2)
    # cv2.imwrite(gt_path.replace("/gt/","/gte/")+gt_file_name.replace('.png', f'_s{diameter_size}_gt.png'), img2)
    
    # marked_name = gt_file_name.replace('.png', f'_s{diameter_size}_marked.png')
    # cv2.imwrite(gt_path.replace("/gt/","/gte/")+marked_name, marked)
    #break
    #
