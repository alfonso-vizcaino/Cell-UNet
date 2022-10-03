#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:10:29 2021

In this code we create a file that stores bounding boxes coordinates for every cell in a image file

@author: alfonso
"""

import os
import cv2
from os import walk
import matplotlib.pyplot as plt
import numpy as np
import albumentations as A

""" 
    To generate BoundingBoxes information for different datasets change 'src_path' and 'gt_path' acordingly
"""
src_path = '../../data/test/src/'
gt_path = '../../data/test/gt/'

""" Location where the BoundingBoxes information will be saved """
bb_path = gt_path.replace("/gt/","/bb/")

_, _, filenames = next(walk(gt_path))

if not os.path.exists(bb_path):
    os.makedirs(bb_path)

sizes = []


def proces_file(img, img_original, filename):
    

    withRect = img_original.copy()
    (imgWidth, imgHeight, _) = withRect.shape
    if imgWidth != 512 or imgHeight != 512:
        img_prepr = A.Resize(512, 512)(image=img) #Make sure we only have 512 by 512 images
        img_prepr = img_prepr['image']
        img = img_prepr
        
        img_prepr = A.Resize(512, 512)(image=withRect)
        img_prepr = img_prepr['image']
        withRect = img_prepr

    count, labels = cv2.connectedComponents(img)

    fn = filename.replace('.png', '.txt')

    f = open(bb_path+fn, "w")
    keys = []

    # print(np.unique(img))
    for c in range(count):
        if c == 0:
            continue  # skip background
        cmp = labels == c
        cmp = np.uint8(cmp*1)

        contours, _ = cv2.findContours(
            cmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        area = cv2.contourArea(contours[0])
        rect = cv2.boundingRect(contours[0])

        left, top, width, height = rect


        right = left + width
        bottom = top + height
        centerX = left+(right - left)/2
        centerY = top +(bottom - top)/2
        
        if width * height < 20:
            print(f'WARNING: skipping bad marking found ({left}, {top}), ({right}, {bottom}) in file {filename}')
            cv2.rectangle(withRect, (left, top), (right, bottom), (255, 0, 0), 5)
            plt.imshow(withRect)
            plt.title('bad marking found')
            plt.show()
            continue
        
        global sizes

        sizes.append((area, width, height))
            

        key = f'0 {centerX/imgWidth:8f} {centerY/imgHeight:8f} {width/imgWidth:8f} {height/imgHeight:8f}'
        cv2.rectangle(withRect, (left, top), (right, bottom), (255, 0, 0), 1)
        if key in keys:
            print(f'WARNING: duplicated key = {key} in file {filename}')
            cv2.rectangle(withRect, (left, top),
                          (right, bottom), (0, 255, 0), 1)
        else:
            keys.append(key)
            f.write(f'{key}\n')

    plt.imshow(withRect)
    plt.title('bb')
    plt.show()

    f.close()


def parse(filename):
    f = open(bb_path+filename, "r")
    lines = f.readlines()

    fn = filename.replace('.txt', '.png')
    img = cv2.imread(src_path+fn)

    plt.imshow(img)
    plt.title(filename)
    plt.show()

    (w, h, c) = img.shape

    # Strips the newline character
    for line in lines:
        d = line.split()
        cx, cy, cw, ch = (float)(d[1]), (float)(
            d[2]), (float)(d[3]), (float)(d[4])
        left = (int)(w*(cx - cw/2))
        right = (int)(w*(cx + cw/2))
        top = (int)(h*(cy - ch/2))
        bottom = (int)(h*(cy + ch/2))

        cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 1)

    plt.imshow(img)
    plt.show()


for filename in filenames:

    # if (filename == "CONTROL_3_7_CA1-1_4.png"):

    if filename.endswith('.txt'):
        #parse(filename)
        #break
        continue
    
    # if (filename != "control2_9_DG-1_1.png"):
    #     continue

    #fig = plt.figure()

    # print(filename)
    img_original = cv2.imread(src_path+filename)

    gt_file_name = filename#.replace('.png', '_label_3d.png')
    img = cv2.imread(gt_path+gt_file_name, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f'skiping {gt_path+gt_file_name}')
        continue

    img[img < 60] = 0
    idx = (img > 59) * (img < 150)
    img[np.where(idx)] = 0
    img[img > 149] = 1
    colors = np.unique(img)
    # print(colors)
    
    plt.imshow(img_original)
    plt.title(filename)
    plt.show()

    plt.imshow(img, cmap='gray')
    plt.title(gt_file_name)
    plt.show()

    proces_file(img, img_original, filename)
    
    #break


sizes = np.array(sizes)
means = np.mean(sizes, axis=0)
stdev = np.std(sizes, axis=0)
print(means)
print(stdev)

