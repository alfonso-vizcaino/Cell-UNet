#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: alfonso
"""

import sys
import logging
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A

from os import listdir
from glob import glob
from enum import Enum
from os.path import splitext
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


sys.path += ['./']
from ice import enhance_img


TrainOnEnum = Enum('TrainOnEnum', 'CellBody CellBorder Cell')


class NeuronDataset(Dataset):

    def __init__(self, imgs_dir='../../data/train/src/',
                 masks_dir='../../dat/train/gt/',
                 mask_suffix='',
                 trainOn=TrainOnEnum.CellBorder,
                 augment=True,
                 debug=False,
                 enhance=True):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.mask_suffix = mask_suffix
        self.augment = augment
        self.enhance = enhance

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        self.total = len(self.ids)

        if self.augment:
            self.augmentations = A.Sequential([A.Compose({
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Transpose(),
                A.Rotate(limit=270),
                A.RandomScale(scale_limit=0.2),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.0, rotate_limit=0, p=0.8),
                A.RandomBrightness(limit=0.12),
                A.RandomContrast(limit=0.12),
                A.HueSaturationValue(),
            }),
                A.Resize(512, 512, always_apply=True, p=1), #Make sure we only have 512 by 512 images
            ])
        else:
            self.augmentations = A.Compose({
                A.Resize(512, 512, always_apply=True, p=1), #Make sure we only have 512 by 512 images
                #A.Normalize(mean=(0.1240, 0.1194, 0.1280),std=(0.0248, 0.0262, 0.0205))
            })
        self.trainOn = trainOn
        self.debug = debug
        logging.info(f'Creating dataset with {self.total} examples')

    def __len__(self):
        l = (self.total*8) if self.augment else self.total  # (self.total *5)
        return l  # len(self.ids)

    def preprocess(self, img, mask):

        if self.debug:
            plt.imshow(img)
            plt.show()

        transformed = self.augmentations(image=img, mask=mask)

        img = transformed["image"]
        mask = transformed["mask"]

        channels = 3 if self.trainOn == TrainOnEnum.Cell else 1
        if self.enhance:
            img = enhance_img(img, channels, self.debug)
        else:
            if channels == 1:
                img = img[:, :, 1]
            else:
                img = img  # [:,:,1] #fixed for now

        if self.trainOn == TrainOnEnum.CellBorder:
            mask[mask > 149] = 0  # Remove cell body information
            # Keep cell border information only
            idx = (mask > 44) * (mask < 150)
            mask[np.where(idx)] = 1
        elif self.trainOn == TrainOnEnum.CellBody:
            mask[mask < 150] = 0   # Remove cell border information
            mask[mask > 149] = 1  # Keep cell body information only
        elif self.trainOn == TrainOnEnum.Cell:
            mask[mask < 45] = 0
            idx = (mask > 44) * (mask < 150)  # cell border information
            mask[np.where(idx)] = 0
            mask[mask > 149] = 1  # cell body information

        if self.debug:
            plt.imshow(img, cmap='gray')
            plt.show()

            plt.imshow(mask, cmap='gray')
            plt.show()

        # Transform to tensor
        img = TF.to_tensor(img)

        return img, mask

    def __getitem__(self, i):
        idx = self.ids[i % self.total]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')

        if self.debug:
            logging.debug(f'Processing file {mask_file}')
            #print('mask_file', mask_file)

        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'

        img_org = cv.imread(img_file[0])
        img_org = cv.cvtColor(img_org, cv.COLOR_BGR2RGB)

        mask = cv.imread(mask_file[0], cv.IMREAD_GRAYSCALE)

        img, mask = self.preprocess(img_org.copy(), mask)

        ic, iw, ih = img.shape
        # mc,
        mw, mh = mask.shape

        assert (iw, ih) == (512, 512), \
            f'Image for file {idx} should be of size (512, 512), but img is {(iw, ih)}'

        assert (iw, ih) == (mw, mh), \
            f'Image and mask for file {idx} should be the same size, but src is {(iw, ih)} and mask is {(mw, mh)}'

        return {
            'image': img,
            'mask': mask
        }
