#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

def deblur(image, debug=False):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    fm = cv.Laplacian(gray, cv.CV_64F).var()

    if fm < 900:
        if debug:
            plt.imshow(image)
            plt.title(f'Blurry {fm}')
            plt.show()

        sharpen = cv.filter2D(image, -1, sharpen_kernel)

        if debug:
            plt.imshow(sharpen)
            plt.title('Fixed Blurry')
            plt.show()

        image = sharpen
    elif debug:
        plt.imshow(image)
        plt.title(f'Not Blurry {fm}')
        plt.show()

    return image


def automatic_brightness_and_contrast(image, clip_hist_percent=1, channels=1, debug=False):

    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    # Calculate grayscale histogram
    hist = cv.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    img = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
    return img


def enhance_img(img, channels=1, debug=False):

    if debug:
        plt.imshow(img)
        plt.title("Augmented")
        plt.show()

    if (debug):
        print(img.shape)
        plt.imshow(img)
        plt.title('original')
        plt.show()

    imhsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)

    clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))

    imhsv[:, :, 2] = clahe.apply(imhsv[:, :, 2])

    if debug:
        plt.imshow(img)
        plt.title('before')
        plt.show()

    img = cv.cvtColor(imhsv, cv.COLOR_HSV2RGB)

    if debug:
        plt.imshow(img)
        plt.title('after')
        plt.show()

    img = automatic_brightness_and_contrast(
        img, channels=channels, debug=debug)
    img = deblur(img, debug=debug)

    return img
