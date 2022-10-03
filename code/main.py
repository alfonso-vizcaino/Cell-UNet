#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 09:49:56 2021

In this code we evaluate the performance of different Deep Learning models 
according to different scenarios: Unprocessed, ICE, GTE, ICE+GTE.

If you want to evaluate certain models only, update var 'models'.
In case you want to run validations agains a specific scenario, update var: 'scenarios'

Prediction results (images and metrics) are stored at: 
images -   /results/{scenario}/{modelName}/
metrics - /results/{scenario}/{mmodelName}_Results.txt

@author: alfonso
"""


import os
import sys
import cv2 as cv
import torch
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join

sys.path += ['./lib', './models']

from ice import enhance_img
from BoundingBox import BoundingBox


from cell_unet import CellUNet 
from unet import UNet
from atention_unet import UNet_Attention
from unetpp import Nested_UNet 

""" DL Models to be evalutated """
models = ['Cell_UNet']#, 'UNet', 'Att_UNet', 'UNetPP']

"""
    Scenarios to be evaluated:
    Scenario 1 - For evaluating models withOUT Ground Truth Enhancement (GTE) and withOUT Image Conditioning Enhancement (ICE) aka Unprocessed
    Scenario 2 - For evaluating models with ICE only
    Scenario 3 - For evaluating models with GTE only
    Scenario 4 - For evaluating models with GTE + ICE
"""
scenarios = [4]#1,2,3,4]

""" Dir location where results will be saved """
results_dir = '../results'

lbp_img = None
sat = None
and_img = None


def to_tensor(pic):
    if isinstance(pic, np.ndarray):
        # handle numpy array
        if pic.ndim == 2:
            pic = pic[:, :, None]

        img = torch.from_numpy(pic.transpose((2, 0, 1))).contiguous()
        # backward compatibility
        if isinstance(img, torch.ByteTensor):
            return img.to(dtype=torch.get_default_dtype()).div(255)
        else:
            return img

# calculate blend parameters

def get_predictions(cgnet_model, img, debug_params=False):
    model = cgnet_model

    # Put image data inside a tensor
    img_tensor = to_tensor(img)
    # if debug_params:
    #     print(img_tensor.shape)

    # Perform the test in CPU
    data = img_tensor.unsqueeze(0).to(
        device=torch.device('cpu'), dtype=torch.float32)

    # Call the model inference on the given image
    model.eval()
    predict = model(data)
    if cell_model.n_classes > 1:
        predict = torch.softmax(predict, 1)
    else:
        predict = torch.sigmoid(predict)
    if debug_params:
        if type(predict) == tuple:
            len(predict)

    return predict


def get_segmentation_results(predict, threshold=None, debug_params=False, border=True, multple_classes=False):

    image_type = 'borders' if border else 'bodies'

    # Get the prediction in np format since it is in tensor format
    # [batch_number, num_channels, width, height]
    if multple_classes == True:
        img_predictions = predict[0].cpu().detach().numpy()[1 if border else 2]

        background = predict[0].cpu().detach().numpy()[0]
        idx = img_predictions > background

        img_predictions_tmp = np.where(idx, img_predictions, 0)
    else:
        img_predictions_tmp = predict[0].cpu().detach().numpy()[0]

    if debug_params:
        # print(img_predictions_tmp.shape)
        # Display prediction results
        plt.imshow(img_predictions_tmp, cmap='gray')
        plt.title(f'Raw prediction {image_type}')
        plt.show()

    if threshold != None:
        img_predictions_tmp[img_predictions_tmp < threshold] = 0
        img_predictions_tmp[img_predictions_tmp >= threshold] = 255
        # Display prediction results
        if debug_params:
            plt.imshow(img_predictions_tmp, cmap='gray')
            plt.title(f'Raw prediction thresholded {image_type}')
            plt.show()

        img_predictions = img_predictions_tmp
    else:
        img_predictions_tmp[img_predictions_tmp > 0.5] = 255
        img_predictions = img_predictions_tmp

    img_result = img_predictions

    return img_result


def extract_points(img):

    count, labels = cv.connectedComponents(img)
    points = []
    for c in range(count):
        if c == 0:
            continue  # skip background
        cmp = labels == c
        cmp = np.uint8(cmp*1)

        contours, _ = cv.findContours(
            cmp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        area = cv.contourArea(contours[0])

        M = cv.moments(contours[0])
        d = float(M['m00'])
        x = int(M['m10']/d if d != 0 else 1)
        y = int(M['m01']/d if d != 0 else 1)

        points.append((x, y))

    return points


def compute_result(test_image, bounding_box, points, file_name=None, save_path=None, save_results=False):
    tp = 0
    fp = 0
    fn = 0

    for point in points:
        found = False
        for bb in bounding_box:
            if bb.unmatched and bb.contains(point[0], point[1]):
                tp += 1
                bb.unmatched = False
                found = True
                cv.rectangle(test_image, (bb.x1, bb.y1),
                             (bb.x2, bb.y2), (0, 255, 0), 1)
                break
        if not found:
            fp += 1
            cv.circle(test_image, (point[0], point[1]), 5, (0, 0, 255), -1)

    for bb in bounding_box:
        if bb.unmatched:
            fn += 1
            cv.rectangle(test_image, (bb.x1, bb.y1),
                         (bb.x2, bb.y2), (255, 0, 0), 1)
            #plt.imshow(test_image)
            #plt.show()

    plt.imshow(test_image)
    plt.show()

    if (save_results):
        tmp = cv.cvtColor(test_image, cv.COLOR_BGR2RGB)
        file_name = file_name.replace('.png', '_result.png')
        cv.imwrite(save_path+file_name, tmp)
        
    return tp, fp, fn


def compare_gt_vs_prediction(
        test_image='../data/test/src/control4_14_DG-4_5.png',
        bounding_box=[],
        debug_params=False,
        threshold_border=None,
        threshold_body=None,
        save_path=None,
        save_results=False):

    # Read test image
    img_original = cv.imread(test_image)
    file_name = os.path.basename(test_image)
    if (save_results):
        cv.imwrite(save_path+file_name, img_original)
    img_original = cv.cvtColor(img_original, code=cv.COLOR_BGR2RGB)
    plt.imshow(img_original)
    plt.title(file_name)
    plt.show()
    

    model = cell_model  # cgnet_model # cell_model

    multple_classes = cell_model.n_classes > 1
    # Preprocess image:
    # make sure its in [0..1] range and (512,512) size
    img_prepr = np.copy(img_original)

    width, height, channels = img_prepr.shape
    if width != 512 or height != 512:
        img_prepr = A.Resize(512, 512)(image=img_prepr)
        img_prepr = img_prepr['image']
        img_original = img_prepr

    # img_original[:,:,1]#
    img_gray = cv.cvtColor(img_original, cv.COLOR_RGB2GRAY)
    not_img = cv.bitwise_not(img_gray)

    w, h, c = img_original.shape
    tmp = img_original.copy()
    if (w != 512) or (h != 512):
        tmp = A.Resize(512, 512)(image=tmp, always_apply=True, p=1)
        tmp = tmp["image"]
    hsv = cv.cvtColor(tmp, cv.COLOR_RGB2HSV)
    hue = to_tensor(hsv[:, :, 0])
    global sat
    sat = to_tensor(hsv[:, :, 1])
    val = to_tensor(hsv[:, :, 2])

    if enhance:
        img_prepr = enhance_img(img_prepr, debug=debug_params, channels=3)
    else:
            img_prepr = img_prepr

    if debug_params:
        total = 0
        for i, p in enumerate(model.parameters()):
            parameters = p.numel() if p.requires_grad else 0
            total += parameters
            print(f'layer={i} parameters={parameters}')
        print(f'TOTAL PARAMETER COUNT={total}')
        print()
        # Display information in a nice format
        # summary(model, input_size=(3, 512, 512),device="cpu")
        print()

    predict = get_predictions(model, img_prepr, debug_params)

    if multple_classes == True:
        cell_bodies = get_segmentation_results(predict, debug_params=debug_params,
               border=False, threshold=threshold_body, multple_classes=multple_classes)

        cell_borders = get_segmentation_results(predict, debug_params=debug_params,
                border=True, threshold=threshold_border, multple_classes=multple_classes)
    else:
        threshold = threshold_body  # threshold_border  # threshold_body
        cell_bodies = get_segmentation_results(predict, debug_params=True,
               border=False, threshold=threshold)

    # Read ground truths

    # concatenate prediction images into one RGB image
    img_pr_conct = np.zeros((512, 512, 3))
    if multple_classes == True:
        img_pr_conct[:, :, 0] = cell_bodies
        img_pr_conct[:, :, 1] = cell_borders
    else:
        img_pr_conct[:, :, 0] = cell_bodies


    mask_cell_bodies = (255-cell_bodies)/255
    mask_pr_original = img_original.copy()
    mask_pr_original[:, :, 0] = mask_pr_original[:, :, 0]*mask_cell_bodies
    mask_pr_original[:, :, 1] = mask_pr_original[:, :, 1]*mask_cell_bodies
    mask_pr_original[:, :, 2] = mask_pr_original[:, :, 2]*mask_cell_bodies

    cell_bodies = cell_bodies.astype(np.uint8)
    img_pr_blend = mask_pr_original.astype(np.uint8)
    img_pr_blend[:, :, 0] += cell_bodies

   
    # cast image data to match types in blend op
    img_original = img_original.astype(np.uint8)
    
    img_pr_conct = img_pr_conct.astype(np.uint8)

    cells = cell_bodies - cell_borders if multple_classes else cell_bodies

    alpha = 0.5
    beta = 1 - alpha
    gamma = 0
    img_filtered = np.zeros((512, 512, 3))
    img_filtered[:, :, 0] = cells
    img_filtered = img_filtered.astype(np.uint8)
    img_filtered = cv.addWeighted(
        img_original, alpha, img_filtered, beta, gamma)


    if (save_results):
        pred_file_name = file_name.replace('.png', '_pred.png')
        cv.imwrite(save_path+pred_file_name, cells)

    predictions = extract_points(cells)
    tp, fp, fn = compute_result(img_original, bounding_box, predictions, file_name=file_name, save_results=save_results, save_path=save_path)
    # *-*-*-*-*-*-*

    return tp, fp, fn


def proces_files_in_path(file_path, gt_file_path, src_prefix, gt_prefix, save_path=None, save_results=False, post_fix=None):
    tps = np.array([])
    fps = np.array([])
    fns = np.array([])
    files = np.array([])

    file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    for i, file_name in enumerate(file_names):

        full_file_name = file_path+file_name
        gt_full_file_name = gt_file_path + gt_prefix + \
            file_name.replace(src_prefix, '')
        if post_fix:
            gt_full_file_name = gt_full_file_name.replace('.png', post_fix)

        img = cv.imread(full_file_name)

        gt_img = cv.imread(gt_full_file_name, cv.IMREAD_GRAYSCALE)

        if np.array(img).size == 1 or np.array(gt_img).size == 1:
            print(f'skiping={full_file_name}')
            continue

        bbs = load_bounding_box(file_name, file_path)

        tp, fp, fn = compare_gt_vs_prediction(
            test_image=full_file_name,
            bounding_box=bbs,
            debug_params=False, threshold_body=.99, threshold_border=0.5,
            save_path=save_path, save_results=save_path)

        tps = np.append(tps, tp)
        fps = np.append(fps, fp)
        fns = np.append(fns, fn)

        files = np.append(files, file_name)

    tns = np.zeros(tps.shape)

    return tps, tns, fps, fns, files


def process_original_img(grp='', save_path=None, save_results=False):
    gt_prefix = ''
    src_prefix = ''
    post_fix = None

    # 'grp' is only meanfull in training rounds. It's left here for convinience 
    if grp == 'dots':
        file_path_for_test_img = '../data/test/src/'
        file_path_for_gt_img = '../data/test/gte/'
    else:
        file_path_for_test_img = '../data/test/src/'
        file_path_for_gt_img = '../data/test/gt/'
        
    tps, tns, fps, fns, files = proces_files_in_path(
        file_path_for_test_img, file_path_for_gt_img, src_prefix, gt_prefix, save_path=save_path, save_results=save_results, post_fix=post_fix)
    return tps, tns, fps, fns, files


def load_bounding_box(file_name, file_path):

    bb_filename = file_name.replace(".png", ".txt")
    bb_path = file_path.replace("/src/", "/bb/")

    f = open(bb_path+bb_filename, "r")
    lines = f.readlines()

    fn = file_name.replace('.txt', '.png')

    img = cv.imread(file_path+fn)

    (w, h, c) = img.shape

    bbs = []

    # Strips the newline character
    for line in lines:
        d = line.split()
        cx, cy, cw, ch = (float)(d[1]), (float)(
            d[2]), (float)(d[3]), (float)(d[4])
        left = (int)(w*(cx - cw/2))
        right = (int)(w*(cx + cw/2))
        top = (int)(h*(cy - ch/2))
        bottom = (int)(h*(cy + ch/2))

        bb = BoundingBox(left, top, right, bottom)
        bbs.append(bb)
        #cv.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 1)

    # plt.imshow(img)
    # plt.show()
    return bbs


def process_images(grp='', save_path=None, save_results=False):

    tps, tns, fps, fns, files = process_original_img(
        grp=grp, save_path=save_path, save_results=save_results)

    g_tps, g_tns, g_fps, g_fns = np.sum(
        tps), np.sum(tns), np.sum(fps), np.sum(fns)
    
    print(f"g_tps = {g_tps}\n g_tns = {g_tns}\n g_fps = {g_fps}\n g_fns = {g_fns} ")

    g_acc = (g_tps + g_tns) / (g_tps + g_tns + g_fps + g_fns)
    g_precision = g_tps / (g_tps + g_fps)
    g_recall = g_tps / (g_tps + g_fns)
    g_f1 = 2 * g_precision * g_recall / (g_precision + g_recall)

    return g_acc, g_precision, g_recall, g_f1, files


for model in models :
    model_name = model
    if model == 'Cell_UNet' :
        cell_model = CellUNet(n_channels=3, n_classes=1)
    elif model == 'UNet' :
        cell_model = UNet(n_channels=3, n_classes=1)
    elif  model == 'Att_UNet' :
        cell_model = UNet_Attention(img_ch=3, output_ch=1) 
    elif  model == 'UNetPP' :
        cell_model = Nested_UNet(in_ch=3, out_ch=1) 
        
    # standardize names accross models 
    cell_model.n_channels = 3
    cell_model.n_classes = 1

    for scenario in scenarios :
        
        if scenario == 1:
            # Scenario Plain
            enhance = False
            grp_img = ''
            dir_checkpoint = 'Unprocessed_scenario'
            
        elif scenario == 2:
            # scenario ICE
            enhance = True
            grp_img = ''
            dir_checkpoint = 'ICE_scenario'
                            
        elif scenario == 3:
            # scenario GTE
            enhance = False
            grp_img = 'gte'
            dir_checkpoint = 'GTE_scenario'
            
        elif scenario == 4:
            # scenario GTE + ICE
            enhance = True
            grp_img = 'gte'
            dir_checkpoint = 'GTE_ICE_scenario'
            
        cell_model_name = f'../data/models/{dir_checkpoint}/{model_name}_best.pth'
        save_path = f'{results_dir}/{dir_checkpoint}/{model_name}/'
        
        try:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        except OSError as e:
            print(e)
            pass
        
        cell_model.load_state_dict(torch.load(cell_model_name))
        g_acc, g_precision, g_recall, g_f1, files = process_images(grp=grp_img, save_path=save_path, save_results=True)
        
        print(files)
        print(f"Global Accuracy = {g_acc:0.4f}")
        print(f"Global Precision = {g_precision:0.4f}")
        print(f"Global Recall = {g_recall:0.4f}")
        print(f"Global F1 = {g_f1:0.4f}")
        
        
        results_path = f'{results_dir}/{dir_checkpoint}/'
        
        try:
            if not os.path.exists(results_path):
                os.makedirs(results_path)
        except OSError:
            pass
        
        results_file = f'{model_name}_Results.txt'
        
        with open(results_path+results_file, 'w') as f:
            f.write(f"Global Accuracy = {g_acc:0.4f}\n")
            f.write(f"Global Precision = {g_precision:0.4f}\n")
            f.write(f"Global Recall = {g_recall:0.4f}\n")
            f.write(f"Global F1 = {g_f1:0.4f}\n")
            f.close()