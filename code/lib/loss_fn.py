#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

beta = 0.25
alpha = 0.25
gamma = 2
epsilon = 1e-5
smooth = 1e-6
count = 0;

def tversky_index(y_true, y_pred):
    y_true_pos = torch.flatten(y_true)
    y_pred_pos = torch.flatten(y_pred)
    true_pos = torch.sum(y_true_pos * y_pred_pos)
    false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.75
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (
                1 - alpha) * false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky_index(y_true, y_pred)

def focal_tversky(y_true, y_pred):
    pt_1 = tversky_index(y_true, y_pred)
    gamma = 0.75
    ftl = torch.pow((1 - pt_1), gamma)
    
    return ftl