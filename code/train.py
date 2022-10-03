#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 11:25:56 2021

In this code we train the Deep Learning models (Cell_UNet, UNet, Att_UNet, UNetPP)
according to different scenarios (Unprocessed, ICE, GTE, ICE+GTE). 
To select the model that wants to be trained, update var: 'modelName'.
To select the scenario that the model will to be trained on, update var: 'scenario'.

In order to run this code successfully, visdom server needs to be running
(use the following command in a terminal to start it: 'python -m visdom.server')
to be able to see live visualizations of the tranning loss. If visdom is not 
up & running before executing this code, trainig will still occour however 
a bounch of error messages will be shown in the logs.

Models are saved at: ../data/models/{scenario}
Best model is: {modelName}_best.pth


@author: alfonso
"""

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torch.autograd import Function
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import random
import logging
import visdom
import sys
import os
import warnings
warnings.filterwarnings('ignore')


sys.path += ['./lib', './models']

from loss_fn import focal_tversky
from neurondataset import NeuronDataset, TrainOnEnum

def random_seed(seed_value, use_cuda=True):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False

seeds = 31495  # random seed
random_seed(seeds)

""" 
    To train a specific DL model change 'modelName' to one of the following params:
    Cell_UNet, UNet, Att_UNet, UNetPP 
"""
modelName = "Cell_UNet"

from cell_unet import CellUNet 
from unet import UNet
from atention_unet import UNet_Attention 
from unetpp import Nested_UNet

if modelName == "Cell_UNet" :
    model = CellUNet(n_channels=3, n_classes=1)  
    batch_size = 4
elif modelName == "UNet" :
    model = UNet(n_channels=3, n_classes=1) 
    batch_size = 4
elif modelName == "Att_UNet" :
    model = UNet_Attention(img_ch=3, output_ch=1) 
    batch_size = 3
elif modelName == "UNetPP" :
    model = Nested_UNet(in_ch=3, out_ch=1) 
    batch_size = 2

# standardize names
model.n_channels = 3
model.n_classes = 1


augment_images = True

"""
    To excecute a traning round with the proper parameters according to article, assign to 'scenario' one of the following values:
    1 - for training models withOUT Ground Truth Enhancement (GTE) and withOUT Image Conditioning Enhancement (ICE) aka Unprocessed
    2 - for training models with ICE only
    3 - for training models with GTE only
    4 - for training models with GTE + ICE
"""
scenario = 4

if scenario == 1:
    # Scenario Plain
    enhance_images = False
    grp_images = ''
    dir_checkpoint = 'Unprocessed_scenario/'
elif scenario == 2:
    # scenario ICE
    enhance_images = True
    grp_images = ''
    dir_checkpoint = 'ICE_scenario/'
elif scenario == 3:
    # scenario GTE
    enhance_images = False
    grp_images = 'gte'
    dir_checkpoint = 'GTE_scenario/'
elif scenario == 4:
    # scenario GTE + ICE
    enhance_images = True
    grp_images = 'gte'
    dir_checkpoint = 'GTE_ICE_scenario/'


dir_checkpoint = f'../data/models/{dir_checkpoint}'

def encode_one_hot_label(pred, target):
    one_hot_label = pred.detach() * 0

    one_hot_label.scatter_(1, target.unsqueeze(1), 1)
    return one_hot_label


def eval_learning(net, loader, device, criterion=None):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()  # turn off trainning
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    f1_pred = 0
    with tqdm(total=n_val, desc='Evaluation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)
            mask_pred = mask_pred.squeeze(1)

            if net.n_classes > 1:

                tot += criterion(mask_pred, true_masks).item()

                pred = F.softmax(mask_pred, 1)

                label = encode_one_hot_label(pred, true_masks.squeeze(1))
                label = label.squeeze(0).detach().cpu().numpy()
                pred = pred.squeeze(0).detach().cpu().numpy()
                pred[pred >= 0.5] = 1
                pred[pred < 0.5] = 0
                f1_pred += f1_score(label.flatten(),
                                    pred.flatten(), average='macro')
            else:
                mask_pred = torch.sigmoid(mask_pred)

                tot += focal_tversky(true_masks, mask_pred).item()

                true_masks = true_masks.squeeze(0).detach().cpu().numpy()
                o = mask_pred.squeeze(0).detach().cpu().numpy()
                o[o >= 0.5] = 1
                o[o < 0.5] = 0
                f1_pred += f1_score(true_masks.flatten(),
                                    o.flatten(), average='macro')
            pbar.update()

    f1_pred /= n_val
    net.train()  # turn on trainning
    return tot / n_val, f1_pred


def train_net(net,
              epochs=10,
              batch_size=batch_size,
              lr=0.001,
              save_cp=True,
              trainOn=TrainOnEnum.Cell,
              grp=''):

    
    if grp == 'gte':
        imgs_dir = '../data/train/src/'
        masks_dir = '../data/train/gte/'

        val_imgs_dir = '../data/val/src/'
        val_masks_dir = '../data/val/gte/'

    else:
        imgs_dir = '../data/train/src/'
        masks_dir = '../data/train/gt/'

        val_imgs_dir = '../data/val/src/'
        val_masks_dir = '../data/val/gt/'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelType = 'cellBorder' if trainOn == TrainOnEnum.CellBorder else 'cellBody' if trainOn == TrainOnEnum.CellBody else 'cell'

    net.to(device=device)
    train_dataset = NeuronDataset(trainOn=trainOn, imgs_dir=imgs_dir, masks_dir=masks_dir,
                                  mask_suffix='', augment=augment_images, enhance=enhance_images)
    val_dataset = NeuronDataset(trainOn=trainOn, imgs_dir=val_imgs_dir,
                                masks_dir=val_masks_dir, mask_suffix='', augment=False, enhance=enhance_images)
    n_val = len(val_dataset)
    n_train = len(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')

    optimizer = optim.Adam(net.parameters(), lr=1.5e-4 /
                           np.sqrt(5), weight_decay=2e-5)  # lr = 1e-4)

    # Reduce learning rate when a metric has stopped improving
    min_or_max = 'min' if net.n_classes > 1 else 'max'
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, min_or_max, patience=2)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.BCEWithLogitsLoss().cuda()

    prev_f1_val = 1e-3
    min_bat_loss = 1e3
    f1_avg = 0
    losses = np.array([])
    elapsed_epochs = np.array([])
    f1s = np.array([])
    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long

                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                masks_pred = masks_pred.squeeze(0)

                masks_pred = torch.sigmoid(masks_pred)
                loss = focal_tversky(true_masks, masks_pred)


                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})
                if min_bat_loss > loss.item():
                    print('')
                    print(
                        f'A better batch model has been found with loss={loss.item()} previous loss={min_bat_loss}')
                    min_bat_loss = loss.item()
                    if save_cp:
                        try:
                            os.mkdir(dir_checkpoint)
                            logging.info('Created checkpoint directory')
                        except OSError:
                            pass
                        torch.save(net.state_dict(), dir_checkpoint +
                                   f'{modelName}_{modelType}_{grp}_unet_best_batch.pth')
                        print("Better batch model saved")

                losses = np.append(losses, [loss.item()])
                elapsed_epochs = np.append(elapsed_epochs, [global_step])
                
                vis.line(
                    Y=losses,
                    X=elapsed_epochs,
                    opts=opts,
                    win=metric_window
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1

            val_score, f1_pred = eval_learning(
                net, val_loader, device, criterion=criterion)
            f1_avg += f1_pred

            if prev_f1_val < f1_pred:
                logging.info(
                    f'A better model has been found with f1: {f1_pred}')
                print('')
                print(
                    f'A better model has been found with f1={f1_pred} previous f1={prev_f1_val}')
                prev_f1_val = f1_pred

                if save_cp:
                    try:
                        os.mkdir(dir_checkpoint)
                        logging.info('Created checkpoint directory')
                    except OSError:
                        pass
                    torch.save(net.state_dict(), dir_checkpoint +
                               f'{modelName}_{modelType}_{grp}_best_val_{f1_pred}_epoch_{epoch + 1}.pth')
                    torch.save(net.state_dict(), dir_checkpoint +
                               f'{modelName}_best.pth')
                    print("Better model saved")

            scheduler.step(val_score)
            writer.add_scalar(
                'learning_rate', optimizer.param_groups[0]['lr'], global_step)

            logging.info('Evaluation loss: {}'.format(val_score))
            writer.add_scalar('Loss/test', val_score, global_step)

            if net.n_classes == 1:
                true_masks = true_masks.unsqueeze(0)
                masks_pred = masks_pred.unsqueeze(0)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'{modelName}_{modelType}_CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')
            print(f'Checkpoint {epoch + 1} saved !')

    writer.close()
    f1_avg /= epochs
    print(f'f1_avg={f1_avg}')


opts = {
    'layoutopts': {
        'plotly': {
            'yaxis': {
                'range': [0, 3],
                'autorange': False,
            }
        }
    },
    'markers': False
}
opts = {'markers': False}
vis = visdom.Visdom()
metric_window = vis.line(
    Y=torch.zeros((1)).cpu(),
    X=torch.zeros((1)).cpu(),
    opts=dict(xlabel='epoch', ylabel='Loss',
              title='training loss', legend=['Loss'])
)


train_net(model, trainOn=TrainOnEnum.Cell, grp=grp_images)
