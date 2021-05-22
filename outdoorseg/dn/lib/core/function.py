# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import logging
import os
import time

import numpy as np
import numpy.ma as ma
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F

from lib.utils.utils import AverageMeter
from lib.utils.utils import get_confusion_matrix
from lib.utils.utils import adjust_learning_rate


def train(config, epoch, num_epoch, epoch_iters, base_lr, num_iters,
         trainloader, optimizer, model, writer_dict, device):
    
    # Training
    model.module.model_train.train()
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    for i_iter, batch in enumerate(trainloader):
        noisy_image_seg, clean_image, _ = batch
        noisy_image_seg = noisy_image_seg.to(device)
        clean_image = clean_image.to(device)

        losses, _ = model(noisy_image_seg, clean_image)
        loss = losses.mean()

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(loss)
        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  0.0001,
                                  num_iters,
                                  i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0:
            print_loss = ave_loss.average()
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {:.6f}, Loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters, 
                      batch_time.average(), lr, print_loss)
            logging.info(msg)
            
            writer.add_scalar('train_loss', print_loss, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1


def validate(config, testloader, model, writer_dict, device):
    
    model.module.model_train.eval()
    ave_loss = AverageMeter()
    psnr_list = []
    std = torch.tensor([0.229, 0.224, 0.225]).view([1, 3, 1, 1]).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406]).view([1, 3, 1, 1]).to(device)
    np.random.seed(0)
    def compute_psnr(im1, im2):
        if im1.shape != im2.shape:
            raise Exception('Shapes of two images are not equal')
        rmse = np.sqrt(((np.asfarray(im1) - np.asfarray(im2)) ** 2).mean())
        psnr = 20 * np.log10(255.0 / rmse)
        return psnr

    with torch.no_grad():
        for i, batch in enumerate(testloader):
            clean_image, _, _ = batch
            clean_image = clean_image.to(device)

            _B, _C, _H, _W = clean_image.size()
            noise_np = ((50 / 255.0) * np.random.randn(_B, _C, _H, _W)).astype(np.float32)
            noise = torch.from_numpy(noise_np).to(device)
            
            noisy_image_seg = clean_image + noise
            noisy_image_seg = noisy_image_seg - mean
            noisy_image_seg = noisy_image_seg / std

            losses, denoise = model(noisy_image_seg, clean_image)

            loss = losses.mean()
            ave_loss.update(loss)

            clean_image = clean_image[:, :, 1:-1, 1:-1]
            denoise = denoise.cpu().numpy() * 255.0
            clean_image = clean_image.cpu().numpy() * 255.0
            for pred, target in zip(denoise, clean_image):
                psnr_list.append(compute_psnr(pred, target))

    np.random.seed(None)
    mean_psnr = sum(psnr_list)/len(psnr_list)
    print_loss = ave_loss.average()

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', print_loss, global_steps)
    writer.add_scalar('valid_psnr', mean_psnr, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return print_loss, mean_psnr
