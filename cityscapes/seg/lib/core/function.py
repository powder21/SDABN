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
        noisy_image_seg, labels, _ = batch
        noisy_image_seg = noisy_image_seg.to(device)
        labels = labels.long().to(device)

        losses, _ = model(noisy_image_seg, labels)
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


def validate(config, testloader, model, writer_dict, device, noise_level):
    
    model.module.model_train.eval()
    ave_loss = AverageMeter()
    std = torch.tensor([0.229, 0.224, 0.225]).view([1, 3, 1, 1]).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406]).view([1, 3, 1, 1]).to(device)
    confusion_matrix = torch.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES)).cuda()
    start = 0
    np.random.seed(0)

    with torch.no_grad():
        for i, batch in enumerate(testloader):
            # torch.cuda.empty_cache()

            # logging.info(i)
            clean_image, label, _ = batch
            size = label.size()
            B = size[0]

            clean_image = clean_image.to(device)
            label = label.long().to(device)

            noise_np = ((noise_level / 255.0) * np.random.randn(B, 3, 1024, 2048)).astype(np.float32)
            noise = torch.from_numpy(noise_np).to(device)
            
            noisy_image_seg = clean_image+noise
            noisy_image_seg = noisy_image_seg - mean
            noisy_image_seg = noisy_image_seg / std

            losses, pred, confusion_mat = model(noisy_image_seg, label, True)
            # pred = F.upsample(input=pred, size=(
            #             size[-2], size[-1]), mode='bilinear')
            loss = losses.mean()
            ave_loss.update(loss)

            confusion_mat = confusion_mat.sum(dim=0)
            confusion_matrix += confusion_mat

    np.random.seed(None)
    confusion_matrix = confusion_matrix.cpu().numpy()
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()
    print_loss = ave_loss.average()

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', print_loss, global_steps)
    writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return print_loss, mean_IoU, IoU_array
    

def testval(config, test_dataset, testloader, model, 
        sv_dir='', sv_pred=False):
    model.module.eval()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            clean_image, noisy_image, clean_image_seg, noisy_image_seg, label, _, name = batch
            size = label.size()
            pred = test_dataset.multi_scale_inference(
                        model, 
                        noisy_image_seg,
                        scales=config.TEST.SCALE_LIST, 
                        flip=config.TEST.FLIP_TEST)
            
            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.upsample(pred, (size[-2], size[-1]), 
                                   mode='bilinear')

            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            if sv_pred:
                sv_path = os.path.join(sv_dir,'test_val_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)
            
            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc

def test(config, test_dataset, testloader, model, 
        sv_dir='', sv_pred=True):
    model.module.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            clean_image, noisy_image, clean_image_seg, noisy_image_seg, size, name = batch
            size = size[0]
            pred = test_dataset.multi_scale_inference(
                        model, 
                        noisy_image_seg,
                        scales=config.TEST.SCALE_LIST, 
                        flip=config.TEST.FLIP_TEST)
            
            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.upsample(pred, (size[-2], size[-1]), 
                                   mode='bilinear')

            if sv_pred:
                sv_path = os.path.join(sv_dir,'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)
