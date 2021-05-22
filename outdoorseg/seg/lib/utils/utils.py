# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import numpy as np
import time
import torch
import torch.nn as nn
from torch.nn import functional as F

class FullModel(nn.Module):
    """
    Distribute the loss on multi-gpu to reduce
    the memory cost in the main gpu.
    You can check the following discussion.
    https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
    """

    def __init__(self, loss, model_list, stages, num_class, ignore_label):
        super(FullModel, self).__init__()
        self.model_fix_list = []
        for i in range(len(model_list)-1):
            name = 'model' + str(i+1)
            setattr(self, name, model_list[i])
            self.model_fix_list.append(eval('self.model' + str(i + 1)))
        self.stages = stages
        self.model_train = model_list[-1]

        self.loss = loss
        self.num_class = num_class
        self.ignore_label = ignore_label

    def forward(self, inputs, labels, iou=False):
        std = torch.tensor([0.229, 0.224, 0.225]).view([1, 3, 1, 1]).cuda()
        mean = torch.tensor([0.485, 0.456, 0.406]).view([1, 3, 1, 1]).cuda()
        noisy_image = inputs * std + mean  # [0,1]
        denoise_image = 0

        with torch.no_grad():
            for i in range(self.stages//2):
                seg_prob = eval('self.model'+str(i*2+1))(inputs)
                h, w = inputs.size(2), inputs.size(3)
                seg_prob = F.upsample(input=seg_prob, size=(h, w), mode='bilinear')

                denoise_in = denoise_image + noisy_image  # [0,1]
                input_cat = torch.cat((denoise_in, seg_prob), 1)
                denoise_image = []
                if not iou:
                    for j in range(len(input_cat)//2):
                        denoise_image.append(eval('self.model'+str(i*2+2))(input_cat[j*2:(j+1)*2]))
                else:
                    for j in range(len(input_cat)):
                        denoise_image.append(eval('self.model'+str(i*2+2))(input_cat[j:(j+1)]))
                denoise_image = torch.cat(denoise_image, 0)
                inputs = (denoise_image - mean)/std

        seg_prob = self.model_train(inputs)   
        loss = self.loss(seg_prob, labels)
        if iou:
            h, w = inputs.size(2), inputs.size(3)
            seg_prob = F.upsample(input=seg_prob, size=(h, w), mode='bilinear')
            confusion_matrix = get_confusion_matrix(labels, seg_prob, self.num_class, self.ignore_label)
            return torch.unsqueeze(loss, 0), seg_prob, torch.unsqueeze(confusion_matrix,0)
        else:
            return torch.unsqueeze(loss, 0), seg_prob

def get_world_size():
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()

def get_rank():
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir(parents=True)

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
            (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)

def get_confusion_matrix(label, pred, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    # begin = time.time()
    # # output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    # # seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)

    # _, max_ind = torch.max(pred, 1, False)  # B H W
    # t1 = time.time()
    # seg_pred = np.asarray(max_ind.cpu().numpy(), dtype=np.uint8)
    # t2 = time.time()
    # seg_gt = np.asarray(label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)
    # t3 = time.time()

    # ignore_index = seg_gt != ignore
    # seg_gt = seg_gt[ignore_index]
    # seg_pred = seg_pred[ignore_index]
    

    # index = (seg_gt * num_class + seg_pred).astype('int32')
    # label_count = np.bincount(index)
    # confusion_matrix = np.zeros((num_class, num_class))
    

    # for i_label in range(num_class):
    #     for i_pred in range(num_class):
    #         cur_index = i_label * num_class + i_pred
    #         if cur_index < len(label_count):
    #             confusion_matrix[i_label,
    #                              i_pred] = label_count[cur_index]
    
    # print(t1-begin,t2-t1,t3-t2)
    # return confusion_matrix
    _, seg_pred = torch.max(pred, 1, False)  # B H W
    seg_gt = label  #long
    ignore = torch.tensor(ignore).cuda().long()
    num_class =torch.tensor(num_class).cuda().long()

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred)
    label_count = torch.bincount(index)
    if len(label_count)<num_class*num_class:
        confusion_matrix = torch.zeros((num_class*num_class)).cuda().long()
        confusion_matrix[:len(label_count)] = label_count
        label_count = confusion_matrix
    label_count = label_count.view((num_class, num_class)).float()
    # label_count = np.asarray(label_count.cpu().numpy(),dtype=np.float32)
    return label_count

def adjust_learning_rate(optimizer, base_lr, max_iters, 
        cur_iters, power=0.9):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr
    return lr
