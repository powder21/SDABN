# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import shutil
import sys
import logging
import time
import timeit
from pathlib import Path
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter

# import _init_paths
import lib.models
import lib.datasets
from lib.config import config
from lib.config import update_config
from lib.core.criterion import CrossEntropy, OhemCrossEntropy, compute_psnr
from lib.core.function import train, validate
from lib.utils.modelsummary import get_model_summary
from lib.utils.utils import create_logger, FullModel, get_rank
from denoising.ddfn_sft import DeepBoosting as DDFN_SFT

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.add_argument('--cfg', help='experiment configure file name', type=str, default='experiments/120x120_adam_lr1e-3_wd1e-4_bs_64_epoch30.yaml')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--stage", type=int, default=2)
    parser.add_argument('--train_data', type=str, default='/media/larry/dataset/SDABN/OST/train8800/images')
    parser.add_argument('--valid_data', type=str, default='/media/larry/dataset/SDABN/OST/valid1100/images')
    parser.add_argument('--resume_seg1', type=str, default='/media/larry/code/SDABN/test_model/ost/s.pth')
    parser.add_argument('--resume_d1', type=str, default='')
    parser.add_argument('--resume_seg2', type=str, default='')
    parser.add_argument('--resume_d2', type=str, default='')
    parser.add_argument('--resume_seg3', type=str, default='')
    parser.add_argument('opts', help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)
    return args


def main():
    args = parse_args()

    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, 'train')
    logger.info(pprint.pformat(args))
    logger.info(config)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)
    device = torch.device('cuda:{}'.format(args.local_rank))

    # build model
    model_list = []
    for i in range(args.stage//2):
        model_list.append(eval('lib.models.'+config.MODEL.NAME + '.get_seg_model')(config))
        model_list.append(DDFN_SFT(category_number=config.DATASET.NUM_CLASSES))
    # model_list.append(eval('lib.models.'+config.MODEL.NAME + '.get_seg_model')(config))

    def resume_params_custom(model, path, model_name=''):
        t1 = time.time()
        print(model_name, 'resuming weights from %s ... ' % path, end='', flush=True)
        if os.path.isfile(path):
            checkpoint = torch.load(path, 'cpu')
            model.load_state_dict(checkpoint['model_weights'])
        else:
            raise AttributeError('No checkpoint found at %s' % path)
        print('Done (time: %.2fs)' % (time.time() - t1))
        for _item in checkpoint:
            if 'weight' not in _item:
                print('     ' + os.path.basename(path) + '->' + _item + ':', checkpoint[_item])
        return model

    model_path = [args.resume_seg1,args.resume_d1,args.resume_seg2,args.resume_d2,args.resume_seg3]
    for i in range(args.stage-1):
        resume_params_custom(model_list[i], model_path[i])

    # copy model file
    this_dir = os.path.dirname(__file__)
    models_dst_dir = os.path.join(final_output_dir, 'models')
    if os.path.exists(models_dst_dir):
        shutil.rmtree(models_dst_dir)
    shutil.copytree(os.path.join(this_dir, './lib/models'), models_dst_dir)

    # prepare data
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    train_dataset = eval('lib.datasets.' + config.DATASET.DATASET + '_train')(
        data_path=args.train_data,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        crop_size=crop_size)

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True)

    test_dataset = eval('lib.datasets.' + config.DATASET.DATASET + '_test')(
        data_path=args.valid_data,
        ignore_label=config.TRAIN.IGNORE_LABEL)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)


    # load fixed model list
    model = FullModel(model_list, args.stage)
    model = model.to(device)
    model = nn.DataParallel(model)

    # optimizer
    if config.TRAIN.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(model_list[-1].parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, amsgrad=False, weight_decay=config.TRAIN.WD)
    else:
        raise ValueError('Only Support adam optimizer')

    epoch_iters = np.int(train_dataset.__len__() / config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))
    best_psnr = 0
    last_epoch = 0
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file,
                        map_location=lambda storage, loc: storage)
            best_psnr = checkpoint['best_psnr']
            last_epoch = checkpoint['epoch']
            model.module.model_train.load_state_dict(checkpoint['model_weights'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))

    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH + config.TRAIN.EXTRA_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters
    for i in range(args.stage-1):
        model.module.model_fix_list[i].eval()
    for epoch in tqdm(range(last_epoch, end_epoch)):
        train(config, epoch, config.TRAIN.END_EPOCH,
              epoch_iters, config.TRAIN.LR, num_iters,
              trainloader, optimizer, model, writer_dict,
              device)
        if epoch % 100 == 0:
            np.random.seed(0)
            valid_loss, mean_psnr = validate(config, testloader, model, writer_dict, device)
            np.random.seed(None)

            logger.info('=> saving checkpoint to {}'.format(
                final_output_dir + 'checkpoint.pth.tar'))
            torch.save({
                'epoch': epoch+1,
                'best_psnr': best_psnr,
                'model_weights':model.module.model_train.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(final_output_dir,'checkpoint.pth.tar'))

            if mean_psnr > best_psnr:
                best_psnr = mean_psnr
                torch.save({
                    'best_psnr': best_psnr,
                    'model_weights': model.module.model_train.state_dict(),
                }, os.path.join(final_output_dir, 'best.pth.tar'))
            msg = 'Loss: {:.3f}, Mean_psnr: {: 4.4f}, Best_psnr: {: 4.4f}'.format(
                    valid_loss, mean_psnr, best_psnr)
            logging.info(msg)
                
        if epoch == end_epoch - 1:
            torch.save({
                'best_psnr': best_psnr,
                'model_weights': model.module.model_train.state_dict(),
            }, os.path.join(final_output_dir, 'final_state.pth.tar'))

            writer_dict['writer'].close()
            end = timeit.default_timer()
            logger.info('Hours: %d' % np.int((end-start)/3600))
            logger.info('Done')


if __name__ == '__main__':
    main()
