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
from torch.utils.data.distributed import DistributedSampler
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
    parser.add_argument('--cfg', help='experiment configure file name', type=str,default='experiments/cityscapes/seg_hrnet_w18_small_v2_512x1024_sgd_lr3e-2_wd5e-4_bs_40_epoch484.yaml')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--noise", type=float, default=50.0)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument('--resume_s1', type=str, default='/media/larry/code/SDABN/test_model/city/50/smallv2_no_pre.pth')
    parser.add_argument('--resume_d1', type=str, default='')
    parser.add_argument('--resume_s2', type=str, default='')
    parser.add_argument('--resume_d2', type=str, default='')
    parser.add_argument('--resume_s3', type=str, default='')
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
    distributed = False
    device = torch.device('cuda:{}'.format(args.local_rank))

    # build model
    model_list = []
    for i in range(args.stage//2):
        model_list.append(eval('lib.models.'+config.MODEL.NAME + '.get_seg_model')(config))
        model_list.append(DDFN_SFT(category_number=config.DATASET.NUM_CLASSES))
    model_list.append(eval('lib.models.'+config.MODEL.NAME + '.get_seg_model')(config))

    def resume_params_custom(model, path, model_name=''):
        if path == 'new':
            print('not resume')
            return model
        else:
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

    model_path = [args.resume_s1,args.resume_d1,args.resume_s2,args.resume_d2,args.resume_s3]
    # model_path = [args.resume_s1,args.resume_d1,args.resume_s2]
    for i in range(args.stage):
        resume_params_custom(model_list[i], model_path[i])

    if args.local_rank == 0:
        # provide the summary of model
        dump_input = torch.rand(
            (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
            )
        logger.info(get_model_summary(model_list[0].to(device), dump_input.to(device)))

        if args.stage>1:
            dump_input = torch.rand(
                (1, 3+config.DATASET.NUM_CLASSES, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
            )
            logger.info(get_model_summary(model_list[1].to(device), dump_input.to(device)))

        # copy model file
        this_dir = os.path.dirname(__file__)
        models_dst_dir = os.path.join(final_output_dir, 'models')
        if os.path.exists(models_dst_dir):
            shutil.rmtree(models_dst_dir)
        shutil.copytree(os.path.join(this_dir, './lib/models'), models_dst_dir)

    # if distributed:
    #     torch.cuda.set_device(args.local_rank)
    #     torch.distributed.init_process_group(
    #         backend="nccl", init_method="env://",
    #     )

    # prepare data
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    train_dataset = eval('lib.datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TRAIN_SET,
                        num_samples=None,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=config.TRAIN.MULTI_SCALE,
                        flip=config.TRAIN.FLIP,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TRAIN.BASE_SIZE,
                        crop_size=crop_size,
                        downsample_rate=config.TRAIN.DOWNSAMPLERATE,
                        scale_factor=config.TRAIN.SCALE_FACTOR,
                        noise_level=args.noise)

    # if distributed:
    #     train_sampler = DistributedSampler(train_dataset)
    # else:
    #     train_sampler = None
    train_sampler = None

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=config.TRAIN.SHUFFLE and train_sampler is None,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler)

    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('lib.datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TEST_SET,
                        num_samples=config.TEST.NUM_SAMPLES,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=False,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TEST.BASE_SIZE,
                        crop_size=test_size,
                        center_crop_test=config.TEST.CENTER_CROP_TEST,
                        downsample_rate=1,
                        noise_level=args.noise)

    # if distributed:
    #     test_sampler = DistributedSampler(test_dataset)
    # else:
    #     test_sampler = None
    test_sampler = None

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
        sampler=test_sampler)

    # criterion
    if config.LOSS.USE_OHEM:
        criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                     thres=config.LOSS.OHEMTHRES,
                                     min_kept=config.LOSS.OHEMKEEP,
                                     weight=train_dataset.class_weights)
    else:
        criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                 weight=train_dataset.class_weights)

    # load fixed model list

    model = FullModel(criterion, model_list, args.stage)
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    model = nn.DataParallel(model)
    # model = nn.parallel.DistributedDataParallel(
        # model, device_ids=[args.local_rank], output_device=args.local_rank)



    # optimizer
    if config.TRAIN.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD([{'params':
                                  filter(lambda p: p.requires_grad,
                                         model.module.model_train.parameters()),
                                  'lr': config.TRAIN.LR}],
                                lr=config.TRAIN.LR,
                                momentum=config.TRAIN.MOMENTUM,
                                weight_decay=config.TRAIN.WD,
                                nesterov=config.TRAIN.NESTEROV,
                                )
    else:
        raise ValueError('Only Support SGD optimizer')

    epoch_iters = np.int(train_dataset.__len__() / 
                        config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))
    best_mIoU = 0
    last_epoch = 0
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file,
                        map_location=lambda storage, loc: storage)
            best_mIoU = checkpoint['best_mIoU']
            last_epoch = checkpoint['epoch']
            model.module.model_train.load_state_dict(checkpoint['model_weights'])
            model.module.loss.load_state_dict(checkpoint['loss_weight'])

            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))

    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH + config.TRAIN.EXTRA_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters
    for i in range(args.stage-1):
        model.module.model_fix_list[i].eval()
    for epoch in tqdm(range(last_epoch, end_epoch)):
        torch.cuda.empty_cache()
        begin = time.time()
        # if distributed:
        #     train_sampler.set_epoch(epoch)
        train(config, epoch, config.TRAIN.END_EPOCH,
              epoch_iters, config.TRAIN.LR, num_iters,
              trainloader, optimizer, model, writer_dict,
              device)
        logger.info('train done!')
        t1 = time.time()

        if epoch >= (14*end_epoch//15):
        
            np.random.seed(0)
            torch.cuda.empty_cache()
            valid_loss, mean_IoU, IoU_array = validate(config,
                        testloader, model, writer_dict, device, noise_level=args.noise)
            np.random.seed(None)
            logger.info('test done!')
        t2 = time.time()
        print('################  train time:',t1-begin,' test time:',t2-t1,'  ###################')

        logger.info('=> saving checkpoint to {}'.format(
            final_output_dir + 'checkpoint.pth.tar'))
        torch.save({
            'epoch': epoch+1,
            'best_mIoU': best_mIoU,
            'model_weights':model.module.model_train.state_dict(),
            'loss_weight':model.module.loss.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(final_output_dir,'checkpoint.pth.tar'))

        if epoch >= (14*end_epoch//15):
            if mean_IoU > best_mIoU:
                best_mIoU = mean_IoU
                torch.save({
                    'best_mIoU': best_mIoU,
                    'model_weights': model.module.model_train.state_dict(),
                    'loss_weight': model.module.loss.state_dict(),
                }, os.path.join(final_output_dir, 'best.pth.tar'))
            msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}'.format(
                    valid_loss, mean_IoU, best_mIoU)
            logging.info(msg)
            logging.info(IoU_array)

        if epoch == end_epoch - 1:
            torch.save({
                'best_mIoU': best_mIoU,
                'model_weights': model.module.model_train.state_dict(),
                'loss_weight': model.module.loss.state_dict(),
            }, os.path.join(final_output_dir, 'final_state.pth.tar'))

            writer_dict['writer'].close()
            end = timeit.default_timer()
            logger.info('Hours: %d' % np.int((end-start)/3600))
            logger.info('Done')


if __name__ == '__main__':
    main()
