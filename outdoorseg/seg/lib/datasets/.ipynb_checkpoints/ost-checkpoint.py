# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os
import glob
import cv2
import numpy as np
from PIL import Image
import time
import torch
import random
import torchvision
from torch.nn import functional as F
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from torch.utils import data

class OST_train(data.Dataset):
    def __init__(self,
                 data_path,
                 ignore_label=-1,
                 crop_size=(480, 480),
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        super(OST_train, self).__init__()

        self.noise_sigma = 50
        self.ignore_label = ignore_label
        self.crop_size = crop_size[0]
        self.base_size = self.crop_size * 2
        self.mean = mean
        self.std = std

        # transform operators
        self.rc = torchvision.transforms.RandomCrop(self.crop_size, padding=0, pad_if_needed=False)
        self.rhf = torchvision.transforms.RandomHorizontalFlip(p=0.5)
        self.rvf = torchvision.transforms.RandomVerticalFlip(p=0.5)

        raw_image_paths = sorted(glob.glob(os.path.join(data_path, '*.jpg')))
        if len(raw_image_paths)!=8800:
            print(raw_image_paths)
        assert len(raw_image_paths)==8800

        count = 1
        raw_images_list = []
        annotations_list = []

        for raw_image_path in raw_image_paths:
            raw_image = np.array(Image.open(raw_image_path))
            raw_images_list.append(raw_image)
            annotation_path = (raw_image_path.replace('images', 'annotations')).replace('jpg', 'png')
            annotation = np.array(Image.open(annotation_path))
            annotations_list.append(annotation)
            # if count==100:
            #     break
            # count += 1
            # print(count)
        self.raw_images = np.array(raw_images_list)
        self.annotations = np.array(annotations_list)
        print('train data: ', len(self.raw_images))


    def __len__(self):
        return len(self.raw_images)

    def __getitem__(self, index):
        def _random_rotation(im):
            r = np.random.choice([0, 1, 2, 3])
            if r:
                im = np.rot90(im, r)
            return im

        def _pad_image(image, h, w, size, padvalue):
            # pad_image = image.copy()
            pad_h = max(size[0] - h, 0)
            pad_w = max(size[1] - w, 0)
            if pad_h > 0 or pad_w > 0:
                # pad = torchvision.transforms.Pad((0,pad_h,0,pad_w),padvalue,'constant')
                pad = torchvision.transforms.Pad((0, 0, pad_w, pad_h), padvalue, 'constant')

                image = pad(image)
            return image

        def _rand_crop(image, label, crop_size):
            h, w = image.shape[:-1]
            image = _pad_image(image, h, w, crop_size, 0.)
            label = _pad_image(label, h, w, crop_size, 8)

            new_h, new_w = label.shape
            x = random.randint(0, new_w - crop_size[1])
            y = random.randint(0, new_h - crop_size[0])
            image = image[y:y + crop_size[0], x:x + crop_size[1]]
            label = label[y:y + crop_size[0], x:x + crop_size[1]]

            return image, label

        def _multi_scale_aug(image, label, rand_scale=1, base=640):
            long_size = np.int(base * rand_scale + 0.5)
            # h, w = image.shape[:2]
            h, w = image.height, image.width
            if h > w:
                new_h = long_size
                new_w = np.int(w * long_size / h + 0.5)
            else:
                new_w = long_size
                new_h = np.int(h * long_size / w + 0.5)

            image = image.resize((new_w, new_h), resample=Image.BILINEAR)
            if label is not None:
                label = label.resize((new_w, new_h), resample=Image.NEAREST)

            else:
                return image

            return image, label

        raw_image = Image.fromarray(self.raw_images[index])
        annotation = self.annotations[index]
        annotation[annotation == 255] = self.ignore_label
        annotation = Image.fromarray(annotation)

        raw_image = raw_image.resize((self.base_size, self.base_size), resample=Image.BICUBIC)
        annotation = annotation.resize((self.base_size, self.base_size), resample=Image.NEAREST)

        rand_scale = 0.5 + random.randint(0, 6) / 10.0
        raw_image, annotation = _multi_scale_aug(raw_image, annotation, rand_scale=rand_scale, base=self.base_size)

        h, w = raw_image.height, raw_image.width
        raw_image = _pad_image(raw_image, h, w, (self.crop_size, self.crop_size), 0)
        annotation = _pad_image(annotation, h, w, (self.crop_size, self.crop_size), self.ignore_label)

        raw_image = np.array(raw_image)
        annotation = np.expand_dims(np.array(annotation), -1)
        img_ann = Image.fromarray(np.concatenate((raw_image, annotation), -1))

        # random crop
        img_ann = self.rc(img_ann)

        # random flip
        img_ann = self.rhf(img_ann)
        img_ann = self.rvf(img_ann)

        img_ann = np.array(img_ann)
        seg_part = img_ann[:, :, 3:4]
        image_part = img_ann[:, :, 0:3]
        img_ann = np.concatenate([image_part, seg_part], -1)  # (120, 120, 2)

        # random rotation
        img_ann = _random_rotation(img_ann)  # (120, 120, 2)

        image_part = img_ann[:, :, 0:3]
        seg_part = img_ann[:, :, 3]

        # generate noisy im
        # image_part = np.expand_dims(image_part, axis=0)
        image_part = image_part.astype(np.float32) / 255.0
        noise = (self.noise_sigma / 255.0) * np.random.randn(self.crop_size, self.crop_size, 3)
        noise_image = image_part + noise.astype(np.float32)


        # seg_part = np.expand_dims(seg_part, axis=0)
        seg_part = seg_part.astype(np.float32)

        # a = np.concatenate([image_part, noise_image], 2)
        # a[a < 0] = 0
        # a[a > 1] = 1
        # pil_im = Image.fromarray(np.cast[np.uint8](a[0] * 255))
        #
        # pil_im.show()

        noisy_image_seg = noise_image - self.mean
        noisy_image_seg = np.cast[np.float32](noisy_image_seg / self.std)

        noisy_image_seg = noisy_image_seg.transpose(2,0,1)
        # image_part = image_part.transpose(2,0,1)

        return noisy_image_seg, seg_part, noisy_image_seg.shape

class OST_test(data.Dataset):
    def __init__(self,
                 data_path,
                 ignore_label=-1):

        super(OST_test, self).__init__()

        self.noise_sigma = 50
        self.ignore_label = ignore_label

        raw_image_paths = sorted(glob.glob(os.path.join(data_path, '*.jpg')))
        if len(raw_image_paths)!=1100:
            print(raw_image_paths)
        assert len(raw_image_paths)==1100

        count = 1
        raw_images_list = []
        annotations_list = []

        for raw_image_path in raw_image_paths:
            raw_image = np.array(Image.open(raw_image_path))
            raw_images_list.append(raw_image)
            annotation_path = (raw_image_path.replace('images', 'annotations')).replace('jpg', 'png')
            annotation = np.array(Image.open(annotation_path))
            annotations_list.append(annotation)
            # if count==100:
            #     break
            # count += 1
            # print(count)
        self.raw_images = np.array(raw_images_list)
        self.annotations = np.array(annotations_list)
        print('test data: ', len(self.raw_images))
        


    def __len__(self):
        return len(self.raw_images)

    def __getitem__(self, index):

        raw_image = self.raw_images[index]
        annotation = self.annotations[index]
        annotation[annotation == 255] = self.ignore_label  # convert void(255) to -1

        # generate noisy im
        # image_part = np.expand_dims(raw_image, axis=0)
        image_part = raw_image.transpose(2, 0, 1)
        image_part = image_part.astype(np.float32) / 255.0
        # _, H, W = np.shape(image_part)
        # noise = (self.noise_sigma / 255.0) * np.random.randn(3, H, W)
        # noise_image = image_part + noise.astype(np.float32)
        # seg_part = np.expand_dims(annotation, axis=0)
        seg_part = annotation.astype(np.float32)

        # a = np.concatenate([image_part, noise_image], 2)
        # a[a < 0] = 0
        # a[a > 1] = 1
        # pil_im = Image.fromarray(np.cast[np.uint8](a[0] * 255))
        #
        # pil_im.show()

        return image_part, seg_part, image_part.shape



