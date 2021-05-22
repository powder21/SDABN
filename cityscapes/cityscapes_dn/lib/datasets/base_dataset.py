# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
import random
import time
import torch
from torch.nn import functional as F
from torch.utils import data

class BaseDataset(data.Dataset):
    def __init__(self, 
                 ignore_label=-1, 
                 base_size=2048, 
                 crop_size=(512, 1024), 
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],
		 noise_level=50.0):

        self.base_size = base_size
        self.crop_size = crop_size
        self.ignore_label = ignore_label
        self.noise_level = noise_level

        self.mean = mean
        self.std = std
        self.scale_factor = scale_factor
        self.downsample_rate = 1./downsample_rate

        self.files = []
    
    def __len__(self):
        return len(self.files)

    def input_transform(self, image):
        image = image.astype(np.float32)[:, :, ::-1]  #bgr->rgb
        clean_image = image / 255.0  # [0,1]

        if 'train' in self.list_path:
            H, W, _ = np.shape(image)
            noise = (self.noise_level / 255.0) * np.random.randn(H, W, 3)
            noisy_image = clean_image + noise.astype(np.float32)
            noisy_image_seg = noisy_image - self.mean
            noisy_image_seg = np.cast[np.float32](noisy_image_seg / self.std)
            return noisy_image_seg, clean_image
        elif 'val' in self.list_path:
            return clean_image, clean_image
    
    def label_transform(self, label):
        return np.array(label).astype('int32')

    def pad_image(self, image, h, w, size, padvalue):
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=padvalue)
        
        return pad_image

    def rand_crop(self, image, label):
        h, w = image.shape[:-1]
        image = self.pad_image(image, h, w, self.crop_size,
                                (0.0, 0.0, 0.0))
        label = self.pad_image(label, h, w, self.crop_size,
                                (self.ignore_label,))
        
        new_h, new_w = label.shape
        x = random.randint(0, new_w - self.crop_size[1])
        y = random.randint(0, new_h - self.crop_size[0])
        image = image[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        label = label[y:y+self.crop_size[0], x:x+self.crop_size[1]]

        return image, label

    def center_crop(self, image, label):
        h, w = image.shape[:2]
        x = int(round((w - self.crop_size[1]) / 2.))
        y = int(round((h - self.crop_size[0]) / 2.))
        image = image[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        label = label[y:y+self.crop_size[0], x:x+self.crop_size[1]]

        return image, label
    
    def image_resize(self, image, long_size, label=None):
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            new_w = np.int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int(h * long_size / w + 0.5)
        
        image = cv2.resize(image, (new_w, new_h), 
                           interpolation = cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h), 
                           interpolation = cv2.INTER_NEAREST)
        else:
            return image
        
        return image, label

    def multi_scale_aug(self, image, label=None, 
            rand_scale=1, random_crop=True):
        long_size = np.int(self.base_size * rand_scale + 0.5)
        if label is not None:
            image, label = self.image_resize(image, long_size, label)
            if random_crop:
                image, label = self.rand_crop(image, label)
            return image, label
        else:
            image = self.image_resize(image, long_size)
            return image

    def gen_sample(self, image, label,
            random_crop=True, is_flip=True, center_crop_test=False):

        if center_crop_test:
            image, label = self.image_resize(image,
                                             self.base_size,
                                             label)
            image, label = self.center_crop(image, label)

        if random_crop:
            image, label = self.rand_crop(image, label)

        noisy_image_seg, clean_image = self.input_transform(image)

        if is_flip:
            flip = np.random.choice(2) * 2 - 1
            noisy_image_seg = noisy_image_seg[::flip, :, :]
            clean_image = clean_image[::flip, :, :]

            flip = np.random.choice(2) * 2 - 1
            noisy_image_seg = noisy_image_seg[:, ::flip, :]
            clean_image = clean_image[:, ::flip, :]

            r = np.random.choice([0, 1, 2, 3])
            if r:
                noisy_image_seg = np.rot90(noisy_image_seg, r)
                clean_image = np.rot90(clean_image, r)


        noisy_image_seg = noisy_image_seg.transpose((2, 0, 1))
        clean_image = clean_image.transpose((2, 0, 1))



        return noisy_image_seg, clean_image

    def inference(self, model, image, flip=False):
        size = image.size()
        pred = model(image)
        pred = F.upsample(input=pred, 
                            size=(size[-2], size[-1]), 
                            mode='bilinear')        
        if flip:
            flip_img = image.numpy()[:,:,:,::-1]
            flip_output = model(torch.from_numpy(flip_img.copy()))
            flip_output = F.upsample(input=flip_output, 
                            size=(size[-2], size[-1]), 
                            mode='bilinear')
            flip_pred = flip_output.cpu().numpy().copy()
            flip_pred = torch.from_numpy(flip_pred[:,:,:,::-1].copy()).cuda()
            pred += flip_pred
            pred = pred * 0.5
        return pred.exp()

    def multi_scale_inference(self, model, image, scales=[1], flip=False):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        device = torch.device("cuda:%d" % model.device_ids[0])
        image = image.numpy()[0].transpose((1,2,0)).copy()
        stride_h = np.int(self.crop_size[0] * 2.0 / 3.0)
        stride_w = np.int(self.crop_size[1] * 2.0 / 3.0)
        final_pred = torch.zeros([1, self.num_classes,
                                    ori_height,ori_width]).to(device)
        padvalue = -1.0  * np.array(self.mean) / np.array(self.std)
        for scale in scales:
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           random_crop=False)
            height, width = new_img.shape[:-1]
                
            if max(height, width) <= np.min(self.crop_size):
                new_img = self.pad_image(new_img, height, width, 
                                    self.crop_size, padvalue)
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = self.inference(model, new_img, flip)
                preds = preds[:, :, 0:height, 0:width]
            else:
                if height < self.crop_size[0] or width < self.crop_size[1]:
                    new_img = self.pad_image(new_img, height, width, 
                                        self.crop_size, padvalue)
                new_h, new_w = new_img.shape[:-1]
                rows = np.int(np.ceil(1.0 * (new_h - 
                                self.crop_size[0]) / stride_h)) + 1
                cols = np.int(np.ceil(1.0 * (new_w - 
                                self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([1, self.num_classes,
                                           new_h,new_w]).to(device)
                count = torch.zeros([1,1, new_h, new_w]).to(device)

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        if h1 == new_h or w1 == new_w:
                            crop_img = self.pad_image(crop_img, 
                                                      h1-h0, 
                                                      w1-w0, 
                                                      self.crop_size, 
                                                      padvalue)
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(model, crop_img, flip)

                        preds[:,:,h0:h1,w0:w1] += pred[:,:, 0:h1-h0, 0:w1-w0]
                        count[:,:,h0:h1,w0:w1] += 1
                preds = preds / count
                preds = preds[:,:,:height,:width]
            preds = F.upsample(preds, (ori_height, ori_width), 
                                   mode='bilinear')
            final_pred += preds
        return final_pred
