from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
# from scipy.misc import imshow


class MsFeat(nn.Module):
    def __init__(self, in_channels):
        super(MsFeat, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 8, 3, padding=1, dilation=1, bias=True), nn.ReLU())
        init.kaiming_normal_(self.conv1[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.conv1[0].bias, 0.0)

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, 8, 3, padding=2, dilation=2, bias=True), nn.ReLU())
        init.kaiming_normal_(self.conv2[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.conv2[0].bias, 0.0)

        self.conv3 = nn.Sequential(nn.Conv2d(8, 8, 3, padding=1, dilation=1, bias=True), nn.ReLU())
        init.kaiming_normal_(self.conv3[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.conv3[0].bias, 0.0)

        self.conv4 = nn.Sequential(nn.Conv2d(8, 8, 3, padding=2, dilation=2, bias=True), nn.ReLU())
        init.kaiming_normal_(self.conv4[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.conv4[0].bias, 0.0)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(inputs)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv1)
        return torch.cat((conv1, conv2, conv3, conv4), 1)


class SegFeat(nn.Module):
    def __init__(self, category_number):
        super(SegFeat, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(category_number, 16, 1), nn.LeakyReLU(0.1, True),
                                  nn.Conv2d(16, 32, 1), nn.LeakyReLU(0.1, True),
                                  nn.Conv2d(32, 64, 1), nn.LeakyReLU(0.1, True),
                                  nn.Conv2d(64, 32, 1), nn.LeakyReLU(0.1, True),
                                  nn.Conv2d(32, 16, 1))

    def forward(self, seg_onehot):
        return self.conv(seg_onehot)


class SFT(nn.Module):
    def __init__(self, feature_channel):
        super(SFT, self).__init__()
        self.scale = nn.Sequential(nn.Conv2d(16, 32, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(32, feature_channel, 1))
        self.shift = nn.Sequential(nn.Conv2d(16, 32, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(32, feature_channel, 1))

    def forward(self, feature_in, condition):
        scale = self.scale(condition)
        shift = self.shift(condition)
        feature_out = torch.mul(feature_in, (scale+1)) + shift
        return feature_out


class Block(nn.Module):
    def __init__(self, in_channels):
        super(Block, self).__init__()
        self.input_sft = SFT(feature_channel=in_channels)

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 32, 1, padding=0, dilation=1, bias=True), nn.ReLU(True))
        init.kaiming_normal_(self.conv1[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.conv1[0].bias, 0.0)

        self.feat1 = nn.Sequential(nn.Conv2d(32, 8, 3, padding=1, dilation=1, bias=True), nn.ReLU(True))
        init.kaiming_normal_(self.feat1[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.feat1[0].bias, 0.0)

        self.feat15 = nn.Sequential(nn.Conv2d(8, 4, 3, padding=2, dilation=2, bias=True), nn.ReLU(True))
        init.kaiming_normal_(self.feat15[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.feat15[0].bias, 0.0)

        self.feat2 = nn.Sequential(nn.Conv2d(32, 8, 3, padding=2, dilation=2, bias=True), nn.ReLU(True))
        init.kaiming_normal_(self.feat2[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.feat2[0].bias, 0.0)

        self.feat25 = nn.Sequential(nn.Conv2d(8, 4, 3, padding=1, dilation=1, bias=True), nn.ReLU(True))
        init.kaiming_normal_(self.feat25[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.feat25[0].bias, 0.0)

        self.feat = nn.Sequential(nn.Conv2d(24, 8, 1, padding=0, dilation=1, bias=True), nn.ReLU(True))
        init.kaiming_normal_(self.feat[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.feat[0].bias, 0.0)

    def forward(self, inputs, condition):
        input_sft = self.input_sft(inputs, condition)
        conv1 = self.conv1(input_sft)
        feat1 = self.feat1(conv1)
        feat15 = self.feat15(feat1)
        feat2 = self.feat2(conv1)
        feat25 = self.feat25(feat2)
        feat = self.feat(torch.cat((feat1, feat15, feat2, feat25), 1))
        return torch.cat((inputs, feat), 1)


class DeepBoosting(nn.Module):
    def __init__(self, in_channels=3, category_number=9):
        super(DeepBoosting, self).__init__()
        self.msfeat = MsFeat(in_channels)  # C32
        self.segfeat = SegFeat(category_number)  # C32
        print('ddfn-sft seg channel is: ', category_number)
        print('ddfn-sft In channel: ', in_channels)
        self.dfus_block0 = Block(32)
        self.dfus_block1 = Block(40)
        self.dfus_block2 = Block(48)
        self.dfus_block3 = Block(56)
        self.dfus_block4 = Block(64)
        self.dfus_block5 = Block(72)
        self.dfus_block6 = Block(80)
        self.dfus_block7 = Block(88)
        self.sft = SFT(feature_channel=96)
        self.convr = nn.Conv2d(96, in_channels, 1, padding=0, dilation=1, bias=False)
        print('ddfn-sft Out channel: ', in_channels)
        init.normal_(self.convr.weight, mean=0.0, std=0.001)

    def forward(self, input_image):
        noisy_image = input_image[:, 0:3, :, :]
        seg_prob = input_image[:, 3:, :, :] # C8

        msfeat = self.msfeat(noisy_image)  # C32
        segfeat = self.segfeat(seg_prob)  # C32
        b0 = self.dfus_block0(msfeat, segfeat)
        b1 = self.dfus_block1(b0, segfeat)
        b2 = self.dfus_block2(b1, segfeat)
        b3 = self.dfus_block3(b2, segfeat)
        b4 = self.dfus_block4(b3, segfeat)
        b5 = self.dfus_block5(b4, segfeat)
        b6 = self.dfus_block6(b5, segfeat)
        b7 = self.dfus_block7(b6, segfeat)
        b7 = self.sft(b7, segfeat)
        convr = self.convr(b7)
        return convr


if __name__ == '__main__':
    """ example of weight sharing """
    # self.convs1_siamese = Conv3x3Stack(1, 12, negative_slope)
    # self.convs1_siamese[0].weight = self.convs1[0].weight

    import numpy as np

    model = DeepBoosting().to('cuda:0')
    x = torch.tensor(np.random.random((8, 1, 268, 268)).astype(np.float32)).to('cuda:0')
    out = model(x)  # (8, 2, 268, 268)

    print('break')
