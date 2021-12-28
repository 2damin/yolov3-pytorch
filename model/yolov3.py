import torch
import torch.nn as nn
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, kernel_size = 3):
        super().__init__()
        self.conv_pointwise = nn.Conv2d(in_channels, mid_channels, kernel_size = 1)
        self.bn_pt = nn.BatchNorm2d(mid_channels)
        self.act = nn.LeakyReLU()
        self.conv = nn.Conv2d(mid_channels, in_channels, kernel_size = kernel_size, padding=1)
        self.bn_conv = nn.BatchNorm2d(in_channels)
        self.model = nn.Sequential(self.conv_pointwise,
                                   self.bn_pt,
                                   self.act,
                                   self.conv,
                                   self.bn_conv,
                                   self.act)
    
    def forward(self, x):
        return self.model(x)

class DarkNet53(nn.Module):
    def __init__(self, n_channels, n_classes, in_width, in_height):
        super().__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.in_width = in_width
        self.in_height = in_height
        
        self.conv1 = ConvBlock(in_channels = 3, out_channels = 32, kernel_size=3, stride = 1, padding = 1)

        self.conv2 = ConvBlock(in_channels = 32, out_channels = 64, kernel_size=3, stride = 2, padding = 1)

        self.block1 = ResBlock(64, 32)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.block2 = ResBlock(128, 64)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.block3 = ResBlock(256, 128)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.block4 = ResBlock(512, 256)

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)

        self.block5 = ResBlock(1024, 512)
    
    def forward(self, x):
        x = self.conv1(x.float())
        x = self.conv2(x)
        x = self.block1(x)
        x = self.conv3(x)
        for i in range(2):
            x = self.block2(x)
        x = self.conv4(x)
        for i in range(8):
            x = self.block3(x)
        x = self.conv5(x)
        for i in range(8):
            x = self.block4(x)
        x = self.conv6(x)
        for i in range(4):
            x = self.block5(x)
        return x



        
