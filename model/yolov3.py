import torch
import torch.nn as nn
import numpy as np

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x):
        return self.conv(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, kernel_size = 3):
        super().__init__()
        self.conv_pointwise = nn.Conv2d(in_channels, mid_channels, kernel_size = 1)
        self.conv = nn.Conv2d(mid_channels, in_channels, kernel_size = kernel_size)
    
    def forward(self, x):
        return nn.Sequential(self.conv(self.conv_pointwise(x)))

class DarkNet53(nn.Module):
    def __init__(self, n_channels, n_classes, in_width, in_height):
        super().__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.in_width = in_width
        self.in_height = in_height
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)

        self.block1 = ResBlock(64, 32)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)

        self.block2 = ResBlock(128, 64)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2)

        self.block3 = ResBlock(256, 128)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2)

        self.block4 = ResBlock(512, 256)

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=2)

        self.block5 = ResBlock(1024, 512)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.block1(x)
        x = self.conv3(x)
        x = self.block2(self.block2(x))
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



        
