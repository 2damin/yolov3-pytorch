import os,sys
#import numpy as np
import torch
import argparse
import cv2
from glob import glob
from torch.utils.data.dataloader import DataLoader
import torchsummary as summary
from model.yolov3 import DarkNet53
from dataloader.yolodata import Yolodata 

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOV3-PYTORCH")
    parser.add_argument('--gpu', dest='gpu', help="GPU to use",
                        default=0, type=int)
    parser.add_argument('--mode', dest='mode', help="train or test",
                        default=None, type=str)
    parser.add_argument('--input_dir', dest='input_dir', help="input directory",
                        default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args
    

def train():
    print("train")

def test():
    print("test")

    train_data = Yolodata(train=True, transform=None)
    train_loader = DataLoader(train_data, batch_size=1, num_workers=1, pin_memory=True, drop_last=True, shuffle=True)

    backbone = DarkNet53(3,5,512,512)
    backbone = backbone.cpu()
    summary.summary(backbone, input_size=(3, 512, 512), device='cpu')
    for i, batch in enumerate(train_loader):
        input_img = batch['image']
        b, c, h, w = input_img.size()
        #input_ = torch.cat([input_image.unsqueeze(1)])
        input_ = input_img.view(-1, c, h, w)
        out = backbone.forward(input_)

if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "test":
        test()
    else:
        print("Unknown mode error")

    print("finish")
