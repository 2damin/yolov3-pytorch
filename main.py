import os,sys
#import numpy as np
import torch
import argparse
import cv2
from glob import glob
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

    backbone = DarkNet53(3,5,512,512)
    backbone = backbone.cpu()

    #summary.summary(backbone, input_size=(3, 1280, 704), device='cpu')


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "test":
        test()
    else:
        print("Unknown mode error")

    print("finish")
