import os,sys
import numpy as np
import torch
import argparse
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOV3-PYTORCH")
    parser.add_argument('--gpu', dest='gpu', help="GPU to use",
                        default=0, type=int)
    parser.add_argument('--mode', dest='mode', help="train or test",
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


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "test":
        test()
    else:
        print("Unknown mode error")

    print("finish")
