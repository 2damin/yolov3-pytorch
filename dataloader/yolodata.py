import torch
from torch.utils.data import Dataset
import os,sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import numpy as np
from util.tools import *
from . import data_transforms
import cv2
import torchvision

class Yolodata(Dataset):
    file_dir = ""
    anno_dir = ""
    file_txt = ""
    train_dir = "/damin/data/GODTrain210611"
    train_txt = "tstld.txt"
    valid_dir = "/damin/data/GODTest_SVKPI_3000km"
    valid_txt = "all.txt"
    class_str = ['ts_circle', 'ts_triangle', 'ts_rectangle', 'tl_car']
    num_class = None
    img_data = []
    def __init__(self, is_train=True, transform=None, cfg_param=None):
        super(Yolodata, self).__init__()
        self.is_train = is_train
        self.transform = transform
        self.num_class = cfg_param['class']
        if self.is_train:
            self.file_dir = self.train_dir+"/JPEGImages/"
            self.file_txt = self.train_dir+"/ImageSets/"+self.train_txt
            self.anno_dir = self.train_dir+"/Annotations/"
        else:
            self.file_dir = self.valid_dir+"/JPEGImages/"
            self.file_txt = self.valid_dir+"/ImageSets/"+self.valid_txt
            self.anno_dir = self.valid_dir+"/Annotation_yolo/"

        img_names = []
        img_data = []
        with open(self.file_txt, 'r', encoding='UTF-8', errors='ignore') as f:
            img_names = [ i.replace("\n", "") for i in f.readlines()]
        for i in img_names:
            if os.path.exists(self.file_dir + i + ".jpg"):
                img_data.append(i+".jpg")
            elif os.path.exists(self.file_dir + i + ".JPG"):
                img_data.append(i+".JPG")
            elif os.path.exists(self.file_dir + i + ".png"):
                img_data.append(i+".png")
            elif os.path.exists(self.file_dir + i + ".PNG"):
                img_data.append(i+".PNG")
        print("data len : {}".format(len(img_data)))
        self.img_data = img_data
        #self.resize = tf.Resize([cfg_param['in_width'],cfg_param['in_height']])
    
    def __getitem__(self, index):
        img_path = self.file_dir + self.img_data[index]

        with open(img_path, 'rb') as f:
            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_origin_h, img_origin_w = img.shape[:2]

        #if anno_dir is didnt exist, Test dataset
        if os.path.isdir(self.anno_dir):
            txt_name = self.img_data[index]
            for ext in ['.png','.PNG','.jpg','.JPG']:
                txt_name = txt_name.replace(ext, ".txt")
            anno_path = self.anno_dir + txt_name
            
            bbox = []
            with open(anno_path, 'r') as f:
                for line in f.readlines():
                    line = line.replace("\n","")
                    gt_data = [ l for l in line.split(" ")]
                    #skip when no data
                    if len(gt_data) < 5:
                        continue
                    #ignore very tiny GTs
                    cx, cy, w, h = float(gt_data[1]), float(gt_data[2]), float(gt_data[3]), float(gt_data[4])
                    #trunc = float(gt_data[5]) if len(gt_data) > 5 else 0
                    #occ = float(gt_data[6]) if len(gt_data) > 6 else 0
                    # if w <= 20 or float(gt_data[7]) - float(gt_data[5]) <= 20:
                    #     continue
                    bbox.append([float(gt_data[0]), cx, cy, w, h])

            #Change gt_box type
            bbox = np.array(bbox)
            
            #skip empty target
            empty_target = False
            if bbox.shape[0] == 0:
                empty_target = True
                bbox = np.array([[0,0,0,0,0]])

            #data augmentation
            img, bbox = self.transform((img, bbox))

            if not empty_target:
                batch_idx = torch.zeros(bbox.shape[0])
                #batch_idx, cls, x, y, w, h
                target_data = torch.cat((batch_idx.view(-1,1),bbox),dim=1)
            else:
                return
            return img, target_data, anno_path
        else:
            img, _ = self.transform((img, bbox))
            return img, None, None


    def __len__(self):
        return len(self.img_data)
