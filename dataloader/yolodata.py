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

class Yolodata(Dataset):
    file_dir = ""
    anno_dir = ""
    file_txt = ""
    train_dir = "C:\\data\\kitti_dataset\\training"
    train_txt = "train.txt"
    valid_dir = "C:\\data\\kitti_dataset\\eval"
    valid_txt = "eval.txt"
    class_str = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']
    num_class = None
    img_data = []
    def __init__(self, is_train=True, transform=None, cfg_param=None):
        super(Yolodata, self).__init__()
        self.is_train = is_train
        self.transform = transform
        self.num_class = cfg_param['class']
        if self.is_train:
            self.file_dir = self.train_dir+"\\Images\\"
            self.file_txt = self.train_dir+"\\ImageSets\\"+self.train_txt
            self.anno_dir = self.train_dir+"\\Annotation\\"
        else:
            self.file_dir = self.valid_dir+"\\Images\\"
            self.file_txt = self.valid_dir+"\\ImageSets\\"+self.valid_txt
            self.anno_dir = self.valid_dir+"\\Annotation\\"

        img_names = []
        img_data = []
        with open(self.file_txt, 'r', encoding='UTF-8', errors='ignore') as f:
            img_names = [ i.replace("\n", "") for i in f.readlines()]
        for i in img_names:
            if os.path.exists(self.file_dir + i):
                img_data.append(i)
        print("data len : {}".format(len(img_data)))
        self.img_data = img_data
        #self.resize = tf.Resize([cfg_param['in_width'],cfg_param['in_height']])
    
    def __getitem__(self, index):
        img_path = self.file_dir + self.img_data[index]

        with open(img_path, 'rb') as f:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_origin_h, img_origin_w = img.shape[:2]

        #if anno_dir is didnt exist, Test dataset
        if os.path.isdir(self.anno_dir):
            anno_path = self.anno_dir + self.img_data[index].replace(".png", ".txt")
            
            bbox = []
            cls = []
            occ = []
            trunc = []
            with open(anno_path, 'r') as f:
                for line in f.readlines():
                    gt_data = [ l for l in line.split(" ")]
                    #ignore very tiny GTs
                    if float(gt_data[6]) - float(gt_data[4]) <= 20 or float(gt_data[7]) - float(gt_data[5]) <= 20:
                        continue
                    cls.append(self.class_str.index(gt_data[0]))
                    bbox.append([float(i) for i in gt_data[4:8]]) #[xmin, ymin, xmax, ymax]
                    trunc.append(float(gt_data[1]))
                    occ.append(int(gt_data[2]))

            #Change gt_box type
            bbox = np.array(bbox)

            sample = {}
            sample['image'] = img
            sample['label'] = bbox
            #data augmentation
            sample = self.transform(sample)

            target = {}
            if bbox.size == 0:
                target['bbox'] = None #torch.FloatTensor(np.zeros((1,4), np.float32))
                target['cls'] = None #torch.tensor(np.zeros(1, np.int32),dtype=torch.int64)
                target['trunc'] = None
                target['occ'] = None
                target['path'] = anno_path
            else:
                #target['bbox'] = torch.FloatTensor(np.array(bbox))
                target['bbox'] = sample['label']
                target['cls'] = torch.tensor(np.array(cls),dtype=torch.int64)
                target['trunc'] = torch.tensor(np.array(trunc),dtype=torch.float32)
                target['occ'] = torch.tensor(np.array(occ),dtype=torch.int64)
                target['path'] = anno_path

            #return torch.div(torch.tensor(np.transpose(np.array(img, dtype=float),(2,0,1)),dtype=torch.float32),255), target
            return sample['image'], target
        else:
            sample = {}
            sample['image'] = img
            sample['label'] = []
            sample = self.transform(sample)
            return sample['image'], {}


    def __len__(self):
        return len(self.img_data)
