import torch
from torch.utils.data import Dataset
from torchvision import transforms as tf
import os,sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import numpy as np
from util.tools import *

class Yolodata(Dataset):
    file_dir = ""
    anno_dir = ""
    file_txt = ""
    train_dir = "C:\\data\\kitti_dataset\\training"
    train_txt = "train.txt"
    valid_dir = "C:\\data\\kitti_dataset\\testing"
    valid_txt = "valid.txt"
    class_str = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']
    num_class = 8
    img_data = []
    def __init__(self, train=True, transform=None, cfg_param=None):
        super(Yolodata, self).__init__()
        self.train = train
        self.transform = transform

        if self.train:
            self.file_dir = self.train_dir+"\\Images\\"
            self.file_txt = self.train_dir+"\\ImageSets\\"+self.train_txt
            self.anno_dir = self.train_dir+"\\Annotation\\"
        else:
            self.file_dir = self.valid_dir+"\\Images\\"
            self.file_txt = self.valid_dir+"\\ImageSets\\"+self.valid_txt  

        img_names = []
        img_data = []
        with open(self.file_txt, 'r', encoding='UTF-8', errors='ignore') as f:
            img_names = [ i.replace("\n", "") for i in f.readlines()]
        for i in img_names:
            if os.path.exists(self.file_dir + i):
                img_data.append(i)
        print("data len : {}".format(len(img_data)))
        self.img_data = img_data
        self.resize = tf.Resize([cfg_param['in_width'],cfg_param['in_height']])
    
    def __getitem__(self, index):
        img_path = self.file_dir + self.img_data[index]

        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img_w, img_h = img.width, img.height
            img = self.resize(img)

        if self.train:
            anno_path = self.anno_dir + self.img_data[index].replace(".png", ".txt")
            bbox = []
            cls = []

            with open(anno_path, 'r') as f:
                for line in f.readlines():
                    gt_data = [ l for l in line.split(" ")]
                    # if gt_data[0] == "DontCare":
                    #     continue
                    cls.append(self.class_str.index(gt_data[0]))
                    bbox.append([float(i) for i in gt_data[4:8]]) #[xmin, ymin, xmax, ymax]

            #Change gt_box format (minmax -> cxcywh)
            bbox = np.array(bbox)
            for i, box in enumerate(bbox):
                box[0] = box[0] / img_w
                box[1] = box[1] / img_h
                box[2] = box[2] / img_w
                box[3] = box[3] / img_h
                #resizeBox(box, [img_w, img_h], [img.width, img.height])
                minmax2cxcy(box)

            target = {}
            if bbox.size == 0:
                target['bbox'] = torch.FloatTensor(np.zeros((1,4), np.float32))
                target['cls'] = torch.tensor(np.zeros(1, np.int32),dtype=torch.int64)
            else:
                target['bbox'] = torch.FloatTensor(np.array(bbox))
                target['cls'] = torch.tensor(np.array(cls),dtype=torch.int64)

            #out = {'img':torch.div(torch.tensor(np.transpose(np.array(img),(2,0,1)),dtype=torch.float32),255), 'target':target}
            return torch.div(torch.tensor(np.transpose(np.array(img, dtype=float),(2,0,1)),dtype=torch.float32),255), target
        else:
            return torch.div(torch.tensor(np.transpose(np.array(img),(2,0,1)),dtype=torch.float32),255), {}


    def __len__(self):
        return len(self.img_data)
