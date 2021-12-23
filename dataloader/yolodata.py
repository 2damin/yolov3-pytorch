from torch.utils.data import Dataset
from torchvision import transforms as tf
import os,sys
from PIL import Image
import numpy as np

class Yolodata(Dataset):
    file_dir = ""
    file_txt = ""
    train_dir = "C:\\data\\kitti_dataset\\training"
    train_txt = "train.txt"
    valid_dir = "./data/valid"
    valid_txt = "valid.txt"
    img_data = []
    def __init__(self, train=True, transform=None):
        super(Yolodata, self).__init__()
        self.train = train
        self.transform = transform

        if self.train:
            self.file_dir = self.train_dir+"\\Images\\"
            self.file_txt = self.train_dir+"\\ImageSets\\"+self.train_txt
        else:
            self.file_dir = self.train_dir+"\\Images\\"
            self.file_txt = self.train_dir+"\\ImageSets\\"+self.train_txt  

        img_names = []
        img_data = []
        with open(self.file_txt, 'r', encoding='UTF-8', errors='ignore') as f:
            img_names = [ i.replace("\n", "") for i in f.readlines()]
        for i in img_names:
            if os.path.exists(self.file_dir + i):
                img_data.append(self.file_dir + i)
        print("data len : {}".format(len(img_data)))

        self.img_data = img_data

        self.resize = tf.Resize([256,256])
    
    def __getitem__(self, index):
        img_path = self.img_data[index]
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = self.resize(img)
        out = {'image':np.transpose(np.array(img, dtype=np.float)/255,(2,0,1))}
        return out

    def __len__(self):
        return len(self.img_data)

        