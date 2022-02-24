import os,sys
#import numpy as np
import torch
import argparse
from glob import glob
import warnings

from eval.evaluator import Evaluator
warnings.filterwarnings("error")
from torch.utils.data.dataloader import DataLoader
import torchsummary as summary
from model.yolov3 import DarkNet53
from model.loss import celoss
from dataloader.yolodata import *
from train.trainer import Trainer
import torch.utils as utils
from tensorboardX import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOV3-PYTORCH")
    parser.add_argument('--gpu', dest='gpu', help="GPU to use",
                        default=0, type=int)
    parser.add_argument('--mode', dest='mode', help="train or test",
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg', help="model config path",
                        default=None, type=str)
    parser.add_argument('--checkpoint', dest='checkpoint', help = "the path of pre-trained model",
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args
    

def _collate_fn(batch):
    _box_tensor = list()
    _cls_tensor = list()
    _img_tensor = list()
    for b in batch:
        _box_tensor.append(b['bbox'])
        _cls_tensor.append(b['cls'])
        _img_tensor.append(b['img'])

    img_tensor = torch.stack(_img_tensor,0)
    #box_tensor = torch.stack(_box_tensor,0)
    #wh_tensor = torch.stack(_wh_tensor,1)
    #cls_tensor = torch.stack(_cls_tensor,1)
    #box_tensor = np.vstack(i['bbox'].reshape(1,i['bbox'].shape[0],i['bbox'].shape[1]) for i in batch)
    #cls_tensor = np.vstack(i['cls'].reshape(1,i['cls'].shape[0]) for i in batch)
    #wh_tensor =  torch.stack(_wh_tensor,1)
    return img_tensor, _box_tensor, _cls_tensor

def collate_fn(batch):
    return tuple(zip(*batch))

def train(cfg_param):
    train_data = Yolodata(train=True, transform=None, cfg_param = cfg_param)
    train_loader = DataLoader(train_data, batch_size=cfg_param['batch'], num_workers=4, pin_memory=True, drop_last=True, shuffle=True, collate_fn=collate_fn)

    model = DarkNet53(args.cfg, is_train=True)

    checkpoint = None
    if args.checkpoint is not None:
        print("load pretrained model ", args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])

    if args.gpu == 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    summary.summary(model, input_size=(3, 416, 416), device='cuda')
    x_test = torch.randn(2,3,416,416, requires_grad=True).to(device)
    torch.onnx.export(model,x_test,"yolov3.onnx",export_params=True, opset_version=11, input_names=['input'],output_names=['output'] )
    
    torch_writer = SummaryWriter("./output")
    trainer = Trainer(model, train_loader, device, cfg_param, checkpoint, torch_writer = torch_writer)
    
    trainer.run()

def test(cfg_param):
    print("test")

    eval_data = Yolodata(train=True, transform=None, cfg_param = cfg_param)
    eval_loader = DataLoader(eval_data, batch_size=1, num_workers=4, pin_memory=True, drop_last=True, shuffle=True)
    
    model = DarkNet53(args.cfg, is_train=False)
    if args.checkpoint is not None:
        print("load pretrained model ", args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])

    if args.gpu == 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if device == torch.device('cuda'):
        print("device is cuda")
    elif device == torch.device('cpu'):
        print('device is cpu')
    
    model = model.to(device)
    
    evaluator = Evaluator(model, eval_loader, device, cfg_param)
    
    evaluator.run()
    
    # summary.summary(backbone, input_size=(3, 416, 416), device='cuda')

    # for i, batch in enumerate(eval_loader):
    #     input_img = batch['image']
    #     b, c, h, w = input_img.size()
    #     #input_ = torch.cat([input_image.unsqueeze(1)])
    #     input_ = input_img.view(-1, c, h, w)
    #     out = backbone(input_)
        
        
if __name__ == "__main__":
    args = parse_args()
    cfg_data = parse_model_config(args.cfg)
    cfg_param = get_hyperparam(cfg_data)

    if args.mode == "train":
        train(cfg_param)
    elif args.mode == "test":
        test(cfg_param)
    else:
        print("Unknown mode error")

    print("finish")
