import os,sys
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#import numpy as np
import torch
import argparse
from glob import glob
import warnings

from eval.evaluator import Evaluator
#warnings.filterwarnings("error")
from torch.utils.data.dataloader import DataLoader
import torchsummary as summary
from model.yolov3 import DarkNet53
from model.loss import celoss
from dataloader.yolodata import *
from train.trainer import Trainer
from demo.demo import Demo
from dataloader.data_transforms import *
import torch.utils as utils
from tensorboardX import SummaryWriter
import pynvml

import onnx,onnxruntime

def get_memory_free_MiB(gpu_index):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.free // 1024 ** 2

def get_memory_total_MiB(gpu_index):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.total // 1024 ** 2

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOV3-PYTORCH")
    parser.add_argument("--gpus", type=int, nargs='+', default=[], help="List of device ids.")
    parser.add_argument('--mode', dest='mode', help="train or test",
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg', help="model config path",
                        default=None, type=str)
    parser.add_argument('--checkpoint', dest='checkpoint', help = "the path of checkpoint",
                        default=None, type=str)
    parser.add_argument('--pretrained', dest='pretrained', help = "the path of pre-trained model",
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

def collate_fn(batch):
    return tuple(zip(*batch))

def train(cfg_param = None, using_gpus = None):
    
    transforms = get_transformations(cfg_param, is_train = True)
    train_data = Yolodata(is_train=True, transform=transforms, cfg_param = cfg_param)
    train_loader = DataLoader(train_data, batch_size=cfg_param['batch'], num_workers=4, pin_memory=True, drop_last=True, shuffle=True, collate_fn=collate_fn)

    model = DarkNet53(args.cfg, is_train=True)
    
    if args.pretrained is not None:
        print("load pretrained model")
        model.load_darknet_weights(args.pretrained)
    
    #Set the device what you use, GPU or CPU
    for i in using_gpus:
        print("GPU total memory : {} free memory : {}".format(get_memory_total_MiB(i), get_memory_free_MiB(i)))
        if get_memory_free_MiB(i) / get_memory_total_MiB(i) < 0.5:
            print("Avaliable memory is {}%, GPU is already used now, Exit process".format(get_memory_free_MiB(i) / get_memory_total_MiB(i)))
            sys.exit(1)
    if len(using_gpus) == 1:
        device = torch.device("cuda:"+str(using_gpus[0]) if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(using_gpus[0])
        model = torch.nn.DataParallel(model, device_ids=using_gpus)
        model = model.cuda()
    elif len(using_gpus) == 0:
        print("Disable to use GPU. Exit process")
        device = torch.device("cpu")
        model = model.to(device)
    elif len(using_gpus) > 1:
        print("using_gpus : {}".format(using_gpus))
        device = torch.device("cuda:"+str(using_gpus[0]) if torch.cuda.is_available() else "cpu")
        model = torch.nn.DataParallel(model, device_ids=using_gpus)
        model = model.cuda()
        model.to(f'cuda:{model.device_ids[0]}')

    torch.backends.cudnn.benchmark = True

    #If checkpoint is existed, load the previous checkpoint.
    checkpoint = None
    if args.checkpoint is not None:
        print("load pretrained model ", args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        for key, value in checkpoint['model_state_dict'].copy().items():
            new_key = "module." + key
            checkpoint['model_state_dict'][new_key] = checkpoint['model_state_dict'].pop(key)
        model.load_state_dict(checkpoint['model_state_dict'])

    #Pre-check the model structure and size of parameters
    summary.summary(model, input_size=(3, cfg_param["in_width"], cfg_param["in_height"]), device='cuda' if device == torch.device('cuda') else 'cpu')
    
    #Setting the torch log directory to use tensorboard
    torch_writer = SummaryWriter("./output")
    
    if len(using_gpus) > 0:
        yolo_model = model.module
    else:
        yolo_model = model
    #Export Yolo model from pytorch to onnx format. yolov3.onnx
    x_test = torch.randn(2, 3, cfg_param["in_width"], cfg_param["in_height"], requires_grad=True).to(device)
    #torch.onnx.export(yolo_model, x_test, "yolov3.onnx", export_params=True, opset_version=11, input_names=['input'], output_names=['output'] )
    
    #Set trainer
    trainer = Trainer(yolo_model, train_loader, device, cfg_param, checkpoint, torch_writer = torch_writer)
    trainer.run()

def eval(cfg_param = None, using_gpus = None):
    print("evaluation")
    transforms = get_transformations(cfg_param, is_train = False)    
    eval_data = Yolodata(is_train = False, transform = transforms, cfg_param = cfg_param)
    eval_loader = DataLoader(eval_data, batch_size = 1, num_workers = 0, pin_memory = True, drop_last = False, shuffle = False, collate_fn=collate_fn)
    
    model = DarkNet53(args.cfg, is_train = False)

    if len(using_gpus) == 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if device == torch.device('cuda'):
        print("device is cuda")
    elif device == torch.device('cpu'):
        print('device is cpu')
    
    model = model.to(device)
    
    model.eval()
    
    torch.backends.cudnn.benchmark = True
    
    if args.checkpoint is not None:
        print("load pretrained model ", args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        
    #Export Yolo model from pytorch to onnx format. yolov3.onnx
    x_test = torch.randn(1, 3, cfg_param["in_width"], cfg_param["in_height"], requires_grad=True).to(device)
    torch.onnx.export(model, x_test, "yolov3.onnx", export_params=True, opset_version=11, input_names=['input'], output_names=['output'] )

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    ort_session = onnxruntime.InferenceSession("./yolov3.onnx")

    # ONNX 런타임에서 계산된 결과값
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x_test)}
    
    ort_outs = ort_session.run(None, ort_inputs)
    
    torch_outs = model(x_test)
    
    print("torch output : ", len(torch_outs), " ", torch_outs[0].shape)
    print("onnx out: ", len(ort_outs), ort_outs[0].shape)
    torch_np_outs = to_numpy(torch_outs[2])
    # ONNX 런타임과 PyTorch에서 연산된 결과값 비교
    #np.testing.assert_allclose(torch_np_outs, ort_outs[2], rtol=1e-03, atol=1e-05)

    evaluator = Evaluator(model, eval_data, eval_loader, device, cfg_param)
    
    evaluator.run()
    
def demo(cfg_param = None, using_gpus = None):
    print("demo")
    transforms = get_transformations(cfg_param, is_train = False)    
    data = Yolodata(is_train = False, transform = transforms, cfg_param = cfg_param)
    demo_loader = DataLoader(data, batch_size = 1, num_workers = 4, pin_memory = True, drop_last = False, shuffle = False)
    
    model = DarkNet53(args.cfg, is_train = False)
    if args.checkpoint is not None:
        print("load pretrained model ", args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])

    if len(using_gpus) == 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if device == torch.device('cuda'):
        print("device is cuda")
    elif device == torch.device('cpu'):
        print('device is cpu')
    
    model = model.to(device)
    
    torch.backends.cudnn.benchmark = True

    demo = Demo(model, data, demo_loader, device, cfg_param)
    
    demo.run()

        
if __name__ == "__main__":
    args = parse_args()
    cfg_data = parse_model_config(args.cfg)
    cfg_param = get_hyperparam(cfg_data)
    
    # multi-gpu
    print("GPUS : ", args.gpus)
    using_gpus = [int(g) for g in args.gpus]

    if args.mode == "train":
        train(cfg_param, using_gpus)
    elif args.mode == "eval":
        eval(cfg_param, using_gpus)
    elif args.mode == "demo":
        demo(cfg_param, using_gpus)
    else:
        print("Unknown mode error")

    print("finish")
