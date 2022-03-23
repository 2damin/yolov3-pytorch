import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.transforms.functional import pad
import time,sys
from util.tools import *

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, name, batchnorm, act='leaky'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        if act == 'leaky':
            self.act = nn.LeakyReLU()
        elif act == 'relu':
            self.act = nn.ReLU()
        
        self.module = nn.Sequential()
        self.module.add_module(name+'_conv', self.conv)
        if batchnorm == 1:
            self.module.add_module(name+"_bn", self.bn)
        if act != 'linear':
            self.module.add_module(name+"_act", self.act)

    def forward(self, x):
        return self.module(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, kernel_size = 3):
        super().__init__()
        self.conv_pointwise = nn.Conv2d(in_channels, mid_channels, kernel_size = 1)
        self.bn_pt = nn.BatchNorm2d(mid_channels)
        self.act = nn.LeakyReLU()
        self.conv = nn.Conv2d(mid_channels, in_channels, kernel_size = kernel_size, padding=1, stride=1)
        self.bn_conv = nn.BatchNorm2d(in_channels)
        self.module = nn.Sequential(self.conv_pointwise,
                                   self.bn_pt,
                                   self.act,
                                   self.conv,
                                   self.bn_conv,
                                   self.act)
    
    def forward(self, x):
        return x + self.module(x)

class ConvUp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, up_ratio):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU()        
        self.upsample = nn.Upsample(scale_factor=up_ratio, mode='nearest')
        self.module = nn.Sequential(self.conv,
                                   self.bn,
                                   self.act,
                                   self.upsample)
    def forward(self, x):
        return self.module(x)

# class Upsample(nn.Module):
#     def __init__(self, size, mode="nearest"):
#         self.size = size
#         self.mode = mode
#     def forward(self, x):
#         x = F.interpolate(x, scale_factor = self.size, mode = self.mode)
#         return x

class YoloLayer(nn.Module):
    def __init__(self, layer_idx, layer_info, in_channel, in_width, in_height, is_train):
        super(YoloLayer, self).__init__()
        self.n_classes = int(layer_info['classes'])
        self.ignore_thresh = float(layer_info['ignore_thresh'])
        self.box_attr = self.n_classes + 5
        mask_idxes = [int(x) for x in layer_info["mask"].split(",")]
        anchor_all = [int(x) for x in layer_info["anchors"].split(",")]
        anchor_all = [(anchor_all[i],anchor_all[i+1]) for i in range(0,len(anchor_all),2)]
        self.anchor = torch.tensor([anchor_all[x] for x in mask_idxes])
        self.in_width = in_width
        self.in_height = in_height
        self.training = is_train
        self.stride = None
        self.lw = None
        self.lh = None
        
    def forward(self, x):
        #batch, num_anchor, x_height, x_width, num_attributes
        self.lw, self.lh = x.shape[3], x.shape[2]
        self.anchor = self.anchor.to(x.device)
        self.stride = torch.tensor([self.in_width // self.lw, self.in_height // self.lh]).to(x.device)
        x = x.view(-1,self.anchor.shape[0],self.box_attr,self.lh,self.lw).permute(0,1,3,4,2).contiguous()
        
        anchor_grid = self.anchor.view(1,-1,1,1,2).to(x.device)

        if not self.training:
            grids = self._make_grid(self.lw, self.lh).to(x.device)
            # Get outputs
            x[...,0:2] = (torch.sigmoid(x[...,0:2]) + grids) * self.stride #center xy
            x[...,2:4] = torch.exp(x[...,2:4]) * anchor_grid     # Width Height
            x[...,4:] = torch.sigmoid(x[...,4:])       # Conf, Class
            x = x.view(x.shape[0], -1, self.box_attr)
        return x

    def _make_grid(self, nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)]) #, indexing='ij'
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

def make_conv_layer(layer_idx, modules, layer_info, in_channel):
    filters = int(layer_info['filters'])
    size = int(layer_info['size'])
    stride = int(layer_info['stride'])
    pad = (size - 1) // 2
    #modules = nn.Sequential()
    modules.add_module('layer_'+str(layer_idx)+'_conv',
                      nn.Conv2d(in_channel,
                                filters,
                                size,
                                stride,
                                pad))

    if layer_info['batch_normalize'] == '1':
        modules.add_module('layer_'+str(layer_idx)+'_bn',
                          nn.BatchNorm2d(filters))
    if layer_info['activation'] == 'leaky':
        modules.add_module('layer_'+str(layer_idx)+'_act',
                          nn.LeakyReLU())
    elif layer_info['activation'] == 'relu':
        modules.add_module('layer_'+str(layer_idx)+'_act',
                          nn.ReLU())
    #return modules

def make_shortcut_layer(modules, layer_idx):
    modules.add_module('layer_'+str(layer_idx)+'_shortcut', nn.Sequential())

class DarkNet53(nn.Module):
    def __init__(self, cfg, param):
        super().__init__()
        self.batch = int(param['batch'])
        self.in_channels = int(param['in_channels'])
        self.in_width = int(param['in_width'])
        self.in_height = int(param['in_height'])
        self.n_classes = int(param['class'])
        self.module_cfg = parse_model_config(cfg)
        self.module_list = self.set_layer(self.module_cfg)
        self.yolo_layers = [layer[0] for layer in self.module_list if isinstance(layer[0], YoloLayer)]
        self.box_per_anchor = 3
        self.fpn_grid_size = [self.in_width // 32, self.in_height // 32, self.in_width // 16, self.in_height // 16, self.in_width // 8, self.in_height // 8]
        self.stride = [self.get_grid_wh(j) for j in range(3)]

        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)
        
        self.initialize_weights()
        
        self.softmax = nn.Softmax(dim=1)
        
        # self.conv1 = ConvBlock(in_channels = 3, out_channels = 32, kernel_size=3, stride = 1, padding = 1, name="layer1", act='leakyrelu')

        # self.conv2 = ConvBlock(in_channels = 32, out_channels = 64, kernel_size=3, stride = 2, padding = 1, name="layer2", act='leakyrelu')

        # self.resblock1 = ResBlock(64, 32)

        # self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        # self.resblock2 = ResBlock(128, 64)

        # self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        # self.resblock3 = ResBlock(256, 128)

        # self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        # self.resblock4 = ResBlock(512, 256)

        # self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)

        # self.resblock5 = ResBlock(1024, 512)
        
        # self.conv7 = nn.Sequential(ConvBlock(in_channels = 1024, out_channels = 512, kernel_size=1, stride=1, padding=0, name="layer12_0", act='leakyrelu'),
        #                            ConvBlock(in_channels = 512, out_channels = 1024, kernel_size=3, stride=1, padding=1, name="layer12_1", act='leakyrelu'),
        #                            ConvBlock(in_channels = 1024, out_channels = 512, kernel_size=1, stride=1, padding=0, name="layer12_2", act='leakyrelu'),
        #                            ConvBlock(in_channels = 512, out_channels = 1024, kernel_size=3, stride=1, padding=1, name="layer12_3", act='leakyrelu'),
        #                            ConvBlock(in_channels = 1024, out_channels = 512, kernel_size=1, stride=1, padding=0, name="layer12_4", act='leakyrelu'))
        # #FPN1
        # self.fpn1_a = ConvUp(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, up_ratio=2)
        # self.fpn1_b = nn.Sequential(ConvBlock(in_channels = 384, out_channels = 128, kernel_size=1, stride=1, padding=0, name="fpn1_b_0", act='leakyrelu'),
        #                             ConvBlock(in_channels = 128, out_channels = 256, kernel_size=3, stride=1, padding=1, name="fpn1_b_1", act='leakyrelu'),
        #                             ConvBlock(in_channels = 256, out_channels = 128, kernel_size=1, stride=1, padding=0, name="fpn1_b_2", act='leakyrelu'),
        #                             ConvBlock(in_channels = 128, out_channels = 256, kernel_size=3, stride=1, padding=1, name="fpn1_b_3", act='leakyrelu'),
        #                             ConvBlock(in_channels = 256, out_channels = 128, kernel_size=1, stride=1, padding=0, name="fpn1_b_4", act='leakyrelu'),
        #                             ConvBlock(in_channels = 128, out_channels = 256, kernel_size=3, stride=1, padding=1, name="fpn1_b_5", act='leakyrelu'))
        # self.fpn1_c = ConvBlock(in_channels=256, out_channels=self.output_channels, kernel_size=1, stride=1, padding=0, name="fpn1_c", act='linear')
        # #FPN2
        # self.fpn2_a = ConvUp(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, up_ratio=2)
        # self.fpn2_b = nn.Sequential(ConvBlock(in_channels = 768, out_channels = 256, kernel_size=1, stride=1, padding=0, name="fpn2_b_0", act='leakyrelu'),
        #                             ConvBlock(in_channels = 256, out_channels = 512, kernel_size=3, stride=1, padding=1, name="fpn2_b_1", act='leakyrelu'),
        #                             ConvBlock(in_channels = 512, out_channels = 256, kernel_size=1, stride=1, padding=0, name="fpn2_b_2", act='leakyrelu'),
        #                             ConvBlock(in_channels = 256, out_channels = 512, kernel_size=3, stride=1, padding=1, name="fpn2_b_3", act='leakyrelu'),
        #                             ConvBlock(in_channels = 512, out_channels = 256, kernel_size=1, stride=1, padding=0, name="fpn2_b_4", act='leakyrelu'))
        # self.fpn2_c = nn.Sequential(ConvBlock(in_channels = 256, out_channels = 512, kernel_size=3, stride=1, padding=1, name="fpn2_c_0", act='leakyrelu'),
        #                             ConvBlock(in_channels=512, out_channels=self.output_channels, kernel_size=1, stride=1, padding=0, name="fpn2_c_1", act='linear'))
        # #FPN3
        # self.fpn3 = nn.Sequential(ConvBlock(in_channels = 512, out_channels = 1024, kernel_size=3, stride=1, padding=1, name="fpn3_a", act='leakyrelu'),
        #                           ConvBlock(in_channels=1024, out_channels=self.output_channels, kernel_size=1, stride=1, padding=0, name='fpn3_b', act='linear'))
    
    def set_layer(self, layer_info):
        module_list = nn.ModuleList()
        in_channels = [self.in_channels]
        for layer_idx, info in enumerate(layer_info):
            print(layer_idx, info['type'])
            modules = nn.Sequential()
            if info['type'] == "convolutional":
                make_conv_layer(layer_idx, modules, info, in_channels[-1])
                in_channels.append(int(info['filters']))
            elif info['type'] == 'shortcut':
                make_shortcut_layer(modules, layer_idx)
                in_channels.append(in_channels[-1])
            elif info['type'] == 'route':
                modules.add_module('layer_'+str(layer_idx)+'_route', nn.Sequential())
                layers = [int(y) for y in info["layers"].split(",")]
                if len(layers) == 1:
                    in_channels.append(in_channels[layers[0]])
                elif len(layers) == 2:
                    in_channels.append(in_channels[layers[0]] + in_channels[layers[1]])
            elif info['type'] == 'upsample':
                modules.add_module('layer_'+str(layer_idx)+'_upsample',
                                       nn.Upsample(scale_factor=int(info['stride']), mode='nearest'))
                in_channels.append(in_channels[-1])
            elif info['type'] == 'yolo':
                yololayer = YoloLayer(layer_idx, info, in_channels[-1], self.in_width, self.in_height, self.training)
                modules.add_module('layer_'+ str(layer_idx)+'_yolo', yololayer)
                in_channels.append(in_channels[-1])
            module_list.append(modules)
        return module_list            
    
    def initialize_weights(self):
        # track all layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def transform_grid_data(self, features, fpn_idx):
        grid_w, grid_h= self.fpn_grid_size[fpn_idx*2: (fpn_idx+1)*2]
        width_per_grid, height_per_grid = self.stride[fpn_idx]
        _outs = []
        for b in range(self.box_per_anchor):
            offset = b * (5+self.n_classes)
            objness = torch.sigmoid(features[:,offset,:,:])
            box_xy = torch.sigmoid(features[:,offset+1:offset+3,:,:])
            box_w = self.anchor_size[b*2] * torch.exp(features[:,offset+3,:,:]) * width_per_grid
            box_h = self.anchor_size[b*2 + 1] * torch.exp(features[:,offset+4,:,:]) * height_per_grid
            for j in range(grid_h):
                for i in range(grid_w):
                    #objness = torch.sigmoid(features[:,offset,j,i])
                    box_x = (i+box_xy[:,0,j,i]) * width_per_grid
                    box_y = (j+box_xy[:,1,j,i]) * height_per_grid
                    #conf = self.softmax(features[:,offset+5:offset+5+self.n_classes,j,i])
                    conf = features[:,offset+5:offset+5+self.n_classes,j,i]
                    _out = torch.hstack((objness[:,j,i].reshape(self.batch,-1), box_x.reshape(self.batch,-1), box_y.reshape(self.batch,-1),
                                         box_w[:,j,i].reshape(self.batch,-1), box_h[:,j,i].reshape(self.batch,-1), conf.reshape(self.batch,-1)))
                    _outs.append(_out)
        out = torch.transpose(torch.stack(_outs),1,0)
        return out
    
    def get_grid_indexes(self, features):
        yv, xv = torch.meshgrid([torch.arange(features.shape[2]), torch.arange(features.shape[3])])
        grid_index = torch.stack((xv,yv), dim=2)
        grid_index = grid_index.view(1,grid_index.shape[0],grid_index.shape[1],2).cuda()
        return grid_index

    def convert_box_type(self, features, yololayer, yolo_idx):
        #get grid idexes
        grid_indexes = self.get_grid_indexes(features)
        #features = features.permute([0,2,3,1]).contiguous()
        height_per_grid, width_per_grid = self.get_grid_wh(yolo_idx)

        if not self.training:
            for a in range(self.box_per_anchor):
                #for each box in anchor
                feat = features[:, :, :, yololayer.box_attr*a:yololayer.box_attr*(a+1)]
                feat[:,:,:,0] = (torch.sigmoid(feat[:,:,:,0]) + grid_indexes[:,:,:,0]) * width_per_grid #x (tx + grid_idx)*grid_w
                feat[:,:,:,1] = (torch.sigmoid(feat[:,:,:,1]) + grid_indexes[:,:,:,1]) * height_per_grid #h (ty + grid_idx)*grid_h
                feat[:,:,:,2] = torch.exp(feat[:,:,:,2]) * yololayer.anchor[a][0]#w
                feat[:,:,:,3] = torch.exp(feat[:,:,:,3]) * yololayer.anchor[a][1]#h
                feat[:,:,:,4:] = torch.sigmoid(feat[:,:,:,4:]) #obj, cls 
        return features
    
    def get_grid_wh(self, grid_idx):
        grid_w, grid_h = self.fpn_grid_size[grid_idx * 2: (grid_idx + 1) * 2]
        w_per_grid, h_per_grid = self.in_width // grid_w, self.in_height // grid_h
        return w_per_grid, h_per_grid
    
    def get_loss(self, features):
        _loss = 0
        
        for f in features.shape[0]:
            for g in self.gt:
                features[f]
        
        return _loss 
    
    def forward(self, x):
        layer_result = []
        yolo_result = []
        for idx, (name, layer) in enumerate(zip(self.module_cfg, self.module_list)):
            #print(layer_result)
            if name['type'] == 'convolutional':
                x = layer(x)
                layer_result.append(x)
            elif name['type'] == 'shortcut':
                x = x + layer_result[int(name['from'])]
                layer_result.append(x)
            elif name['type'] == 'yolo':
                yolo_x = layer(x)
                #yolo_x = self.convert_box_type(yolo_x, layer, len(yolo_result))
                layer_result.append(yolo_x)
                yolo_result.append(yolo_x)
            elif name['type'] == 'upsample':
                x = layer(x)
                layer_result.append(x)
            elif name['type'] == 'route':
                layers = [int(y) for y in name["layers"].split(",")]
                x = torch.cat([layer_result[l] for l in layers], 1)
                layer_result.append(x)
        return yolo_result if self.training else torch.cat(yolo_result, dim=1)

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""
        # Open the weights file
        with open(weights_path, "rb") as f:
            # First five are header values
            header = np.fromfile(f, dtype=np.int32, count=5)
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights
        # Establish cutoff for loading backbone weights
        cutoff = None
        # If the weights file has a cutoff, we can find out about it by looking at the filename
        # examples: darknet53.conv.74 -> cutoff is 74
        filename = os.path.basename(weights_path)
        if ".conv." in filename:
            try:
                cutoff = int(filename.split(".")[-1])  # use last part of filename
            except ValueError:
                pass

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_cfg, self.module_list)):
            print(i, module_def, module)
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(
                    weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w



        
