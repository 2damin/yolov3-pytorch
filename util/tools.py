import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import sys

def minmax2cxcy(box):
    if len(box) != 4:
        return torch.FloatTensor([0,0,0,0])
    else:
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        w = box[2] - box[0]
        h = box[3] - box[1]
        box[0] = cx
        box[1] = cy
        box[2] = w
        box[3] = h

def cxcy2minmax(box):
    if len(box) != 4:
        return [0,0,0,0]
    else:
        xmin = box[0] - box[2] / 2
        ymin = box[1] - box[3] / 2
        xmax = box[2] + box[2] / 2
        ymax = box[3] + box[3] / 2
        box[0] = xmin
        box[1] = ymin
        box[2] = xmax
        box[3] = ymax

def resizeBox(box, original_wh, resize_wh):
    if len(box) != 4:
        return torch.FloatTensor([0,0,0,0])
    else:
        ratio_w, ratio_h = resize_wh[0] / original_wh[0], resize_wh[1] / original_wh[1]
        box[0] = box[0] * ratio_w
        box[1] = box[1] * ratio_h
        box[2] = box[2] * ratio_w
        box[3] = box[3] * ratio_h
        # xmin, xmax = box[0] * ratio_w, box[2] * ratio_w
        # ymin, ymax = box[1] * ratio_h, box[3] * ratio_h
        # return torch.FloatTensor([xmin, ymin, xmax, ymax])
    
#Get the anchor's indexes of box based on box location 
def getAnchorIdx(box, anchors):
    box_anchor_idxes = []
    if len(box) == 0 or len(anchors) == 0:
        return box_anchor_idxes
    else:
        for b in box:
            if len(b) != 4:
                box_anchor_idxes.append([0,0,0,0])
                continue
            idx_per_box = []
            for anc in anchors:
                fi, fj = int(torch.div(b[0], anc[0]).item()), int(torch.div(b[1], anc[1]).cpu().detach().item())
                idx_per_box.append([fi,fj])
            box_anchor_idxes.append(idx_per_box)
        return box_anchor_idxes
    
def convert_gt_box(box, yololayer):
    if len(box) == 0:
        return
    new_box = torch.zeros_like(box, dtype=torch.float32).cuda()
    for i in range(box.shape[0]):
        fi, fj = int(torch.div(box[i,0], yololayer.stride[0]).item()), int(torch.div(box[i,1], yololayer.stride[1]).item())
        new_box[i,0] = box[i,0] - fi * yololayer.stride[0] #(fi + 0.5) * yololayer.stride[0]
        new_box[i,1] = box[i,1] - fj * yololayer.stride[1] #(fj + 0.5) * yololayer.stride[1]
        new_box[i,2] = torch.log(torch.div(box[i,2],yololayer.stride[0]))
        new_box[i,3] = torch.log(torch.div(box[i,3],yololayer.stride[1]))
    return new_box
    
def iou(a, b, mode = 0):
    #mode 0 : cxcywh. mode 1 : minmax
    if mode == 0:
        a_x1, a_y1 = a[:,0]-a[:,2]/2, a[:,1]-a[:,3]/2
        a_x2, a_y2 = a[:,0]+a[:,2]/2, a[:,1]+a[:,3]/2
        b_x1, b_y1 = b[:,0]-b[:,2]/2, b[:,1]-b[:,3]/2
        b_x2, b_y2 = b[:,0]+b[:,2]/2, b[:,1]+b[:,3]/2
    else:
        a_x1, a_y1, a_x2, a_y2 = a[:,0], a[:,1], a[:,2], a[:,3]
        b_x1, b_y1, b_x2, b_y2 = b[:,0], b[:,1], b[:,2], b[:,3]
    xmin = torch.max(a_x1, b_x1)
    xmax = torch.min(a_x2, b_x2)
    ymin = torch.max(a_y1, b_y1)
    ymax = torch.min(a_y2, b_y2)
    #get intersection area 
    inter_area = torch.clamp(xmax - xmin, min=0) * \
                 torch.clamp(ymax - ymin, min=0)
    #get each box area
    a_area = (a_x2 - a_x1 + 1) * (a_y2 - a_y1 + 1)
    b_area = (b_x2 - b_x1 + 1) * (b_y2 - b_y1 + 1)
    
    return inter_area / (a_area + b_area - inter_area + 1e-6)

def sigmoid(a):
    a_shape = a.shape
    a = np.reshape(a,[-1])
    a = [1 / (1 + np.exp(-v)) for v in a]
    a = np.reshape(a, a_shape)
    return a

def softmax(a):
    exp_a = np.exp(a - np.max(a))
    return exp_a / exp_a.sum()

def drawBox(_img, boxes, mode = 0):
    _img = _img * 255
    #img dim is [C,H,W]
    if _img.shape[0] < 4:
        _img_data = np.array(np.transpose(_img, (1,2,0)), dtype=np.uint8)
        img_data = Image.fromarray(_img_data)
    else:
        img_data = Image.fromarray(_img, 'BGR')
    draw = ImageDraw.Draw(img_data)
    for box in boxes:
        if box[4] < 0.5:
            continue
        
        if mode == 0:
            draw.rectangle((box[0] - box[2]/2, box[1] - box[3]/2, box[0] + box[2]/2, box[1] + box[3]/2), outline=(0,255,0), width=1)
        else:
            draw.rectangle((box[0],box[1],box[2],box[3]), outline=(0,255,0), width=1)
    plt.imshow(img_data)
    plt.show()

def check_outrange(box, img_size):
    box = box.detach().cpu().numpy()
    xmin = box[0] - box[2]/2
    ymin = box[1] - box[3]/2
    xmax = box[0] + box[2]/2
    ymax = box[1] + box[3]/2
    if xmin < 0 or ymin < 0 or xmax > img_size[0] or ymax > img_size[1]:
        return 0
    else:
        return 1
    
def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs

def get_hyperparam(cfg):
    for c in cfg:
        if c['type'] == 'net':
            batch = int(c['batch'])
            subdivision = int(c['subdivisions'])
            momentum = float(c['momentum'])
            decay = float(c['decay'])
            saturation = float(c['saturation'])
            hue = float(c['hue'])
            exposure = float(c['exposure'])
            lr = float(c['learning_rate'])
            burn_in = int(c['burn_in'])
            max_batch = int(c['max_batches'])
            lr_policy = c['policy']
            steps = [int(x) for x in c['steps'].split(',')]
            scales = [float(x) for x in c['scales'].split(',')]

            return {'batch':batch, 'subdivision':subdivision,
                    'momentum':momentum, 'decay':decay,
                    'saturation':saturation, 'hue':hue,
                    'exposure':exposure, 'lr':lr,
                    'burn_in':burn_in, 'max_batch':max_batch,
                    'lr_policy':lr_policy, 'steps':steps,
                    'scales':scales}
        else:
            continue
        
def non_max_sup(input, num_classes, conf_th = 0.5, nms_th = 0.4):
    
    box = input.new(input.shape)
    box[:,:,0] = input[:,:,0] - input[:,:,2] / 2
    box[:,:,1] = input[:,:,1] - input[:,:,3] / 2
    box[:,:,2] = input[:,:,0] + input[:,:,2] / 2
    box[:,:,3] = input[:,:,1] + input[:,:,3] / 2
    input[:,:,:4] = box[:,:,:4]
    
    output = [None for _ in range(len(input))]
    for i, pred in enumerate(input):
        conf_mask = (pred[:,4] >= conf_th).squeeze()
        pred = pred[conf_mask]
        if not pred.size(0):
            continue
        #get the highst score & class 
        class_conf, class_pred = torch.max(pred[:,5:5+num_classes], 1, keepdim=True)
        #Convert predictions type [x,y,w,h,obj,class_conf,class_pred]
        detections = torch.cat((pred[:,:5], class_conf.float(), class_pred.float()),1)
        
        unique_labels = detections[:,-1].cpu().unique()
        if input.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            #get the detection with certain class
            detections_c = detections[detections[:,-1] == c]
            #sort the detections by maximum obj score
            _, conf_sort_index = torch.sort(detections_c[:,4], descending=True)
            detections_c = detections_c[conf_sort_index]
            #perforom non-maximum suppression
            max_detections = []
            while detections_c.size(0):
                #get detection with highest confidence
                max_detections.append(detections_c[0].unsqueeze(0))
                
                if len(detections_c) == 1:
                    break
                
                #get IOUs for all boxes with lower conf
                ious = iou(max_detections[-1], detections_c[1:])
                #remove detections iou >= nms threshold
                detections_c = detections_c[1:][ious < nms_th]
            
            max_detections = torch.cat(max_detections).data
            #update outputs
            output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))
    return output

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']