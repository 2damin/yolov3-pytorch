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
        
        if cx - w/2 < 0 or cx + w/2 > 1:
            w -= 0.001
        if cy - h/2 < 0 or cy + h/2 > 1:
            h -= 0.001
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
        xmax = box[0] + box[2] / 2
        ymax = box[1] + box[3] / 2
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
    
def iou(a, b, mode = 0, device = None):
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
    inter = (xmax - xmin).clamp(0) * (ymax - ymin).clamp(0)
    #get each box area
    a_area = (a_x2 - a_x1) * (a_y2 - a_y1)
    b_area = (b_x2 - b_x1) * (b_y2 - b_y1)
    union = a_area + b_area - inter
    
    if device is not None:
        iou = torch.zeros(b.shape[0]).to(device)
    else:
        iou = torch.zeros(b.shape[0])
    iou[union > 1] = inter[union > 1] / union[union > 1]

    return iou

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area =    torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                    torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def sigmoid(a):
    a_shape = a.shape
    a = np.reshape(a,[-1])
    a = [1 / (1 + np.exp(-v)) for v in a]
    a = np.reshape(a, a_shape)
    return a

def softmax(a):
    exp_a = np.exp(a - np.max(a))
    return exp_a / exp_a.sum()

def drawBox(_img, boxes = None, cls = None, mode = 0, color = (0,255,0)):
    _img = _img * 255
    #img dim is [C,H,W]
    if _img.shape[0] == 3:
        _img_data = np.array(np.transpose(_img, (1,2,0)), dtype=np.uint8)
        img_data = Image.fromarray(_img_data)
    elif _img.ndim == 2:
        _img_data = np.array(_img, dtype=np.uint8)
        img_data = Image.fromarray(_img_data, 'L')
    draw = ImageDraw.Draw(img_data)
    
    if cls is None:
        cls = torch.zeros((boxes.shape[0]))
    
    if boxes is not None:
        for i, box in enumerate(boxes):
            # if (box[4] + box[5]) / 2 < 0.5:
            #     continue
            # if cls[i] == 8:
            #     color = (255,0,0)
            # if i == 2:
            #     color = (0,255,255)
            if mode == 0:
                draw.rectangle((box[0] - box[2]/2, box[1] - box[3]/2, box[0] + box[2]/2, box[1] + box[3]/2), outline=color, width=1)
            else:
                draw.rectangle((box[0],box[1],box[2],box[3]), outline=color, width=1)
            draw.text((box[0],box[1]), str(int(cls[i])), fill ="red")
    plt.imshow(img_data)
    plt.show()

def drawBoxes(_img, boxes, gt, mode = 0):
    _img = _img * 255
    #img dim is [C,H,W]
    if _img.shape[0] == 3:
        _img_data = np.array(np.transpose(_img, (1,2,0)), dtype=np.uint8)
        img_data = Image.fromarray(_img_data)
    elif _img.ndim == 2:
        _img_data = np.array(_img, dtype=np.uint8)
        img_data = Image.fromarray(_img_data, 'L')
    draw = ImageDraw.Draw(img_data)
    for box in boxes:
        if (box[4] + box[5]) / 2 < 0.5:
            continue
        
        if mode == 0:
            draw.rectangle((box[0] - box[2]/2, box[1] - box[3]/2, box[0] + box[2]/2, box[1] + box[3]/2), outline=(0,255,0), width=1)
        else:
            draw.rectangle((box[0],box[1],box[2],box[3]), outline=(0,255,0), width=1)
    
    for g in gt:
        if mode == 0:
            draw.rectangle((g[0] - g[2]/2, g[1] - g[3]/2, g[0] + g[2]/2, g[1] + g[3]/2), outline=(255,0,0), width=1)
        else:
            draw.rectangle((g[0],g[1],g[2],g[3]), outline=(255,0,0), width=1)
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
            in_width = int(c['width'])
            in_height = int(c['height'])
            _class = int(c['class'])
            ignore_cls = int(c['ignore_cls'])

            return {'batch':batch,
                    'subdivision':subdivision,
                    'momentum':momentum,
                    'decay':decay,
                    'saturation':saturation,
                    'hue':hue,
                    'exposure':exposure,
                    'lr':lr,
                    'burn_in':burn_in,
                    'max_batch':max_batch,
                    'lr_policy':lr_policy,
                    'steps':steps,
                    'scales':scales,
                    'in_width':in_width,
                    'in_height':in_height,
                    'class':_class,
                    'ignore_cls':ignore_cls}
        else:
            continue
        
def non_max_sup(input, num_classes, conf_th = 0.5, nms_th = 0.5, objectness = True):
    
    box = input.new(input.shape)
    box[:,:,0] = input[:,:,0] - input[:,:,2] / 2
    box[:,:,1] = input[:,:,1] - input[:,:,3] / 2
    box[:,:,2] = input[:,:,0] + input[:,:,2] / 2
    box[:,:,3] = input[:,:,1] + input[:,:,3] / 2
    box[:,:,4:] = input[:,:,4:]
    input[:,:,:4] = box[:,:,:4]
    
    #output = [None for _ in range(len(input))]
    output = None
    for i, pred in enumerate(box):
        #get the highst score & class of all pred
        class_conf_all, _ = torch.max(pred[:,5:5+num_classes], 1, keepdim=True)
        if objectness:
            pred_score = pred[:,4]
        else:
            pred_score = class_conf_all * 0.3 + pred[:,4] * 0.7

        conf_mask = (pred_score >= conf_th).squeeze()
        pred = pred[conf_mask]
        if not pred.size(0):
            continue

        #get the highst score & class of masked pred
        class_conf, class_pred = torch.max(pred[:,5:5+num_classes], 1, keepdim=True)
        
        #Convert predictions type [x,y,w,h,obj,class_conf,class_pred]
        detections = torch.cat((pred[:,:5], class_conf.float(), class_pred.float()),1)
        
        device = detections.device
        
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
                ious = iou(max_detections[-1], detections_c[1:], device = device)
                #remove detections iou >= nms threshold
                detections_c = detections_c[1:][ious < nms_th]

            max_detections = torch.cat(max_detections).data
            #update outputs
            output = max_detections if output is None else torch.cat((output, max_detections))
    return output

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']