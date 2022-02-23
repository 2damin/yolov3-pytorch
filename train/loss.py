import torch
from torch._C import device
import torch.nn as nn
from util.tools import *
import sys

class YoloLoss(nn.Module):
    
    def __init__(self, device, num_class):
        super(YoloLoss, self).__init__()
        self.mseloss = nn.MSELoss().to(device)
        self.bceloss = nn.BCELoss().to(device)
        self.softmax = nn.Softmax(dim=1).to(device)
        self.num_class = num_class
        self.lambda_xy = 2.5
        self.lambda_wh = 2.5
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0
        
    def compute_loss(self, input, targets = None, input_wh = None, yolo_layers = None):
        output_list = []
        #for yolo_layers
        for yidx, yl in enumerate(yolo_layers):
            batch = input[yidx].shape[0]
            in_h = input[yidx].shape[2]
            in_w = input[yidx].shape[3]
            stride = [input_wh[0] //in_w, input_wh[1] // in_h]
            #scaled anchors
            anchors = [[anc[0] / stride[0], anc[1] / stride[1]] for anc in yl.anchor]
            pred = input[yidx].view(batch,len(anchors),yl.box_attr,in_h,in_w).permute(0,1,3,4,2).contiguous()

            # Get outputs
            x = torch.sigmoid(pred[...,0])          # Center x
            y = torch.sigmoid(pred[...,1])          # Center y
            w = pred[...,2]                         # Width
            h = pred[...,3]                         # Height
            conf = torch.sigmoid(pred[...,4])       # Conf
            pred_cls = torch.sigmoid(pred[...,5:])  # Cls pred.
            
            if targets is not None:
                mask, noobj_mask, tx, ty, tw, th, tconf, tcls = self.get_targets(targets, anchors,
                                                               in_w, in_h,
                                                               yl.ignore_thresh)
                #noobj_mask : if target_box is fitting well with anchor, the target is ignore when calculate no_obj confidence loss
                mask, noobj_mask = mask.cuda(), noobj_mask.cuda()
                tx, ty, tw, th = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda()
                tconf, tcls = tconf.cuda(), tcls.cuda()
                #loss
                loss_x = self.bceloss(x * mask, tx * mask)
                loss_y = self.bceloss(y * mask, ty * mask)
                loss_w = self.mseloss(w * mask, tw * mask)
                loss_h = self.mseloss(h * mask, th * mask)
                loss_conf = self.bceloss(conf * mask, tconf) + \
                    0.5 * self.bceloss(conf * noobj_mask, noobj_mask * 0.0)
                loss_cls = self.bceloss(pred_cls[mask==1], tcls[mask==1])
                #total_loss
                loss = loss_x * self.lambda_xy + loss_y * self.lambda_xy + \
                    loss_w * self.lambda_wh + loss_h * self.lambda_wh + \
                    loss_conf * self.lambda_conf + loss_cls * self.lambda_cls
                output_list.append([loss, loss_x.item(), loss_y.item(), loss_w.item(),
                                    loss_h.item(), loss_conf.item(), loss_cls.item()]) 
            else:
                FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
                LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
                #calculate offsets of each grid
                grid_x = torch.linspace(0, in_w-1, in_w).repeat(in_w, 1).repeat(
                    batch * len(anchors), 1, 1).view(x.shape).type(FloatTensor)
                grid_y = torch.linspace(0,in_h-1,in_h).repeat(in_h, 1).t().repeat(
                    batch * len(anchors), 1, 1).view(y.shape).type(FloatTensor)
                #calculate anchor width height
                anchor_w = FloatTensor(anchors).index_select(1, LongTensor([0]))
                anchor_h = FloatTensor(anchors).index_select(1, LongTensor([1]))
                anchor_w = anchor_w.repeat(batch, 1).repeat(1, 1, in_h * in_w).view(w.shape)
                anchor_h = anchor_h.repeat(batch, 1).repeat(1, 1, in_h * in_w).view(h.shape)

                #add offset and anchors
                pred_boxes = FloatTensor(pred[...,:4].shape)
                pred_boxes[..., 0] = x.data + grid_x
                pred_boxes[..., 1] = y.data + grid_y
                pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
                pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
                #results
                scale = torch.Tensor([stride[0],stride[1]] * 2).type(FloatTensor)
                output = torch.cat((pred_boxes.view(batch, -1, 4) * scale,
                                    conf.view(batch, -1, 1), pred_cls.view(batch, -1, yl.n_classes)),-1)
                output_list.append(output.data)
        return output_list

    
    def get_targets(self, targets, anchors, in_w, in_h, ignore_thresh):
        bs = len(targets)
        
        mask = torch.zeros(bs, len(anchors), in_h, in_w, requires_grad=False)
        noobj_mask = torch.ones(bs, len(anchors), in_h, in_w, requires_grad=False)
        tx = torch.zeros(bs, len(anchors), in_h, in_w, requires_grad=False)
        ty = torch.zeros(bs, len(anchors), in_h, in_w, requires_grad=False)
        tw = torch.zeros(bs, len(anchors), in_h, in_w, requires_grad=False)
        th = torch.zeros(bs, len(anchors), in_h, in_w, requires_grad=False)
        tconf = torch.zeros(bs, len(anchors), in_h, in_w, requires_grad=False)
        tcls = torch.zeros(bs, len(anchors), in_h, in_w, self.num_class, requires_grad=False)
        
        for b in range(bs):
            target_box = targets[b]['bbox']
            target_cls = targets[b]['cls']
            for t in range(target_box.shape[0]):
                if target_box.sum() == 0:
                    print(target_box, target_cls)
                    continue
                #get box position relative to grid(anchor)
                gx = target_box[t,0] * in_w
                gy = target_box[t,1] * in_h
                gw = target_box[t,2] * in_w
                gh = target_box[t,3] * in_h
                #get index of grid
                gi = int(gx)
                gj = int(gy)
                
                #make gt_box shape
                gt_box = torch.FloatTensor(np.array([0,0,gw,gh])).unsqueeze(0)
                #make box shape of each anchor 
                anchor_shape = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)),
                                                                 np.array(anchors)),1))
                #print(target_box[t,0:4], gx, gy, gw, gh, gi,gj, "gtbox : ", gt_box, anchor_shape)
                
                #get iou between gt and anchor_shape
                anc_iou = iou(gt_box, anchor_shape)

                #mask to zero to ignore the prediction if larger than ignore_threshold
                noobj_mask[b,anc_iou > ignore_thresh, gj, gi] = 0
                #get best anchor box 
                best_box = np.argmax(anc_iou)
                
                #mask
                mask[b, best_box, gj, gi] = 1
                #coordinate and width,height
                tx[b, best_box, gj, gi] = gx - gi
                ty[b, best_box, gj, gi] = gy - gj
                tw[b, best_box, gj, gi] = np.log(gw/anchors[best_box][0] + 1e-16)
                th[b, best_box, gj, gi] = np.log(gh/anchors[best_box][1] + 1e-16)
                #objectness
                tconf[b, best_box, gj, gi] = 1
                
                #ignore class
                if int(target_cls[t]) == 8:
                    mask[b, best_box, gj, gi] = 0
                    noobj_mask[b,anc_iou > ignore_thresh, gj, gi] = 0
                    tconf[b, best_box, gj, gi] = 0
                else:
                    #class confidence
                    tcls[b, best_box, gj, gi, int(target_cls[t])] = 1
        return mask, noobj_mask, tx, ty, tw, th, tconf, tcls
                

    def forward(self, out, positive_pred, negative_pred, _cls_gt, bboxes_gt, batch_idx):
        #negative pred loss
        obj_gt_neg = torch.zeros(negative_pred.shape[0], dtype=torch.float32).cuda()
        #objness loss
        loss = 0
        for i, neg in enumerate(negative_pred):
            loss += 0.5 * self.bceloss(out[neg[0]][batch_idx, neg[1], neg[2], neg[3] * (self.num_class + 5) + 4], obj_gt_neg[i]) #0.5 * (0 - sigmoid(out[b,neg, 0]))**2
        print(loss)

        #make one hot vector of class_gt
        cls_gt = torch.zeros(len(_cls_gt), self.num_class, dtype=torch.float32).cuda()
        cls_gt = cls_gt.scatter_(1, _cls_gt.unsqueeze(1), 1.)

        #positive pred loss
        #iterate each gt object.
        exist_pos = False
        for k, pos in enumerate(positive_pred):
            if len(pos) == 0:
                continue
            exist_pos = True
            #objness loss
            obj_gt_pos = torch.ones(len(pos), dtype=torch.float32).cuda()
            #loss += self.bceloss(out[pos[0]][batch_idx, pos[1], pos[2], pos[3] * (self.num_class + 5)], obj_gt_pos)#1 * (1 - sigmoid(out[batch_idx,p, 0]))**2
            #class loss
            for i, p in enumerate(pos):
                loss += self.bceloss(out[p[0]][batch_idx, p[1], p[2], p[3] * (self.num_class + 5) + 4], obj_gt_pos[i])
                loss += self.bceloss(self.softmax(out[p[0]][batch_idx, p[1], p[2], p[3] * (self.num_class + 5) + 5 : (p[3] + 1) * (self.num_class + 5)].reshape(-1,1)), cls_gt[k].reshape(-1,1))
                #bbox loss
                loss += 5 * (self.mseloss(out[p[0]][batch_idx, p[1], p[2], p[3] * (self.num_class + 5)], bboxes_gt[k][0]) + 
                                self.mseloss(out[p[0]][batch_idx, p[1], p[2], p[3] * (self.num_class + 5) + 1], bboxes_gt[k][1]) +
                                self.mseloss(out[p[0]][batch_idx, p[1], p[2], p[3] * (self.num_class + 5) + 2], bboxes_gt[k][2]) +
                                self.mseloss(out[p[0]][batch_idx, p[1], p[2], p[3] * (self.num_class + 5) + 3], bboxes_gt[k][3]))

        # if exist_pos == False:
        #     total_loss = conf_loss_neg
        #     print("total l : ", conf_loss_neg)
        # else:
        #     total_loss = cls_loss_pos + conf_loss_pos + bbox_loss_pos + conf_loss_neg
        #     print("total l : ", cls_loss_pos, conf_loss_pos, bbox_loss_pos, conf_loss_neg)
        return loss
    