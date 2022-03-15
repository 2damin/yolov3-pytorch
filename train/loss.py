from wsgiref.simple_server import demo_app
import torch
from torch._C import device
import torch.nn as nn
from util.tools import *
import sys
import math


class YoloLoss(nn.Module):
    
    def __init__(self, device, num_class, ignore_cls):
        super(YoloLoss, self).__init__()
        self.device = device
        self.mseloss = nn.MSELoss(size_average=False).to(device)
        self.bceloss = nn.BCELoss(size_average=False).to(device)
        self.softmax = nn.Softmax(dim=1).to(device)
        self.num_class = num_class
        self.lambda_xy = 2.5
        self.lambda_wh = 2.5
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0
        self.lambda_iou = 1.0
        self.ignore_cls = ignore_cls
        
    def compute_loss(self, input, targets = None, nw = None, nh = None, yolo_layers = None, tmp_img = None):
        output_list = []
        #for yolo_layers
        for yidx, yl in enumerate(yolo_layers):
            batch = input[yidx].shape[0]
            lh = input[yidx].shape[2]
            lw = input[yidx].shape[3]
            stride = [nw / lw, nh / lh]
            #scaled anchors
            scaled_anchors = [[anc[0] / stride[0], anc[1] / stride[1]] for anc in yl.anchor]
            #original anchors
            anchors = [[anc[0], anc[1]] for anc in yl.anchor]
            pred = input[yidx].view(batch,len(anchors),yl.box_attr,lh,lw).permute(0,1,3,4,2).contiguous()

            # Get outputs
            x = torch.sigmoid(pred[...,0])          # Center x
            y = torch.sigmoid(pred[...,1])          # Center y
            w = pred[...,2]                         # Width
            h = pred[...,3]                         # Height
            conf = torch.sigmoid(pred[...,4])       # Conf
            pred_cls = torch.sigmoid(pred[...,5:])  # Cls pred.
            
            preds_box = torch.stack((x,y,w,h,conf), dim = 4)
            
            if targets is not None:
                mask, noobj_mask, tx, ty, tw, th, tconf, tcls, pred_ious = self.get_targets(targets, anchors, nw, nh,
                                                                                 lw, lh, stride, yl.ignore_thresh,
                                                                                 preds = preds_box)

                #noobj_mask : if target_box is fitting well with anchor, the target is ignore when calculate no_obj confidence loss
                mask, noobj_mask = mask.to(self.device), noobj_mask.to(self.device)
                tx, ty, tw, th = tx.to(self.device), ty.to(self.device), tw.to(self.device), th.to(self.device)
                tconf, tcls = tconf.to(self.device), tcls.to(self.device)
                
                #loss variables
                loss_x, loss_y = torch.zeros(1).to(self.device), torch.zeros(1).to(self.device)
                loss_w, loss_h = torch.zeros(1).to(self.device), torch.zeros(1).to(self.device)
                loss_conf, loss_iou = torch.zeros(1).to(self.device), torch.zeros(1).to(self.device)
                loss_cls, loss = torch.zeros(1).to(self.device), torch.zeros(1).to(self.device)
                debug_loss_iou = torch.zeros(1)
                
                num_mask = torch.sum(mask).to(self.device)
                num_noobj_mask = torch.sum(noobj_mask).to(self.device)
                
                print("num_mask :", num_mask.item(), " no mask : ", num_noobj_mask.item())
                #print("pred : ", w, h)
                
                if num_mask != 0:
                    loss_x += self.mseloss(x * mask, tx * mask) / num_mask
                    loss_y += self.mseloss(y * mask, ty * mask) / num_mask
                    loss_w += self.mseloss(w * mask, tw * mask) / num_mask
                    loss_h += self.mseloss(h * mask, th * mask) / num_mask
                    loss_conf += self.bceloss(torch.clamp(conf * mask, min=1e-10, max=1 - 1e-10), mask) / num_mask \
                        + 0.5 * self.bceloss(torch.clamp(conf * noobj_mask, min=1e-10, max=1 - 1e-10),
                                             torch.clamp(noobj_mask * 0.0, min=1e-10, max=1 - 1e-10)) / num_noobj_mask
                    loss_iou += torch.sum(pred_ious * mask).to(self.device) / num_mask
                    # debug_pred_ious = pred_ious.cpu()
                    # debug_num_mask = num_mask.cpu()
                    # debug_loss_iou += torch.sum(1 - debug_pred_ious)/ debug_num_mask
                    if tcls[mask==1].nelement() != 0 or pred_cls[mask==1].nelement() != 0:
                        loss_cls += self.bceloss(pred_cls[mask==1], tcls[mask==1]) / num_mask
                        # print(pred_cls[mask==1].shape, tcls[mask==1].shape)
                else:
                    if torch.sum(torch.isfinite(conf)).cpu().item() != torch.numel(conf):
                        #print(conf)
                        print("WARNING conf has nan value, skip this step of loss backward")
                        for b in range(len(targets)):
                            target_box = targets[b]['bbox']
                            target_cls = targets[b]['cls']
                            target_occ = targets[b]['occ']
                            target_trunc = targets[b]['trunc']
                            print(targets[b]['path'])
                            for t in range(target_box.shape[0]):
                                print(target_box[t,0] - target_box[t,2] / 2,
                                      target_box[t,1] - target_box[t,3] / 2,
                                      target_box[t,0] + target_box[t,2] / 2,
                                      target_box[t,1] + target_box[t,3] / 2,
                                      target_cls[t])
                                show_pred_box = torch.zeros(4).to(self.device)
                                show_pred_box[0] = (target_box[t,0] - target_box[t,2] / 2) * nw
                                show_pred_box[1] = (target_box[t,1] - target_box[t,3] / 2) * nh
                                show_pred_box[2] = (target_box[t,0] + target_box[t,2] / 2) * nw
                                show_pred_box[3] = (target_box[t,1] + target_box[t,3] / 2) * nh
                                drawBox(tmp_img.cpu().detach().numpy()[0],
                                        show_pred_box, cls = 1, mode = 1, text = None)
                        #sys.exit(1)
                        output_list = []
                        return output_list
                    else:
                        loss_conf += 0.5 * self.bceloss(torch.clamp(conf * noobj_mask, min=1e-10, max=1 - 1e-10),
                                                        torch.clamp(noobj_mask * 0.0, min=1e-10, max=1 - 1e-10)) / num_noobj_mask
                
                #total_loss
                loss += loss_x * self.lambda_xy + loss_y * self.lambda_xy + \
                    loss_w * self.lambda_wh + loss_h * self.lambda_wh + \
                    loss_conf * self.lambda_conf + loss_cls * self.lambda_cls + \
                    loss_iou * self.lambda_iou
                # print("debug_loss_iou: ", debug_loss_iou)
                # debug_loss = loss.cpu()
                # print("total:", debug_loss)
                print("{}th layer loss: {:.3f} x {:.3f} y {:.3f} w {:.3f} h {:.3f} conf {:.3f} cls {:.3f} iou {:.3f}".format(yidx,
                                                                                                    loss.item(),
                                                                                                    loss_x.item(), 
                                                                                                    loss_y.item(),
                                                                                                    loss_w.item(),
                                                                                                    loss_h.item(),
                                                                                                    loss_conf.item(),
                                                                                                    loss_cls.item(),
                                                                                                    loss_iou.item()))
                output_list.append([loss, loss_x.item(), loss_y.item(), loss_w.item(),
                                    loss_h.item(), loss_conf.item(), loss_cls.item(), loss_iou.item()]) 
            else:
                FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
                LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
                #calculate offsets of each grid
                grid_x = torch.linspace(0, lw-1, lw).repeat(lh, 1).repeat(
                    batch * len(anchors), 1, 1).view(x.shape).type(FloatTensor)
                grid_y = torch.linspace(0,lh-1,lh).repeat(lw, 1).t().repeat(
                    batch * len(anchors), 1, 1).view(y.shape).type(FloatTensor)
                #calculate anchor width height
                anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
                anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))

                anchor_w = anchor_w.repeat(batch, 1).repeat(1, 1, lh * lw).view(w.shape)
                anchor_h = anchor_h.repeat(batch, 1).repeat(1, 1, lh * lw).view(h.shape)

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

    
    def get_targets(self, targets, anchors, nw, nh, lw, lh, stride, ignore_thresh, tmp_img = None, preds = None):
        bs = len(targets)
        
        mask = torch.zeros(bs, len(anchors), lh, lw, requires_grad=False)
        noobj_mask = torch.ones(bs, len(anchors), lh, lw, requires_grad=False)
        tx = torch.zeros(bs, len(anchors), lh, lw, requires_grad=False)
        ty = torch.zeros(bs, len(anchors), lh, lw, requires_grad=False)
        tw = torch.zeros(bs, len(anchors), lh, lw, requires_grad=False)
        th = torch.zeros(bs, len(anchors), lh, lw, requires_grad=False)
        tconf = torch.zeros(bs, len(anchors), lh, lw, requires_grad=False)
        tcls = torch.zeros(bs, len(anchors), lh, lw, self.num_class, requires_grad=False)
        pred_ious = torch.zeros(bs, len(anchors), lh, lw, requires_grad=False).to(self.device)
        
        scale_w = nw / lw
        scale_h = nh / lh
        
        for b in range(bs):
            target_box = targets[b]['bbox']
            target_cls = targets[b]['cls']
            target_occ = targets[b]['occ']
            target_trunc = targets[b]['trunc']

            #if target object dont exist
            if target_box is None and target_cls is None:
                continue
            
            for t in range(target_box.shape[0]):
                #ignore Dontcare objects and occluded objects
                if int(target_cls[t]) == self.ignore_cls or target_occ[t] > 1 or target_trunc[t] > 0.5:
                    continue
                #get box position relative to grid(anchor)
                gx = target_box[t,0] * lw
                gy = target_box[t,1] * lh
                gw = target_box[t,2] * nw
                gh = target_box[t,3] * nh
                #get index of grid
                gi = int(gx)
                gj = int(gy)

                # for anc in range(3):
                #     show_pred_box = torch.zeros(4).to(self.device)
                #     show_pred_box[0] = (preds[b, anc, gj, gi, 0] + gi) * scale_w
                #     show_pred_box[1] = (preds[b, anc, gj, gi, 1] + gj) * scale_h
                #     show_pred_box[2] = (torch.exp(preds[b, anc, gj, gi, 2]) * anchors[anc][0])
                #     show_pred_box[3] = (torch.exp(preds[b, anc, gj, gi, 3]) * anchors[anc][1])
                #     drawBox(tmp_img.cpu().detach().numpy()[0],
                #             [torch.tensor([gx * scale_w, gy * scale_h, gw, gh]),
                #              torch.tensor([gi * scale_w,gj * scale_h, anchors[anc][0], anchors[anc][1]]),
                #              show_pred_box], cls = target_cls, text = None)

                #make gt_box shape
                gt_box = torch.FloatTensor(np.array([0,0,gw,gh])).unsqueeze(0)
                #make box shape of each anchor 
                anchor_shape = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)),
                                                                 np.array(anchors)),1))
                
                #get iou between gt and anchor_shape
                anc_iou = iou(gt_box, anchor_shape, mode = 0)

                # print("tbox :",target_box[t,0:4], gx, gy, gw, gh, "gbox : ", gt_box, "anc :", anchor_shape)
                # print("anchor iou :", anc_iou)
                # anc_iou_b = bbox_iou(gt_box, anchor_shape, x1y1x2y2 = False)

                #mask to zero to ignore the prediction if larger than ignore_threshold
                noobj_mask[b, anc_iou > ignore_thresh, gj, gi] = 0
                #get best anchor box 
                best_box = np.argmax(anc_iou)
                positive_box = anc_iou > ignore_thresh
                if anc_iou[best_box] >= 0.5:
                    pred_box = torch.zeros(4).to(self.device)
                    pred_box[0] = (preds[b, best_box, gj, gi, 0] + gi) * scale_w
                    pred_box[1] = (preds[b, best_box, gj, gi, 1] + gj) * scale_h
                    pred_box[2] = (torch.exp(preds[b, best_box, gj, gi, 2]) * anchors[best_box][0])
                    pred_box[3] = (torch.exp(preds[b, best_box, gj, gi, 3]) * anchors[best_box][1])
                    gt_box = torch.tensor([gx * scale_w, gy * scale_h, gw , gh]).to(self.device)
                    
                    pred_iou = iou(gt_box.unsqueeze(0), pred_box.unsqueeze(0), mode = 0, device = self.device)
                    pred_ious[b, best_box, gj, gi] = 1 - pred_iou

                    #print("pred_iou : ", pred_iou)
                    #mask
                    mask[b, best_box, gj, gi] = 1
                    #if IOU between pred and gt is under 0.1, negative box
                    if pred_iou < 0.1:
                        mask[b, best_box, gj, gi] = 0
                        pred_ious[b, best_box, gj, gi] = 0
                    
                    #coordinate and width,height
                    tx[b, best_box, gj, gi] = gx - gi
                    ty[b, best_box, gj, gi] = gy - gj
                    tw[b, best_box, gj, gi] = math.log(gw/anchors[best_box][0] + 1e-16) #np.log(gw/(anchors[best_box][0]/stride[0]) + 1e-16)
                    th[b, best_box, gj, gi] = math.log(gh/anchors[best_box][1] + 1e-16) #np.log(gh/(anchors[best_box][1]/stride[1]) + 1e-16)
                    # print("Target : " ,tx[b, best_box, gj, gi], ty[b, best_box, gj, gi], tw[b, best_box, gj, gi], th[b, best_box, gj, gi])

                    #objectness
                    tconf[b, best_box, gj, gi] = 1

                    #class confidence
                    tcls[b, best_box, gj, gi, int(target_cls[t])] = 1
        return mask, noobj_mask, tx, ty, tw, th, tconf, tcls, pred_ious
                

    def forward(self, out, positive_pred, negative_pred, _cls_gt, bboxes_gt, batch_idx):
        #negative pred loss
        obj_gt_neg = torch.zeros(negative_pred.shape[0], dtype=torch.float32).to(self.device)
        #objness loss
        loss = 0
        for i, neg in enumerate(negative_pred):
            loss += 0.5 * self.bceloss(out[neg[0]][batch_idx, neg[1], neg[2], neg[3] * (self.num_class + 5) + 4], obj_gt_neg[i]) #0.5 * (0 - sigmoid(out[b,neg, 0]))**2
        print(loss)

        #make one hot vector of class_gt
        cls_gt = torch.zeros(len(_cls_gt), self.num_class, dtype=torch.float32).to(self.device)
        cls_gt = cls_gt.scatter_(1, _cls_gt.unsqueeze(1), 1.)

        #positive pred loss
        #iterate each gt object.
        exist_pos = False
        for k, pos in enumerate(positive_pred):
            if len(pos) == 0:
                continue
            exist_pos = True
            #objness loss
            obj_gt_pos = torch.ones(len(pos), dtype=torch.float32).to(self.device)
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
    