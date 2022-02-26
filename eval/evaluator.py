import torch
import numpy as np
import PIL, time
from train.loss import *
import csv
import pandas as pd

class Evaluator:
    def __init__(self, model, eval_data, eval_loader, device, hparam):
        self.model = model
        self.class_list = eval_data.class_str
        self.eval_loader = eval_loader
        self.device = device
        self.yololoss = YoloLoss(self.device, self.model.n_classes)
        self.gt_total = torch.zeros(self.model.n_classes, dtype=torch.int64, requires_grad=False)
        self.tp = torch.zeros(self.model.n_classes, dtype=torch.int64, requires_grad=False) #tp
        self.fn = torch.zeros(self.model.n_classes, dtype=torch.int64, requires_grad=False) #fn
        self.fp = torch.zeros(self.model.n_classes, dtype=torch.int64, requires_grad=False) #fp
        self.preds = None

    def run(self):
        for i, batch in enumerate(self.eval_loader):
            input_img, targets = batch
            if self.device == torch.device('cuda'):
                input_img = input_img.cuda()
            input_wh = [input_img.shape[3], input_img.shape[2]]

            with torch.no_grad():
                start_time = time.time()
            
                output = self.model(input_img)
                            
                output_list = self.yololoss.compute_loss(output, targets = None, input_wh = input_wh, yolo_layers = self.model.yolo_layers)

                output_all = torch.cat(output_list, dim=1)
                best_box_list = non_max_sup(output_all, self.model.n_classes, conf_th=0.2, nms_th=0.4)
                #print(best_box_list)
                
                self.evaluate(best_box_list, targets)
                
                if i % 100 == 0:
                    print("-------{} th iter -----".format(i))
                
                if best_box_list is None:
                    continue
                
                #drawBox(input_img.detach().cpu().numpy()[0,:,:,:], best_box_list, mode=1)
                
                #temporary transform the format of GT boxes to draw box in image
                # for i in range(targets['bbox'][0].shape[0]):
                #     if targets['cls'][0][i] == 8:
                #         continue
                #     cxcy2minmax(targets['bbox'][0,i])
                #     targets['bbox'][:,i,0] = targets['bbox'][:,i,0] * self.model.in_width
                #     targets['bbox'][:,i,2] = targets['bbox'][:,i,2] * self.model.in_width
                #     targets['bbox'][:,i,1] = targets['bbox'][:,i,1] * self.model.in_height
                #     targets['bbox'][:,i,3] = targets['bbox'][:,i,3] * self.model.in_height
                # drawBoxes(input_img.detach().cpu().numpy()[0,:,:,:], best_box_list, targets['bbox'][0], mode=1)
        
        #Calculate map, recall, tp, fp, fn.
        self.evaluate_result()
        
    
    def evaluate(self, preds, targets):
        if preds is None:
            return
        #move the preds and targets from device to cpu
        preds = preds.detach().cpu()
        targets['bbox'] = targets['bbox'].detach().cpu()
        targets['cls'] = targets['cls'].detach().cpu()
        #remove ignore class GT data
        targets_bbox_valid = targets['bbox'][0][targets['cls'][0] != 8]
        targets_cls_valid = targets['cls'][0][targets['cls'][0] != 8]
        #make mask tensor 
        pred_mask = torch.ones(preds.shape[0], requires_grad=False)
        gt_mask = torch.zeros(targets_cls_valid.shape[0], requires_grad=False)
        #print("pred_mask shape : ", pred_mask.shape, "gt_mask shape : ", gt_mask.shape)
        
        for i in range(targets_bbox_valid.shape[0]):
            tbox = targets_bbox_valid
            tcls = targets_cls_valid
            #change the target box format cxcywh to minmax
            cxcy2minmax(tbox[i])
            tbox[i,0] = tbox[i,0] * self.model.in_width
            tbox[i,2] = tbox[i,2] * self.model.in_width
            tbox[i,1] = tbox[i,1] * self.model.in_height
            tbox[i,3] = tbox[i,3] * self.model.in_height
    
            for j, (pbox, pobj_score, pcls_score, pcls_idx) in enumerate(zip(preds[:,:4], preds[:,4:5], preds[:,5:6], preds[:,6:])):
                #print(pbox.shape, pobj_score.shape, pcls_score.shape, pcls_idx.shape, tbox.shape, tcls.shape)
                
                if tcls[i] != pcls_idx or pred_mask[j] == 0:
                    continue
                
                iou_value = iou(tbox[i:i+1], preds[j:j+1,:4], mode=1)
                
                #print("box {} {} / iou : {}".format(tbox[i:i+1], preds[j:j+1,:4], iou_value))

                if iou_value > 0.5:
                    gt_mask[i] = 1
                    pred_mask[j] = 0
        
        gt_matched = (gt_mask == 1).nonzero(as_tuple=True)
        gt_missed = (gt_mask == 0).nonzero(as_tuple=True)
        pred_false = (pred_mask == 1).nonzero(as_tuple=True)
        pred_true = (pred_mask == 0).nonzero(as_tuple=True)

        if gt_matched[0].nelement() != 0:
            for p in range(gt_matched[0].shape[0]):
                self.tp[targets_cls_valid[gt_matched[0][p]]] += 1
        if gt_missed[0].nelement() != 0:
            for p in range(gt_missed[0].shape[0]):
                self.fn[targets_cls_valid[gt_missed[0][p]]] += 1
        if pred_false[0].nelement() != 0:
            for p in range(pred_false[0].shape[0]):
                self.fp[int(preds[pred_false[0][p],6])] += 1
        if pred_true[0].nelement() != 0:
            for p in range(pred_true[0].shape[0]):
                if self.preds is None:
                    self.preds = preds[pred_true[0][p]].reshape(1,-1)
                else:
                    self.preds = torch.cat((self.preds, preds[pred_true[0][p]].reshape(1,-1)), dim = 0)
                    
        
    def evaluate_result(self):
        precision = self.tp / (self.tp + self.fp + 1e-6)
        recall = self.tp / (self.tp + self.fn + 1e-6)
        precision = precision.detach().numpy().tolist()
        recall = recall.detach().numpy().tolist()
        tp = self.tp.detach().numpy().tolist()
        fp = self.fp.detach().numpy().tolist()
        fn = self.fn.detach().numpy().tolist()
        
        self.class_list.remove('DontCare')
        data = {'name' : ['precision', 'recall', 'TP', 'FP', 'FN']}
        
        for i, cls in enumerate(self.class_list):
            data[cls] = [precision[i], recall[i], tp[i], fp[i], fn[i]]
        
        df = pd.DataFrame(data)
        print(df)
        
        df.to_csv('./evaluation.csv')

        
