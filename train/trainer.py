import time, os
import torch
import torch.optim as optim
import torch.profiler
import torch.utils.data
import torchvision.transforms as transforms

from util.tools import *
from train.loss import *
from train.scheduler import *
from torch.utils.data.dataloader import DataLoader
from dataloader.yolodata import *
from dataloader.data_transforms import *

from terminaltables import AsciiTable

class Trainer:
    def __init__(self, model, train_loader, eval_loader, hparam, class_str, device, checkpoint = None, torch_writer = None):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.max_batch = hparam['max_batch']
        self.decay_step = hparam['steps']
        self.device = device
        self.epoch = 0
        self.iter = 0
        self.torch_writer = torch_writer
        self.yololoss = YoloLoss(self.device, self.model.n_classes, hparam['ignore_cls'])
        self.optimizer = optim.SGD(model.parameters(), lr=hparam['lr'], momentum=hparam['momentum'])
        self.class_str = class_str

        
        if checkpoint is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            self.iter = checkpoint['iteration']

        # self.lr_scheduler = optim.lr_scheduler.StepLR(
        #     self.optimizer,
        #     step_size = 50,
        #     gamma = 0.9)

        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_batch-hparam['burn_in'])
        self.lr_scheduler = LearningRateWarmUP(optimizer=self.optimizer,
                                               warmup_iteration=hparam['burn_in'],
                                               target_lr=hparam['lr'],
                                               after_scheduler=scheduler_cosine)

    def run(self):
        while True:
            #evaluate
            self.model.eval()
            self.run_eval()
            self.model.train()
            loss = self.run_iter()
            self.epoch += 1
            if self.epoch % 1 == 0:
                checkpoint_path = os.path.join("./output", "model_epoch" + str(self.epoch) + ".pth")
                torch.save({'epoch': self.epoch,
                            'iteration': self.iter,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': loss}, checkpoint_path)
            
            if self.max_batch <= self.iter:
                break

    def run_iter(self):
        #torch.autograd.set_detect_anomaly(True)
        for i, batch in enumerate(self.train_loader):
            #drop the invalid frames
            if batch is None:
                continue
            input_img, targets, anno_path = batch
            
            #input_wh = [input_img.shape[3], input_img.shape[2]]
            # inv_img = inv_normalize(input_img)
            # for b in range(len(targets)):
            #     target_box = targets[b]['bbox']
            #     target_cls = targets[b]['cls']
            #     target_occ = targets[b]['occ']
            #     for t in range(target_box.shape[0]):
            #         print(target_box[t], input_wh)
            #         target_box[t,0] *= input_wh[0]
            #         target_box[t,2] *= input_wh[0]
            #         target_box[t,1] *= input_wh[1]
            #         target_box[t,3] *= input_wh[1]
            #         drawBox(input_img.detach().numpy()[b], target_box, cls = targets[b]['cls'])
            # continue
            
            input_img = input_img.to(self.device, non_blocking=True)

            start_time = time.time()

            output = self.model(input_img)
            
            loss, loss_list = self.yololoss.compute_loss(pred = output,
                                                        targets = targets,
                                                        yolo_layers = self.model.yolo_layers,
                                                        tmp_img = None)
            
            calc_time = time.time() - start_time
            print("{} iter {:.6f} lr {:.4f} loss / {} time".format(self.iter, get_lr(self.optimizer), loss.item(), calc_time))
            
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.lr_scheduler.step(self.iter)
            self.iter += 1
            
            loss_name = ['total_loss','obj_loss', 'cls_loss', 'box_loss']

            if i % 100 == 0:
                duration = float(time.time() - start_time)
                latency = self.model.batch / duration
                print("epoch {} iter {} lr {} , loss : {}".format(self.epoch, self.iter, get_lr(self.optimizer), loss.item()))
                self.torch_writer.add_scalar("lr", get_lr(self.optimizer), self.iter)
                self.torch_writer.add_scalar('example/sec', latency, self.iter)
                self.torch_writer.add_scalar('total_loss', loss, self.iter)
                for ln, ls in zip(loss_name, loss_list):
                    self.torch_writer.add_scalar(ln, ls, self.iter)
        return loss
    
    def run_eval(self):
        predict_all = []
        gt_labels = []
        for i, batch in enumerate(self.eval_loader):
            #skip invalid frames
            if batch is None:
                continue
            input_img, targets, _ = batch
            
            input_img = input_img.to(self.device, non_blocking=True)
            
            gt_labels += targets[...,1].tolist()

            targets[...,2:6] = cxcy2minmax(targets[...,2:6])
            input_wh = [input_img.shape[3], input_img.shape[2]]
            targets[...,2] *= input_wh[0]
            targets[...,4] *= input_wh[0]
            targets[...,3] *= input_wh[1]
            targets[...,5] *= input_wh[1]
            start_time = time.time()
            with torch.no_grad():
                output = self.model(input_img)
                best_box_list = non_max_suppression(output, conf_thres=0.1, iou_thres=0.5)
                
            predict_all += get_batch_statistics(best_box_list, targets, iou_threshold=0.5)
                
            if len(predict_all) == 0:
                print("no detection in eval data")
                return None
            if i % 100 == 0:
                print("-------eval {}th iter -----".format(i))
        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [
            np.concatenate(x, 0) for x in list(zip(*predict_all))]

        metrics_output = ap_per_class(
            true_positives, pred_scores, pred_labels, gt_labels)
        
        #print eval result
        if metrics_output is not None:
            precision, recall, ap, f1, ap_class = metrics_output
            ap_table = [["Index", "Class", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, self.class_str[c], "%.5f" % ap[i]]]
            print(AsciiTable(ap_table).table)
        print("---- mAP {AP.mean():.5f} ----")
        

