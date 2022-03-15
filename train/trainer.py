import time, os
import torch
import torch.optim as optim
import torch.profiler
import torch.utils.data
import torchvision.transforms as transforms

from util.tools import *
from train.loss import *
from train.scheduler import *

class Trainer:
    def __init__(self, model, train_loader, device, hparam, checkpoint = None, torch_writer = None):
        self.model = model
        self.train_loader = train_loader
        self.max_batch = hparam['max_batch']
        self.decay_step = hparam['steps']
        self.device = device
        self.epoch = 0
        self.iter = 0
        self.torch_writer = torch_writer
        self.yololoss = YoloLoss(self.device, self.model.n_classes, hparam['ignore_cls'])
        self.optimizer = optim.SGD(model.parameters(), lr=hparam['lr'], momentum=hparam['momentum'])
        
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
        # with torch.profiler.profile(
        #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/yolov3'),
        #     record_shapes=True,
        #     with_stack=True) as prof:
        while True:
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
        for i, batch in enumerate(self.train_loader):
            _input_img, targets = batch #batch['img'], batch['target']
            
            #stack [1,c,h,w] images to make [n,c,h,w] image
            input_img = torch.stack(_input_img,0)
            input_wh = [input_img.shape[3], input_img.shape[2]]
            
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
            #         drawBox(input_img.detach().numpy()[b], target_box, cls = targets[b]['cls'], text = None)
            # continue
            
            input_img = input_img.to(self.device)

            start_time = time.time()

            output = self.model(input_img)
            
            losses = self.yololoss.compute_loss(input = output,
                                                targets = targets,
                                                nw = self.model.in_width,
                                                nh = self.model.in_height,
                                                yolo_layers = self.model.yolo_layers,
                                                tmp_img = input_img)

            loss_name = ['total', 'x', 'y', 'w', 'h', 'conf', 'cls', 'iou']
            total_loss = [[] for _ in loss_name]
            for _, loss in enumerate(losses):
                for k, l in enumerate(loss):
                    total_loss[k].append(l)
                    #print(loss_name[k], " : ", l)
                    
            total_loss = [sum(l) for l in total_loss]
                    
            if total_loss[0] > 1000:
                for b in range(len(targets)):
                    target_box = targets[b]['bbox']
                    target_cls = targets[b]['cls']
                    target_occ = targets[b]['occ']
                    print(targets[b]['path'])
                    for t in range(target_box.shape[0]):
                        print(target_box[t], input_wh)
                        target_box[t,0] *= input_wh[0]
                        target_box[t,2] *= input_wh[0]
                        target_box[t,1] *= input_wh[1]
                        target_box[t,3] *= input_wh[1]
                        drawBox(input_img.detach().cpu().numpy()[b], target_box, cls = targets[b]['cls'], text = None)
                sys.exit(1)
            
            #if nan values in loss calculation, skip backprop
            if len(losses) == 0:
                print("WARNING: non-finite loss, skip backprop")
            else:
                print("{} iter ".format(self.iter), total_loss)
                loss = total_loss[0]
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step(self.iter)
                self.iter += 1

                if i % 100 == 0:
                    duration = float(time.time() - start_time)
                    latency = self.model.batch / duration
                    print("epoch {} iter {} lr {} , loss : {}".format(self.epoch, self.iter, get_lr(self.optimizer), loss.item()))
                    self.torch_writer.add_scalar("lr", get_lr(self.optimizer), self.iter)
                    self.torch_writer.add_scalar('example/sec', latency, self.iter)
                    for ln, ls in zip(loss_name, total_loss):
                        self.torch_writer.add_scalar(ln, ls, self.iter)
            
        return loss.item()