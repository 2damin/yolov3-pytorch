import time, os
import torch
import torch.optim as optim
import torch.profiler
import torch.utils.data
import torchvision.transforms as T

from util.tools import *
from train.loss import *


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
        self.yololoss = YoloLoss(self.device, self.model.n_classes)
        self.optimizer = optim.SGD(model.parameters(), lr=hparam['lr'], momentum=hparam['momentum'])
        
        if checkpoint is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            self.iter = checkpoint['iteration']

        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size = 25,
            gamma = 0.9)

    def run(self):
        # with torch.profiler.profile(
        #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/yolov3'),
        #     record_shapes=True,
        #     with_stack=True) as prof:
        while True:
            loss = self.run_iter()
            self.epoch += 1
            if self.epoch % 10 == 0:
                checkpoint_path = os.path.join("./output", "model_epoch" + str(self.epoch) + ".pth")
                torch.save({'epoch': self.epoch,
                            'iteration': self.iter,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': loss}, checkpoint_path)
            self.lr_scheduler.step()
            if self.max_batch <= self.iter:
                break

    def run_iter(self):
        for i, batch in enumerate(self.train_loader):
            _input_img, targets = batch #batch['img'], batch['target']
            
            #stack [1,c,h,w] images to make [n,c,h,w] image
            input_img = torch.stack(_input_img,0)
            input_wh = [input_img.shape[3], input_img.shape[2]]
            
            # print("input_img : ", input_img.shape, "target_bbox : ", targets[0]['bbox'].shape, "targets_cls : ", targets[0]['cls'].shape)
            # print(targets[0]['bbox'])
            # draw_boxes = targets[0]['bbox']
            # draw_boxes[:,0] *= input_wh[0]
            # draw_boxes[:,2] *= input_wh[0]
            # draw_boxes[:,1] *= input_wh[1]
            # draw_boxes[:,3] *= input_wh[1]
            # drawBox(input_img.detach().numpy()[0], draw_boxes)

            input_img = input_img.to(self.device)

            start_time = time.time()

            self.optimizer.zero_grad()
            output = self.model(input_img)
            
            losses = self.yololoss.compute_loss(output, targets, input_wh, self.model.yolo_layers)
            loss_name = ['total', 'x', 'y', 'w', 'h', 'conf', 'cls']
            total_loss = [[] for _ in loss_name]
            for _, loss in enumerate(losses):
                for k, l in enumerate(loss):
                    total_loss[k].append(l)
            total_loss = [sum(l) for l in total_loss]
                
            loss = total_loss[0]
            loss.backward()
            self.optimizer.step()
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