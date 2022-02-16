import torch
import numpy
import PIL, time
from train.loss import *

class Evaluator:
    def __init__(self, model, eval_loader, device, hparam):
        self.model = model
        self.eval_loader = eval_loader
        self.device = device
        self.yololoss = YoloLoss(self.device, self.model.n_classes)

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

                # for bidx, b in enumerate(targets['bbox'][0]):
                #     new_box_x1 = b[0] - b[2]/2
                #     new_box_y1 = b[1] - b[3]/2
                #     new_box_x2 = b[0] + b[2]/2
                #     new_box_y2 = b[1] + b[3]/2
                #     print(targets['bbox'][0][bidx])
                #     targets['bbox'][0][bidx] = torch.tensor(np.array([new_box_x1,new_box_y1,new_box_x2,new_box_y2]) * 416, dtype=torch.int32) 

                # drawBox(input_img.detach().cpu().numpy()[0,:,:,:], targets['bbox'][0], mode=1)
                
                for best_box in best_box_list:
                    if best_box is None:
                        continue
                    print("nms after shape : ", best_box.shape)
                    print(best_box)
                    drawBox(input_img.detach().cpu().numpy()[0,:,:,:], best_box[:,], mode=1)
