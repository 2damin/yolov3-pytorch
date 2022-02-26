import torch
import time
from train.loss import *

class Demo:
    def __init__(self, model, data, data_loader, device):
        self.model = model
        self.class_list = data.class_str
        self.data_loader = data_loader
        self.device = device
        self.yololoss = YoloLoss(self.device, self.model.n_classes)
        self.preds = None

    def run(self):
        for i, batch in enumerate(self.data_loader):
            input_img, _ = batch
            if self.device == torch.device('cuda'):
                input_img = input_img.cuda()
            input_wh = [input_img.shape[3], input_img.shape[2]]

            with torch.no_grad():
                start_time = time.time()
            
                output = self.model(input_img)
                            
                output_list = self.yololoss.compute_loss(output, targets = None, input_wh = input_wh, yolo_layers = self.model.yolo_layers)

                output_all = torch.cat(output_list, dim=1)
                best_box_list = non_max_sup(output_all, self.model.n_classes, conf_th=0.5, nms_th=0.4)
                #print(best_box_list)
                
                
                
                if i % 100 == 0:
                    print("-------{} th iter -----".format(i))
                
                if best_box_list is None:
                    continue
                
                drawBox(input_img.detach().cpu().numpy()[0,:,:,:], best_box_list, mode=1)

        
