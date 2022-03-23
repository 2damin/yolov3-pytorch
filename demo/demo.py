import torch
import time
from train.loss import *

class Demo:
    def __init__(self, model, data, data_loader, device, hparam):
        self.model = model
        self.class_list = data.class_str
        self.data_loader = data_loader
        self.device = device
        self.yololoss = YoloLoss(self.device, self.model.n_classes, hparam['ignore_cls'])
        self.preds = None

    def run(self):
        for i, batch in enumerate(self.data_loader):
            input_img, _, _ = batch
            
            input_img = input_img.to(self.device, non_blocking=True)

            with torch.no_grad():
                start_time = time.time()
            
                output = self.model(input_img)
                            
                # _, output_list = self.yololoss.compute_loss(output, targets = None, nw = nw, nh = nh, yolo_layers = self.model.yolo_layers)

                output_all = torch.cat(output, dim=1)
                print(output_all.shape)
                best_box_list = non_max_sup(output_all, self.model.n_classes, conf_th=0.5, nms_th=0.5)
                
                if best_box_list is None:
                    continue
                print(best_box_list.shape)
                final_box_list = best_box_list[best_box_list[:,4] > 0.85]
                print("final :", final_box_list.shape)
                if i % 100 == 0:
                    print("-------{} th iter -----".format(i))
                if final_box_list is None:
                    continue
                show_img = input_img.detach().cpu().numpy()[0,:,:,:]
                drawBox(show_img, best_box_list, best_box_list[:,6], mode=1)

        
