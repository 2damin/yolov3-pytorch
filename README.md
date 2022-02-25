# YOLOv3-pytorch

single stage object detection Yolov3.

This is made with Pytorch.


<img src=https://production-media.paperswithcode.com/methods/Screen_Shot_2020-06-24_at_12.52.19_PM_awcwYBa.png width=416>

----------------------------

## Install

### Windows

#### Use Anaconda

1. Download Anaconda : https://www.anaconda.com/products/individual#windows
2. conda create --name ${environment_name} python=3.8
3. activate ${environment_name}
4. git clone https://github.com/2damin/yolov3-pytorch.git


### Linux

#### Use docker

I recommend Nvidia NGC docker image. [link](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)

1. docker pull nvcr.io/nvidia/pytorch:xx.xx-py3
2. docker run --gpus all -it --rm -v local_dir:container_dir -p 8888:8888 nvcr.io/nvidia/pytorch:xx.xx-py3
   1. check "nvidia-smi"
   2. check "nvcc --version"
3. git clone https://github.com/2damin/yolov3-pytorch.git


## Dependency

```
python >= 3.6

Numpy

torch >= 1.9

torchvision >= 0.10

tensorboard

tensorboardX

torchsummary

pynvml

imgaug
```

-------------------

## Run

If training,

```{r, engine='bash', count_lines}
(single gpu) python main.py --mode train --cfg ./cfg/yolov3.cfg --gpus 0 --checkpoint ${saved_checkpoint_path}

(multi gpu) python main.py --mode train --cfg ./cfg/yolov3.cfg --gpus 0 1 2 3 --checkpoint ${saved_checkpoint_path}
```

If test,

```{r, engine='bash', count_lines}
python main.py --mode test --cfg ./cfg/yolov3.cfg --gpus 0 --checkpoint ${saved_checkpoint_path}
```

### option

--mode : train/test.

--cfg : the path of model.cfg.

--gpu : if you use GPU, set 1. If you use CPU, set 0.

--checkpoint : the path of saved model checkpoint. If you want to load the previous train, or if you test(evaluate) the model.



## Visualize training graph

Using Tensorboard,

```{r, engine='bash', count_lines}
tensorboard --logdir=./output --port 8888
```

-------------------------

# Reference

[YOLOv3 paper](https://arxiv.org/abs/1804.02767)

[KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d)