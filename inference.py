#/usr/bin/env python3
#-*- coding:utf-8 -*-

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
#%matplotlib inline

import config as cfg
from ssd_model import SSD
from preprocess import DataTransform

def inference():
    ssd_cfg = {
            'num_classes':cfg.NUM_CLASSES,
            'input_size':cfg.INPUT_SIZE,
            'bbox_aspect_num':cfg.BBOX_ASPECT_NUM,
            'feature_maps':cfg.FEATURE_MAPS,
            'steps':cfg.STEPS,
            'min_sizes':cfg.MIN_SIZES,
            'max_sizes':cfg.MAX_SIZES,
            'aspect_ratios':cfg.ASPECT_RATIOS
            }

    net = SSD(phase='inference', ssd_cfg=ssd_cfg)

    # Load weights
    net_weights = torch.load('./weights/ssd300_50.pth',
                         map_location={'cuda:0': 'cpu'})

    net.load_state_dict(net_weights)


    image_file_path = "./data/cowboy-757575_640.jpg"
    img = cv2.imread(image_file_path)  # [height][width][channles(BGR)]
    height, width, channels = img.shape

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    transform = DataTransform(cfg.INPUT_SIZE, cfg.COLOR_MEAN)

    phase = "val"
    img_transformed, boxes, labels = transform(img, phase, "", "")  # アノテーションはないので、""にする
    img = torch.from_numpy(img_transformed[:, :, (2, 1, 0)]).permute(2, 0, 1)

    net.eval()
    x = img.unsqueeze(0)  # Convert to minibatch:torch.Size([1, 3, 300, 300])

    # Do inference
    detections = net(x)

    print(detections.shape)
    print(detections)

if __name__ == '__main__':
    inference()
