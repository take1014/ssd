#!/usr/bin/env python3
#-*- coding:utf-8 -*-
from itertools import product as product
from math import sqrt as sqrt
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Function
import torch.utils.data as data
import torch

def make_vgg():
    layers=[]
    in_channels = 3

    vgg_cfg = [64,64,'M',128,128,'M',256,256,256,'MC', 512,512,512,'M',512,512,512]

    for v in vgg_cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'MC':
            # ceil_mode:   float -> round(float)
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

    return nn.ModuleList(layers)

def make_extras():
    layers = []
    in_channles = 1024
    extras_cfg = [256, 512, 128, 256, 128, 256, 128, 256]

    layers += [nn.Conv2d(in_channles, extras_cfg[0], kernel_size=(1))]
    layers += [nn.Conv2d(extras_cfg[0], extras_cfg[1], kernel_size=(3), stride=2, padding=1)]
    layers += [nn.Conv2d(extras_cfg[1], extras_cfg[2], kernel_size=(1))]
    layers += [nn.Conv2d(extras_cfg[2], extras_cfg[3], kernel_size=(3), stride=2, padding=1)]
    layers += [nn.Conv2d(extras_cfg[3], extras_cfg[4], kernel_size=(1))]
    layers += [nn.Conv2d(extras_cfg[4], extras_cfg[5], kernel_size=(3))]
    layers += [nn.Conv2d(extras_cfg[5], extras_cfg[6], kernel_size=(1))]
    layers += [nn.Conv2d(extras_cfg[6], extras_cfg[7], kernel_size=(3))]

    return nn.ModuleList(layers)

def make_loc_conf(num_classes=21, bbox_aspect_num=[4,6,6,6,4,4]):
    loc_layers=[]
    conf_layers=[]

    # set loc layers
    loc_layers  += [nn.Conv2d(512,  bbox_aspect_num[0] * 4, kernel_size=3, padding=1)]
    loc_layers  += [nn.Conv2d(1024, bbox_aspect_num[1] * 4, kernel_size=3, padding=1)]
    loc_layers  += [nn.Conv2d(512,  bbox_aspect_num[2] * 4, kernel_size=3, padding=1)]
    loc_layers  += [nn.Conv2d(256,  bbox_aspect_num[3] * 4, kernel_size=3, padding=1)]
    loc_layers  += [nn.Conv2d(256,  bbox_aspect_num[4] * 4, kernel_size=3, padding=1)]
    loc_layers  += [nn.Conv2d(256,  bbox_aspect_num[5] * 4, kernel_size=3, padding=1)]

    # set conf layers
    conf_layers += [nn.Conv2d(512,  bbox_aspect_num[0] * num_classes, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(1024, bbox_aspect_num[1] * num_classes, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512,  bbox_aspect_num[2] * num_classes, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256,  bbox_aspect_num[3] * num_classes, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256,  bbox_aspect_num[4] * num_classes, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256,  bbox_aspect_num[5] * num_classes, kernel_size=3, padding=1)]

    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)

class L2Norm(nn.Module):
    def __init__(self, input_channels=512, scale=20):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale
        self.reset_parameters()
        self.eps = 1e-10

    def reset_parameters(self):
        init.constant_(self.weight, self.scale)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = torch.div(x, norm)

        # First, we convert from Tensor-torch.Size([512]) to Tensor-torch.Size(1, 512, 1, 1).
        # Secondry, we convert from Tensor-torch.Size(1, 512, 1, 1) to Tensor-torch.Size([batch_num, 512, 38, 38])
        weights = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)

        return  weights * x

class DBox(object):
    def __init__(self, dbox_cfg):
        super(DBox, self).__init__()
        self.image_size    = dbox_cfg['input_size']
        self.feature_maps  = dbox_cfg['feature_maps']
        self.num_priors    = len(dbox_cfg['feature_maps'])
        self.steps         = dbox_cfg['steps']
        self.min_sizes     = dbox_cfg['min_sizes']
        self.max_sizes     = dbox_cfg['max_sizes']
        self.aspect_ratios = dbox_cfg['aspect_ratios']

    def make_dbox_list(self):
        mean = []
        # feature_maps = [38, 19, 10, 5, 3, 1]
        for k, f in enumerate(self.feature_maps):
            # feature image size
            f_k = self.image_size / self.steps[k]
            for i, j in product(range(f), repeat=2):
                # Default box center
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
                # minimum aspect default box
                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]
                # maximum aspect default box
                s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]
                # other default box
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]

        # torch.Size([8732,4])
        output = torch.Tensor(mean).view(-1,4)
        # set DBox's values range ->(min:0, max:1)
        output.clamp_(max=1, min=0)
        return output

class SSD(nn.Module):
    def __init__(self, phase, ssd_cfg):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = ssd_cfg['num_classes']
        self.vgg = make_vgg()
        self.extras = make_extras()
        self.L2Norm = L2Norm()
        self.loc, self.conf = make_loc_conf(ssd_cfg['num_classes'], ssd_cfg['bbox_aspect_num'])
        dbox = DBox(ssd_cfg)
        self.dbox_list = dbox.make_dbox_list()

        if phase == 'inference':
            self.detect = Detect()

if __name__ == '__main__':
    vgg_test = make_vgg()
    extras_test = make_extras()
    loc_test, conf_test = make_loc_conf()
    print(vgg_test)
    print(extras_test)
    print(loc_test)
    print(conf_test)

    #===== import for test =====
    import config as cfg
    import pandas as pd
    #==========================
    dbox_cfg = {
            'num_classes':cfg.NUM_CLASSES,
            'input_size':300,
            'bbox_aspect_num':cfg.BBOX_ASPECT_NUM,
            'feature_maps':cfg.FEATURE_MAPS,
            'steps':cfg.STEPS,
            'min_sizes':cfg.MIN_SIZES,
            'max_sizes':cfg.MAX_SIZES,
            'aspect_ratios':cfg.ASPECT_RATIOS
            }
    dbox = DBox(dbox_cfg)
    dbox_list = dbox.make_dbox_list()
    print(pd.DataFrame(dbox_list.numpy()))


    ssd_test = SSD(phase='train', ssd_cfg = dbox_cfg)
    print(ssd_test)
