#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import os
import random
import time

import cv2
import numpy  as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch .utils.data as data

# my modules
from dataset import make_datapath_list, VOCDataset, od_collate_fn
from preprocess import DataTransform, Anno_xml2list
import config as cfg
from ssd_model import SSD
from ssd_loss import MultiBoxLoss

def train():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(cfg.ROOT_PATH)

    train_dataset = VOCDataset(train_img_list, train_anno_list, phase='train', transform=DataTransform(cfg.INPUT_SIZE, cfg.COLOR_MEAN), transform_anno=Anno_xml2list(cfg.VOC_CLASSES))
    val_dataset   = VOCDataset(val_img_list, val_anno_list, phase='train', transform=DataTransform(cfg.INPUT_SIZE, cfg.COLOR_MEAN), transform_anno=Anno_xml2list(cfg.VOC_CLASSES))


    train_dataloader = data.DataLoader(
            train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=od_collate_fn )
    val_dataloader = data.DataLoader(
            val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=od_collate_fn )


    dataloaders_dict = {'train':train_dataloader, 'val':val_dataloader}


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

    net = SSD(phase='train', ssd_cfg=ssd_cfg)

    vgg_weights = torch.load('./weights/vgg16_reducedfc.pth')
    net.vgg.load_state_dict(vgg_weights)

    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    net.extras.apply(weights_init)
    net.loc.apply(weights_init)
    net.conf.apply(weights_init)

    criterion = MultiBoxLoss(jaccard_thresh=0.5, neg_pos=3, device=device)
    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)


    net.to(device)
    torch.backends.cudnn.benchmark=True

    iteration = 1
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0
    logs=[]

    for epoch in range(cfg.EPOCH+1):
        t_epoch_start = time.time()
        t_iter_start  = time.time()
        print("--------------------")
        print('Epoch {}/{}'.format(epoch+1, cfg.EPOCH))
        print("--------------------")

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
                print('(train)')
            else:
                if((epoch+1) % 10 == 0):
                    net.eval()
                    print('----------')
                    print('(val)')
                else:
                    continue

            for images, targets in dataloaders_dict[phase]:
                images  = images.to(device)
                targets = [ann.to(device) for ann in targets]

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(images)

                    # Calc loss
                    loss_l, loss_c = criterion(outputs, targets)
                    loss = loss_l + loss_c

                    if phase == 'train':
                        loss.backward()
                        # Clipping gradiendt parameters
                        nn.utils.clip_grad_value_(net.parameters(), clip_value=2.0)
                        # update parameters
                        optimizer.step()

                        if (iteration % 10 == 0):
                            t_iter_finish = time.time()
                            duration = t_iter_finish = t_iter_start
                            print('Iteration {} || Loss: {:.4f} || 10iter: {:.4f} sec'.format(iteration, loss.item(), duration))
                            t_iter_start = time.time()
                        epoch_train_loss += loss.item()
                        iteration += 1
            t_epoch_finish = time.time()
            print('-------------')
            print('epoch {} || Epoch_TRAIN_Loss:{:.4f} ||Epoch_VAL_Loss:{:.4f}'.format(
                epoch+1, epoch_train_loss, epoch_val_loss))
            print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
            t_epoch_start = time.time()

            log_epoch = {'epoch': epoch+1,
                         'train_loss': epoch_train_loss, 'val_loss': epoch_val_loss}
            logs.append(log_epoch)
            df = pd.DataFrame(logs)
            df.to_csv("log_output.csv")

            epoch_train_loss = 0.0
            epoch_val_loss = 0.0

            if ((epoch+1) % 10 == 0):
                torch.save(net.state_dict(), 'weights/ssd300_' + str(epoch+1) + '.pth')

if __name__ == '__main__':
    train()
    #print(torch.manual_seed(1234))
    #print(np.random.seed(1234))
    #print(random.seed(1234))
