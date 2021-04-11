#!/usr/bin/env python3
#-*- coding:utf-8 -*-

ROOT_PATH = '/home/take/fun/dataset/VOC2012/'

VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair','cow', 'diningtable',
               'dog', 'horse', 'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

# DBox parameters
NUM_CLASSES     = 21
BBOX_ASPECT_NUM = [4,6,6,6,4,4]
FEATURE_MAPS    = [38,19,10,5,3,1]
STEPS           = [8,16,32,64,100,300]
MIN_SIZES       = [30, 60, 111, 162, 213, 264]
MAX_SIZES       = [60, 111, 162, 213, 264, 315]
ASPECT_RATIOS   = [[2], [2,3], [2,3],[2,3],[2],[2]]

#hyperparameters
BATCH_SIZE = 4
LR_BASE = 0.01
