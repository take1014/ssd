#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import os
import config as cfg
import numpy as np
import cv2
from preprocess import Anno_xml2list, DataTransform

# pytorch library
#import torch.nn as nn
#import torch.nn.init as init
#import torch.nn.functional as F
#from torch.autograd import Function
import torch.utils.data as data
import torch

def create_pathlist(rootpath:str, id_names:list)->list:
    # create path template
    imgpath_template  = os.path.join(rootpath, 'JPEGImages' , '%s.jpg')
    annopath_template = os.path.join(rootpath, 'Annotations', '%s.xml')

    image_list = list()
    anno_list  = list()

    for line in open(id_names):
        file_id = line.strip()
        image_list.append(imgpath_template  % file_id)
        anno_list.append(annopath_template % file_id)

    return image_list, anno_list

def make_datapath_list(rootpath:str)->list:
    # get ids
    train_id_names = os.path.join(rootpath + 'ImageSets/Main/train.txt')
    val_id_names   = os.path.join(rootpath + 'ImageSets/Main/val.txt')

    # create list
    train_img_list , train_anno_list = create_pathlist(rootpath, train_id_names)
    val_img_list , val_anno_list = create_pathlist(rootpath, val_id_names)

    return train_img_list, train_anno_list, val_img_list, val_anno_list

class VOCDataset(data.Dataset):
    def __init__(self, img_list, anno_list, phase, transform, transform_anno)->None:
        self.img_list  = img_list
        self.anno_list = anno_list
        self.phase     = phase
        self.transform = transform
        self.transform_anno = transform_anno

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        im, gt, _, _ = self.pull_item(index)
        return im, gt

    def pull_item(self, index):
        image_file_path = self.img_list[index]
        img = cv2.imread(image_file_path)
        height, width, channels = img.shape

        anno_file_path = self.anno_list[index]

        # transform annotation
        anno_list = self.transform_anno(anno_file_path, width, height)

        # preproccesing
        img, boxes, labels = self.transform(img, self.phase, anno_list[:,:4], anno_list[:, 4])

        # convert BGR to RGB
        # permute (height, width, channels) to (channels, height, width)
        img = torch.from_numpy(img[:,:,(2,1,0)]).permute(2, 0, 1)

        # create set of bbox, labels
        gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return img, gt, height, width

# Override collate function
def od_collate_fn(batch):
    imgs = list()
    targets = list()
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))

    imgs = torch.stack(imgs, dim=0)

    return imgs, targets


# test code
if __name__ == '__main__':
    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(cfg.ROOT_PATH)

    color_mean = (104, 117, 123)
    input_size = 300

    train_dataset = VOCDataset(train_img_list, train_anno_list, phase='train',
                               transform=DataTransform(input_size, color_mean),
                               transform_anno=Anno_xml2list(cfg.VOC_CLASSES))

    val_dataset   = VOCDataset(val_img_list, val_anno_list, phase='val',
                               transform=DataTransform(input_size, color_mean),
                               transform_anno=Anno_xml2list(cfg.VOC_CLASSES))    #transform = DataTransform(input_size, color_mean)

    im, gt = val_dataset.__getitem__(1)
    print(gt)

    batch_size=4

    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=od_collate_fn)
    val_dataloader   = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=od_collate_fn)

    dataloaders_dict={'train':train_dataloader, 'val':val_dataloader}

    batch_iterator = iter(dataloaders_dict['val'])
    images, targets = next(batch_iterator)
    print(images.size())
    print(len(targets))
    print(targets[1].size())

    print(train_dataset.__len__())
    print(val_dataset.__len__())
