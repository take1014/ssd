#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import numpy as np
import utils.data_augumentation as da
import xml.etree.ElementTree as ET

# Annotation convertion
class Anno_xml2list():
    def __init__(self, classes):
        self.classes = classes

    def __call__(self, xml_path:str, width:int, height:int)->list:
        ret = list()
        xml = ET.parse(xml_path).getroot()

        for obj in xml.iter('object'):
            if int(obj.find('difficult').text):
                continue

            bndbox = list()

            # get object name
            name = obj.find('name').text.lower().strip()
            # get bounding box
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']

            for pt in pts:
                # VOC data is (-1,-1) origin. So convert to (0,0) origin
                cur_pixel = int(bbox.find(pt).text) - 1

                # normalize
                if pt == 'xmin' or pt == 'xmax':
                    cur_pixel /= width
                else:
                    cur_pixel /= height
                # add bounding box's location
                bndbox.append(cur_pixel)
            # get label's index
            label_idx = self.classes.index(name)
            # add bounding box's class index
            bndbox.append(label_idx)
            ret += [bndbox]
        return np.array(ret)

# Data Transform
class DataTransform():
    def __init__(self, input_size, color_mean):
        self.data_transform = {
                'train': da.Compose([
                    da.ConvertFromInts(),
                    da.ToAbsoluteCoords(),
                    da.PhotometricDistort(),
                    da.Expand(color_mean),
                    da.RandomSampleCrop(),
                    da.RandomMirror(),
                    da.ToPercentCoords(),
                    da.Resize(input_size),
                    da.SubtractMeans(color_mean)
                    ]),
                'val': da.Compose([
                    da.ConvertFromInts(),
                    da.Resize(input_size),
                    da.SubtractMeans(color_mean)
                    ])
                }
    def __call__(self, img, phase, boxes, labels):
        # Switch preprocessing depending on phase
        return self.data_transform[phase](img, boxes, labels)

# test code
if __name__ == '__main__':
    #===== import library for testing =====
    import config as cfg
    import cv2
    import matplotlib.pyplot as plt
    from dataset import make_datapath_list
    #======================================

    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(cfg.ROOT_PATH)

    image_file_path = train_img_list[0]
    img = cv2.imread(image_file_path)
    height, width, channels = img.shape
    print(img.shape)

    trasform_anno = Anno_xml2list(cfg.VOC_CLASSES)
    anno_list = trasform_anno(train_anno_list[0], width, height)
    print(anno_list)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    color_mean = (104, 117, 123)
    input_size = 300
    preprocesser = DataTransform(input_size, color_mean)
    # phase TRAIN
    phase = 'train'
    img_transformed, boxes, labels = preprocesser(img, phase, anno_list[:,:4], anno_list[:,4])
    plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
    plt.show()

    # phase VALIDATIOn
    phase = 'val'
    img_transformed, boxes, labels = preprocesser(img, phase, anno_list[:,:4], anno_list[:,4])
    plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
    plt.show()
