#!/usr/bin/env python3
#-*- coding:utf-8 -*-
from itertools import product as product
from math import sqrt as sqrt
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Function
from utils import match
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

def decode(loc, dbox_list):
    '''
    loc  = [d_cx,d_cy,d_width,d_height]
    DBox = [cx_dbox, cy_dbox, w_dbox, h_dbox]

    calc bounding box from offset information.
    cx = cx_dbox + 0.1*d_cx * w_dbox
    cy = cy_dbox + 0.1*d_cy * h_dbox
    w  = w_dbox * exp(0.2*d_w)
    h  = h_dbox * exp(0.2*d_h)
    '''

    boxes = torch.cat((
        dbox_list[:,2] + loc[:, :2] * 0.1 * dbox_list[:, 2:],   # [cx, cy]
        dbox_list[:,2:] * torch.exp(loc[:, 2:] * 0.2)), dim=1)  # [w, h]

    # [cx, cy, width, height] -> [xmin, ymin, xmax, ymax]
    boxes[:, :2] -= boxes[:, 2:] / 2    # [cx - width/2, cy - height/2] = [xmin, ymin]
    boxes[:, 2:] += boxes[:, :2]        # [xmin + width, xmax + height] = [xmax, ymax]

    return boxes

# Non-Maximum Suppression
def nm_suppression(boxes, scores, overlap=0.45, top_k=200):

    '''
    boxes  : [Counts of BBox.(confidence > 0.01), 4]
        BBox information.
    scores : [Counts of BBox.(confidence > 0.01)]
        Confidence information

    returns: keep:list, count : int
    '''

    count = 0
    keep = scores.new(scores.size(0)).zero_().long()

    x1 = boxes[:, 0] # xmin
    y1 = boxes[:, 1] # ymin
    x2 = boxes[:, 2] # xmax
    y2 = boxes[:, 3] # ymax

    # calc bounding box area
    area = torch.mul(x2-x1, y2-y1)

    tmp_x1 = boxes.new()
    tmp_y1 = boxes.new()
    tmp_x2 = boxes.new()
    tmp_y2 = boxes.new()
    tmp_w  = boxes.new()
    tmp_h  = boxes.new()

    # sort
    v, idx = scores.sort()

    # get top 200 indexs
    idx = idx[-top_k:]

    while idx.numel() > 0:
        # get top score's index
        i = idx[-1]

        #===== update outputs =====
        keep[count] = i
        count += 1
        #==========================

        if idx.size(0) == 1:
            break

        # update idx's list size
        idx = idx[:-1]
        torch.index_select(x1, 0, idx, out=tmp_x1)
        torch.index_select(y1, 0, idx, out=tmp_y1)
        torch.index_select(x2, 0, idx, out=tmp_x2)
        torch.index_select(y2, 0, idx, out=tmp_y2)

        # limits values
        tmp_x1 = torch.clamp(tmp_x1, min=x1[i])
        tmp_y1 = torch.clamp(tmp_y1, min=y1[i])
        tmp_x2 = torch.clamp(tmp_x2, max=x2[i])
        tmp_y2 = torch.clamp(tmp_y2, max=y2[i])

        tmp_w.resize_as_(tmp_x2)
        tmp_h.resize_as_(tmp_y2)

        tmp_w = tmp_x2 - tmp_x1
        tmp_h = tmp_y2 - tmp_y1

        tmp_w = torch.clamp(tmp_w, min=0.0)
        tmp_h = torch.clamp(tmp_h, min=0.0)

        inter = tmp_w * tmp_h

        rem_areas = torch.index_select(area, 0, idx)
        union     = (rem_areas - inter) + area[i]
        IoU = inter / union

        idx = idx[IoU.le(overlap)]
    return keep, count

class Detect(Function):
    def __init__(self, conf_thresh=0.01, top_k=200, nms_thresh=0.45):
        self.softmax = nn.Softmax(dim=-1)
        self.conf_thresh = conf_thresh
        self.top_k = top_k
        self.nms_thresh = nms_thresh

    def forward(self, loc_data, conf_data, dbox_list):

        '''
        loc_data  = [batch_num, 8732, 4]
            location information
        conf_data = [batch_num, 8732, num_classes]
            confidence information
        dbox_list = [8732, 4]
            DBox information
        '''
        num_batch   = loc_data.size(0)
        num_dbox    = loc_data.size(1)
        num_classes = conf_data.size(2)

        conf_data = self.softmax(conf_data)

        # create output template
        output = torch.zeros(num_batch, num_classes, self.top_k, 5)

        # [batch_num, 8732, num_classes]  ->  [batch_num, num_classes, 8732]
        conf_preds = conf_data.transpose(2, 1)

        for i in range(num_batch):
            # calc bounding box's [xmin, ymin, xmax, ymax]
            decoded_boxes = decode(loc_data[i], dbox_list)
            conf_scores = conf_preds[i].clone()
            for cl in range(1, num_classes):
                # create conf mask.  conf > conf_thresh
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]

                if scores.nelement()==0:
                    continue

                # l_mask:torch.Size([8732, 4])
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                # torch.Size([Bounding box's count, 4])
                boxes = decoded_boxes[l_mask].view(-1, 4)

                ids, count = nm_suppression(boxes, scores, self.nms_thresh, self.top_k)

                # update output
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)

        return output

class MultiBoxLoss(nn.Module):
    def __init__(self, jaccard_thresh=0.5, neg_pos=3, device='cpu'):
        super(MultiBoxLoss, self).__init__()
        self.jaccard_thresh = jaccard_thresh
        self.negpos_ratio   = neg_pos
        self.device         = device

    def forward(self, predictions, targets):
        """
        MultiBoxLoss
        Inputs
        ----------
        predictions:tupple
            (loc=torch.Size([num_batch, 8732, 4]), conf=torch.Size([num_batch, 8732, 21]), dbox_list=torch.Size [8732,4])

        targets : [num_batch, num_objs, 5]

        Outputs
        -------
        loss_l : loc  loss
        loss_c : conf loss
        """
        # loc_data  = torch.Size([num_batch, 8732, 4])
        # conf_data = torch.Size([num_batch, 8732, 21])
        # dbox_list = torch.Size [8732,4])
        loc_data, conf_data, dbox_list = predictions
        num_batch   = conf_data.size(0) # num_batch
        num_dbox    = conf_data.size(1) # 8732
        num_classes = conf_data.size(2) # 21

        conf_t_label = torch.LongTensor(num_batch, num_dbox).to(self.device)
        loc_t        = torch.Tensor(num_batch, num_dbox, 4).to(self.device)

        # Create true values. values: loc_t and conf_t_label
        for idx in range(num_batch):
            # targets : [num_batch, num_objs, 5(cx,cy,w,h,label)]
            # Get annotation BBox's(cx, cy, w, h) in now batch
            # truths = [[cx1,cy1, w1, h1], [cx2,cy2,w2,h2], ..., [cx8732, cy8732, w8732, h8732]]
            truths = targets[idx][:, :-1].to(self.device)   # BBox
            # labels = [label1, label2, ... , label8732]
            labels = targets[idx][:, -1].to(self.device)
            # dbox = [8732, 4(cx, cy, w, h)]
            dbox = dbox_list.to(self.device)

            # Match each prior box with the ground truth box of the highest jaccard
            # overlap, encode the bounding boxes, then return the matched indices
            # corresponding to both confidence and location preds.
            variance = [0.1, 0.2]
            match(self.jaccard_thresh, truths, dbox, variance, labels, loc_t, conf_t_label, idx)

        #===== Calc 'loc loss' just for Positive Box =====
        # pos_mask: torch.Size([num_batch, 8732])
        pos_mask = conf_t_label > 0
        # torch.Size([num_bath, 8732]) -> torch.Size([num_batch, 8732, 4])
        # pos_mask.dim() == 3
        pos_idx = pos_mask.unsqueeze(pos_mask.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)   # pred
        loc_t = loc_t[pos_idx].view(-1, 4)      # true
        # Calc loc loss
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')
        #==========================================

        # conf_data = torch.Size([num_batch, 8732, 21])
        # batch_conf = torch.Size([num_batch*8732, 21])
        batch_conf = conf_data.view(-1, num_classes)
        # Calc cross_entropy. conf_t_label.view(-1) = torch.Size([num_batch*8732])
        loss_c = F.cross_entropy(batch_conf, conf_t_label.view(-1), reduction='none')

        #===== Calc Positive Box's loss. =====
        # loss_c: torch.Size([num_batch, 8732])
        loss_c = loss_c.view(num_batch, -1)
        # Set positive box confidence loss to 0.
        loss_c[pos_mask] = 0
        #====================================

        # Hard Negative Mining
        # loss_idx: torch.Size([num_batch, 8732])
        # _: sorted Tensor, loss_idx: indices
        _, loss_idx = loss_c.sort(dim=1, descending=True)

        # idx_rank: torch.Size([num_batch, 8732])
        # _: sorted Tensor, idx_rank: indices(sorted)
        _, idx_rank = loss_idx.sort(dim=1)

        # pos_mask: torch.Size([num_batch, 8732])
        # num_pos: torch.Size([num_batch, 1])
        num_pos = pos_mask.long().sum(dim=1, keepdim=True)
        # num_neg: torch.Size([num_batch, 1])
        num_neg = torch.clamp(num_pos*self.negpos_ratio, max=num_dbox)
        neg_mask = idx_rank < (num_neg).expand_as(idx_rank)

        # pos_idx_mask : torch.Size([num_batch, 8732, 21])
        pos_idx_mask = pos_mask.unsqueeze(2).expand_as(conf_data)
        # neg_idx_mask : torch.Size([num_batch, 8732, 21])
        neg_idx_mask = neg_mask.unsqueeze(2).expand_as(conf_data)

        # Get positive and negative conf data and create conf_hnm.
        # conf_p_hnm: torch.Size([num_pos+neg_pos, 21])
        # gt(0): greater than 0
        conf_p_hnm = conf_data[(pos_idx_mask+neg_idx_mask).gt(0)].view(-1, num_classes)
        # conf_t_label_hnm: torch.Size([pos+neg])
        # gt(0): greater than 0
        conf_t_label_hnm = conf_t_label[(pos_mask+neg_mask).gt(0)]

        loss_c = F.cross_entropy(conf_p_hnm, conf_t_label_hnm, reduction='sum')

        # Summary of positive box
        N = num_pos.sum()

        # Calc average
        loss_l /= N
        loss_c /= N

        return loss_l, loss_c

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

    def forward(self, x):
        sources = list()
        loc     = list()
        conf    = list()

        for k in range(23):
            x = self.vgg[k](x)

        source1 = self.L2Norm(x)
        sources.append(source1)

        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)

        sources.append(x)

        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        # loc size  -> torch.Size([batch_num, 34928])
        loc  = torch.cat([o.view(o.size(0), -1) for o in loc],1)
        # conf size -> torch.Size([batch_num, 183372])
        conf = torch.cat([o.view(o.size(0), -1) for o in conf],1)

        # loc size  -> torch.Size([batch_num, 8732, 4])
        loc  = loc.view(loc.size(0), -1, 4)
        # conf size -> torch.Size([batch_num, 8732, 21])
        conf = conf.view(conf.size(0), -1, self.num_classes)

        # create output:tupple
        output = (loc, conf, self.dbox_list)

        if self.phase == 'inference':
            # torch.Size([batch_num, 21, 200, 5])
            return self.detect(output[0], output[1], output[2])
        else:
            # tupple (loc, conf, dbox_list)
            return output

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

    loss = MultiBoxLoss()

