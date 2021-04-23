#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import match

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

if __name__ == '__main__':
    loss = MultiBoxLoss()
