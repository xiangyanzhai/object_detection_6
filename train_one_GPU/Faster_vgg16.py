# !/usr/bin/python
# -*- coding:utf-8 -*-
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import torch.nn as nn
import torch

is_gpu = torch.cuda.is_available()


def cuda(x):
    if is_gpu:
        x = x.cuda()
    return x


# torch.backends.cudnn.benchmark =True
# torch.backends.cudnn.deterministic = True
from pytorch_object_detection.tool.config import Config

from pytorch_object_detection.tool.get_anchors import get_anchors
from pytorch_object_detection.tool.torch_ATC_test import AnchorTargetCreator
from pytorch_object_detection.tool.torch_PC_test import ProposalCreator
from pytorch_object_detection.tool.torch_PTC_test import ProposalTargetCreator
from maskrcnn_benchmark.layers import ROIAlign
from torchvision.models import vgg16
from pytorch_object_detection.tool.RPN_net import RPN_net
from pytorch_object_detection.tool.Fast_net import Fast_net
import torch.nn.functional as F
from pytorch_object_detection.tool.read_Data import Read_Data
from torch.utils.data import DataLoader

ce_loss = nn.CrossEntropyLoss()
roialign = ROIAlign((7, 7), 1 / 16., 2)


def SmoothL1Loss(net_loc_train, loc, sigma, num):
    t = torch.abs(net_loc_train - loc)
    a = t[t < 1]
    b = t[t >= 1]
    loss1 = (a * sigma) ** 2 / 2
    loss2 = b - 0.5 / sigma ** 2
    loss = (loss1.sum() + loss2.sum()) / num
    return loss


class Faster_Rcnn(nn.Module):
    def __init__(self, config):
        super(Faster_Rcnn, self).__init__()
        self.config = config
        self.Mean = torch.tensor(config.Mean, dtype=torch.float32)
        self.num_anchor = len(config.anchor_scales) * len(config.anchor_ratios)
        self.anchors = get_anchors(np.ceil(self.config.img_max / 16 + 1), self.config.anchor_scales,
                                   self.config.anchor_ratios)
        self.ATC = AnchorTargetCreator(n_sample=config.rpn_n_sample, pos_iou_thresh=config.rpn_pos_iou_thresh,
                                       neg_iou_thresh=config.rpn_neg_iou_thresh, pos_ratio=config.rpn_pos_ratio)
        self.PC = ProposalCreator(nms_thresh=config.roi_nms_thresh,
                                  n_train_pre_nms=config.roi_train_pre_nms,
                                  n_train_post_nms=config.roi_train_post_nms,
                                  n_test_pre_nms=config.roi_test_pre_nms, n_test_post_nms=config.roi_test_post_nms,
                                  min_size=config.roi_min_size)
        self.PTC = ProposalTargetCreator(n_sample=config.fast_n_sample,
                                         pos_ratio=config.fast_pos_ratio, pos_iou_thresh=config.fast_pos_iou_thresh,
                                         neg_iou_thresh_hi=config.fast_neg_iou_thresh_hi,
                                         neg_iou_thresh_lo=config.fast_neg_iou_thresh_lo)

        self.features = vgg16().features[:-1]
        self.rpn = RPN_net(512, self.num_anchor)
        self.fast = Fast_net(config.num_cls, 512 * 7 * 7, 4096)
        self.a = 0
        self.b = 0
        self.c = 0
        self.d = 0
        self.fast_num = 0
        self.fast_num_P = 0

    def rpn_loss(self, rpn_logits, rpn_loc, bboxes, tanchors, img_size):
        inds, label, indsP, loc = self.ATC(bboxes, tanchors, img_size)

        rpn_logits_train = rpn_logits[inds]
        rpn_loc_train = rpn_loc[indsP]
        rpn_cls_loss = ce_loss(rpn_logits_train, label)
        rpn_box_loss = SmoothL1Loss(rpn_loc_train, loc, 3.0, 240.0)
        self.a = rpn_cls_loss
        self.b = rpn_box_loss
        return rpn_cls_loss, rpn_box_loss

    def fast_train_data(self, loc, score, anchor, img_size, bboxes):
        roi = self.PC(loc, score, anchor, img_size)
        roi, loc, label = self.PTC(roi, bboxes[:, :4], bboxes[:, -1].long())
        roi_inds = torch.zeros((roi.size()[0], 1)).cuda(loc.device)

        roi = torch.cat([roi_inds, roi], dim=1)
        return roi, loc, label

    def fast_loss(self, fast_logits, fast_loc, label, loc):
        fast_num = label.shape[0]
        fast_num_P = loc.shape[0]
        fast_loc_train = fast_loc[torch.arange(fast_num_P), label[:fast_num_P].long()]

        fast_cls_loss = ce_loss(fast_logits, label.long())
        fast_box_loss = SmoothL1Loss(fast_loc_train, loc, 1.0, float(fast_num))
        self.c = fast_cls_loss
        self.d = fast_box_loss
        self.fast_num = fast_num
        self.fast_num_P = fast_num_P
        return fast_cls_loss, fast_box_loss

    def process_im(self, x, bboxes):
        x = x[None]
        x = x[..., [2, 1, 0]]
        x = x.permute(0, 3, 1, 2)
        H, W = x.shape[2:]
        ma = max(H, W)
        mi = min(H, W)
        scale = min(self.config.img_max / ma, self.config.img_min / mi)
        nh = int(H * scale)
        nw = int(W * scale)
        x = F.interpolate(x, size=(nh, nw))
        bboxes[:, :4] = bboxes[:, :4] * scale
        x = x.permute(0, 2, 3, 1)

        # NHWC RGB
        return x, bboxes

    def get_loss(self, x, bboxes, num_b, H, W):
        x = x.view(-1)[:H * W * 3].view(H, W, 3)
        bboxes = bboxes[:num_b]
        x = x.float()
        x = cuda(x)
        bboxes = cuda(bboxes)
        x, bboxes = self.process_im(x, bboxes)
        x = x - cuda(self.Mean)
        x = x.permute(0, 3, 1, 2)
        img_size = x.shape[2:]

        x = self.features(x)
        rpn_logits, rpn_loc = self.rpn(x)
        map_H, map_W = x.shape[2:]
        tanchors = self.anchors[:map_H, :map_W]
        tanchors = cuda(tanchors.contiguous().view(-1, 4))

        rpn_cls_loss, rpn_box_loss = self.rpn_loss(rpn_logits, rpn_loc, bboxes, tanchors, img_size)
        roi, loc, label = self.fast_train_data(rpn_loc.data, F.softmax(rpn_logits.data, dim=-1)[:, 1], tanchors,
                                               img_size, bboxes)

        x = roialign(x, roi)
        fast_logits, fast_loc = self.fast(x)
        fast_cls_loss, fast_box_loss = self.fast_loss(fast_logits, fast_loc, label, loc)
        return rpn_cls_loss + rpn_box_loss + fast_cls_loss + fast_box_loss

    def forward(self, imgs, bboxes, num_b, num_H, num_W):
        loss = list(map(self.get_loss, imgs, bboxes, num_b, num_H, num_W))
        return sum(loss)


def func(batch):
    m = len(batch)
    num_b = []
    num_H = []
    num_W = []
    for i in range(m):
        num_b.append(batch[i][2])
        num_H.append(batch[i][3])
        num_W.append(batch[i][4])

    max_b = max(num_b)
    max_H = max(num_H)
    max_W = max(num_W)
    imgs = []
    bboxes = []
    for i in range(m):
        imgs.append(batch[i][0].resize_(max_H, max_W, 3)[None])
        bboxes.append(batch[i][1].resize_(max_b, 5)[None])

    imgs = torch.cat(imgs, dim=0)
    bboxes = torch.cat(bboxes, dim=0)
    return imgs, bboxes, torch.tensor(num_b, dtype=torch.int64), torch.tensor(num_H, dtype=torch.int64), torch.tensor(
        num_W, dtype=torch.int64)


from datetime import datetime


def train(model, config, step, x, pre_model_file, model_file=None):

    model = model(config)
    model.eval()
    model_dic = model.state_dict()

    pretrained_dict = torch.load(pre_model_file, map_location='cpu')

    a = pretrained_dict['classifier.0.weight']
    b = pretrained_dict['classifier.0.bias']
    c = pretrained_dict['classifier.3.weight']
    d = pretrained_dict['classifier.3.bias']
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dic}
    print(len(pretrained_dict))
    model_dic.update(pretrained_dict)
    print(list(model_dic.keys()))
    model_dic['fast.fast_head.0.weight'] = a
    model_dic['fast.fast_head.0.bias'] = b
    model_dic['fast.fast_head.2.weight'] = c
    model_dic['fast.fast_head.2.bias'] = d
    model.load_state_dict(model_dic)

    if step > 0:
        model.load_state_dict(torch.load(model_file, map_location='cpu'))
        print(model_file)
    else:
        print(pre_model_file)

    train_params = list(model.parameters())
    for p in train_params[:8]:
        p.requires_grad = False

    cuda(model)
    train_params = list(model.parameters())

    lr = config.lr * config.batch_size_per_GPU
    if step >= 60000 * x:
        lr = lr / 10
    if step >= 80000 * x:
        lr = lr / 10
    print('lr        ******************', lr)
    print('weight_decay     ******************', config.weight_decay)

    if True:
        bias_p = []
        weight_p = []
        print(len(train_params))
        for name, p in model.named_parameters():
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        print(len(weight_p), len(bias_p))
        opt = torch.optim.SGD(
            [{'params': weight_p, 'weight_decay': config.weight_decay, 'lr': lr},
             {'params': bias_p, 'lr': lr * config.bias_lr_factor}],
            momentum=0.9, )
    else:
        bias_p = []
        weight_p = []
        bn_weight_p = []
        print(len(train_params))
        for name, p in model.named_parameters():
            print(name, p.shape)
            if len(p.shape) == 1:
                if 'bias' in name:
                    bias_p.append(p)
                else:
                    bn_weight_p.append(p)
            else:
                weight_p.append(p)
        print(len(weight_p), len(bias_p), len(bn_weight_p))
        opt = torch.optim.SGD([{'params': weight_p, 'weight_decay': config.weight_decay, 'lr': lr},
                               {'params': bn_weight_p, 'lr': lr},
                               {'params': bias_p, 'lr': lr * config.bias_lr_factor}],
                              momentum=0.9, )
    dataset = Read_Data(config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size_per_GPU, collate_fn=func,
                            shuffle=True, drop_last=True, pin_memory=True, num_workers=16)

    epochs = 10000
    flag = False
    print('start:  step=', step)
    for epoch in range(epochs):
        for imgs, bboxes, num_b, num_H, num_W in dataloader:
            loss = model(imgs, bboxes, num_b, num_H, num_W)
            loss = loss / imgs.shape[0]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 10, norm_type=2)
            opt.step()
            if step % 20 == 0:
                print(datetime.now(), 'loss:%.4f' % loss, 'rpn_cls_loss:%.4f' % model.a,
                      'rpn_box_loss:%.4f' % model.b,
                      'fast_cls_loss:%.4f' % model.c, 'fast_box_loss:%.4f' % model.d,
                      model.fast_num,
                      model.fast_num_P, step)
            step += 1

            if step == 60000 * x or step == 80000 * x:
                for param_group in opt.param_groups:
                    param_group['lr'] = param_group['lr'] / 10
                    print('*******************************************', param_group['lr'])

            if (step <= 10000 and step % 1000 == 0) or step % 5000 == 0 or step == 1:
                torch.save(model.state_dict(), './models/vgg16_%d_1.pth' % step)
            if step == 90010 * x:
                flag = True
                break
        if flag:
            break


if __name__ == "__main__":
    Mean = [123.68, 116.78, 103.94]
    path = '/home/zhai/PycharmProjects/Demo35/pytorch_Faster_tool/data_preprocess/'
    Bboxes = [path + 'Bboxes_07.pkl', path + 'Bboxes_12.pkl']
    img_paths = [path + 'img_paths_07.pkl', path + 'img_paths_12.pkl']
    files = [img_paths, Bboxes]
    config = Config(True, Mean, files, lr=0.001, weight_decay=0.0005, batch_size_per_GPU=1, img_max=1000, img_min=600,
                    bias_lr_factor=2)

    step = 0
    model = Faster_Rcnn
    x = 1
    pre_model_file = '/home/zhai/PycharmProjects/Demo35/py_Faster_tool/pre_model/vgg16_cf.pth'
    model_file = '/home/zhai/PycharmProjects/Demo35/pytorch_object_detection/train_one_GPU/models/vgg16_30000_1.pth'
    train(model, config, step, x, pre_model_file, model_file=model_file)
