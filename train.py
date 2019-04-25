"""
@author: autocyz
@contact: autocyz@163.com
@file: train.py
@function: train model
@time: 19-04-15
"""

import os
import time

import numpy as np
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataloader import DataLoader
from models.CornerNet import CornerNet, aeloss
from params import params
from sample.voc import VOC
from sample.cf import CF
from utils.torch_utils import save_params, get_lr
from utils.log import Logger

total_iter = 0


def train(train_loader, net, criterion, optimizer, epoch, writer, use_gpu=True, loader_info=''):
    """
    train model
    Args:
        train_loader: dataloader
        net: network
        criterion: loss function
        optimizer:
        epoch:
        writer:  summary writer
        loader_info:

    Returns:

    """
    time4 = 0
    net.train()
    for i, (img, tl_heatmap, br_heatmap, tl_tags, br_tags, tag_masks) in enumerate(train_loader):

        time4_last = time4
        time0 = time.time()
        if use_gpu:
            img = img.cuda()
            tl_heatmap = tl_heatmap.cuda()
            br_heatmap = br_heatmap.cuda()
            tl_tags = tl_tags.cuda()
            br_tags = br_tags.cuda()
            tag_masks = tag_masks.cuda()

        time1 = time.time()

        # predict is a list [tl_heat, br_heat, tl_tag, br_tag, ...]
        predict = net(*[img, tl_tags, br_tags])
        time2 = time.time()

        loss = criterion(predict, [tl_heatmap, br_heatmap, tag_masks])
        time3 = time.time()

        optimizer.zero_grad()
        loss.backward()
        # loss_paf.backward()
        optimizer.step()
        time4 = time.time()

        # writer some train information
        global total_iter
        total_iter += 1

        writer.add_scalar('train_loss', loss.item(), total_iter)

        if total_iter % 5 == 0:
            writer.add_image('0_img', img[0].cpu())
            tl_heatmap = tl_heatmap[0].cpu()
            br_heatmap = br_heatmap[0].cpu()
            tl_heatmap_predict = predict[-4][0].cpu()
            br_heatmap_predict = predict[-3][0].cpu()

            writer.add_image('1_tl_heatmap',
                             torchvision.utils.make_grid([tl_heatmap, tl_heatmap_predict,
                                                          br_heatmap, br_heatmap_predict],
                                                         normalize=True, padding=10,
                                                         pad_value=1))

        print('Epoch [{:03d}/{:03d}]\tStep [{}/{}  {:5d}]\tLr [{}]'
              '\tloss {:.4f}\n'
              'T_preprocess:{:.5f} T_forward:{:.5f} T_loss: {:.5f} T_backward:{:.5f}'.
              format(epoch, params['epoch_num'], i, len(train_loader), total_iter, get_lr(optimizer),
                     loss.item(),
                     time0 - time4_last, time2 - time1, time3 - time2, time4 - time3))


def eval(test_loader, net, criterion, epoch, writer, use_gpu=True):
    net.eval()
    val_loss = 0.
    with torch.no_grad():
        for i, (img, tl_heatmap, br_heatmap, tl_tags, br_tags, tag_masks) in enumerate(test_loader):
            if use_gpu:
                img = img.cuda()
                tl_heatmap = tl_heatmap.cuda()
                br_heatmap = br_heatmap.cuda()
                tl_tags = tl_tags.cuda()
                br_tags = br_tags.cuda()
                tag_masks = tag_masks.cuda()

            predict = net(*[img, tl_tags, br_tags])

            loss = criterion(predict, [tl_heatmap, br_heatmap, tag_masks])
            val_loss += loss
            print('Eval [{}/{}], epoch [{}]: current loss:{} calculate_loss:{}'.format(
                i, len(test_loader), epoch, loss.item(), val_loss.item()))
    val_loss = val_loss.item() / len(test_loader)
    writer.add_scalar('val_loss', val_loss, epoch)
    return val_loss


if __name__ == "__main__":
    root_dir = "/home/cyz/data/dataset/voc/VOCdevkit/VOC2012"

    date = params['date']
    if not os.path.exists('./result/logdir/' + date):
        os.mkdir('./result/logdir/' + date)
    logger = Logger("log", './result/logdir/' + date).get_logger()

    print('loading trainset')
    trainset = VOC(root_dir, "train", logger=logger)
    valset = VOC(root_dir, "val", logger=logger)
    train_loader = DataLoader(trainset, batch_size=params['batch_size'],
                              shuffle=True, num_workers=params['num_workers'])
    val_loader = DataLoader(valset, batch_size=params['batch_size'],
                            shuffle=True, num_workers=params['num_workers'])
    print("loading over")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    net = CornerNet()
    if params['pretrain_model']:
        print("loading pre_trained model :", params['pretrain_model'])
        params['has_checkpoint'] = True
        net.load_state_dict(torch.load(params['pretrain_model']))
        print("loading over")

    if params['use_gpu']:
        net = net.cuda()

    # optimizer = torch.optim.Adam(net.parameters(), lr=params['learning_rate'],
    #                              weight_decay=params['weight_decay'])
    optimizer = torch.optim.SGD(net.parameters(), lr=params['learning_rate'],
                                weight_decay=params['weight_decay'])
    lr_scheduler = StepLR(optimizer, step_size=params['step_size'], gamma=0.1)
    criterion = aeloss

    writer = SummaryWriter(log_dir='./result/logdir/' + date)
    save_model_path = os.path.join('./result/checkpoint/', date)
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)

    params['train_sample_nums'] = len(trainset)
    params['test_sample_nums'] = len(valset)
    params['train_iter_nums'] = len(train_loader)
    params['test_iter_nums'] = len(val_loader)
    save_params(save_model_path, 'parameter', params)

    best_loss = np.inf
    for epoch in range(params['epoch_num']):
        lr_scheduler.step()
        train(train_loader, net, criterion, optimizer, epoch, writer, use_gpu=params['use_gpu'])
        val_loss = eval(val_loader, net, criterion, epoch, writer, use_gpu=params['use_gpu'])
        print('epoch [{}] val_loss [{:.4f}]'.format(epoch, val_loss))
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(net.state_dict(), os.path.join(save_model_path, 'epoch_{}_{:.3f}.cpkt'.format(epoch, val_loss)))
