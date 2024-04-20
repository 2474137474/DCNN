import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import *
from conf import settings


# def train(epoch,model_s,model_t):
#
#     model_s.train()
#     model_t.train()
#
#     for batch_index, (images, labels) in enumerate(cifar10_training_loader):
#
#         if args.gpu:
#             labels = labels.cuda()
#             images = images.cuda()
#
#
#         output_s,middle_fea_s,middle_out_s = model_s(images)
#         output_t,middle_fea_t,middle_out_t = model_t(images)
#         loss_s = loss_function(output_s, labels)
#         loss_t = loss_function(output_t, labels)
#
#         middle_out_loss = loss_kd_output(middle_out_s,middle_out_t,t=2)
#         # middle_fea_loss = loss_kd_feature(middle_fea_s,middle_fea_t)
#
#         # total_loss = loss_s + loss_t + middle_out_loss + middle_fea_loss
#         total_loss = loss_s + loss_t + middle_out_loss
#
#         optimizer.zero_grad()
#         total_loss.backward()
#         optimizer.step()
#
#         n_iter = (epoch - 1) * len(cifar10_training_loader) + batch_index + 1
#
#
#         # print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tstu_loss:{:0.4f} \ttea_loss:{:0.4f} \tTotal_Loss: {:0.4f}\tLR: {:0.6f}'.format(
#         #     stu_loss=loss_s.item(),
#         #     tea_loss=loss_t.item(),
#         #     Total_Loss=total_loss.item(),
#         #     optimizer.param_groups[0]['lr'],
#         #     epoch=epoch,
#         #     trained_samples=batch_index * args.b + len(images),
#         #     total_samples=len(cifar10_training_loader.dataset)
#         # ))
#
#         print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\t'
#               '{stu_loss:0.4f}\t{tea_loss:0.4f}\t{total_loss:0.4f}\t{LR:0.6f}'.format(
#             epoch=epoch,
#             trained_samples=batch_index * args.b + len(images),
#             total_samples=len(cifar10_training_loader.dataset),
#             stu_loss = loss_s.item(),
#             tea_loss = loss_t.item(),
#             total_loss = total_loss.item(),
#             LR=optimizer.param_groups[0]['lr'],
#
#         ))
#
#         if epoch <= args.warm:
#             warmup_scheduler_s.step()
#             warmup_scheduler_t.step()



def train(epoch,model_s,model_t):

    model_s.train()
    model_t.train()

    for batch_index, (images, labels) in enumerate(cifar10_training_loader):

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()


        output_s,middle_fea_s,middle_out_s = model_s(images)
        output_t,middle_fea_t,middle_out_t = model_t(images)
        loss_s = loss_function(output_s, labels)
        loss_t = loss_function(output_t, labels)

        middle_out_loss = loss_kd_output(middle_out_s,middle_out_t,t=2)
        total_loss = loss_s + loss_t + middle_out_loss

        optimizer.zero_grad()
        # total_loss.backward()
        loss_s.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(cifar10_training_loader) + batch_index + 1

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\t'
              '{stu_loss:0.4f}\t{LR:0.6f}'.format(
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar10_training_loader.dataset),
            stu_loss = loss_s.item(),
            LR=optimizer.param_groups[0]['lr'],

        ))

        if epoch <= args.warm:
            warmup_scheduler_s.step()
            warmup_scheduler_t.step()





def eval(model_s,model_t):

    start = time.time()
    model_s.eval()
    model_t.eval()

    test_loss_s = 0.0 # cost function error
    test_loss_t = 0.0
    correct_s = 0.0
    correct_t = 0.0

    for (images, labels) in cifar10_test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        output_s = model_s(images)
        output_t = model_t(images)
        loss_s = loss_function(output_s, labels)
        loss_t = loss_function(output_t, labels)
        test_loss_s += loss_s.item()
        test_loss_t += loss_t.item()
        _, preds_s = output_s.max(1)
        _, preds_t = output_t.max(1)
        correct_s += preds_s.eq(labels).sum()
        correct_t += preds_t.eq(labels).sum()

    print('Stu_Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss_s / len(cifar10_test_loader.dataset),
        correct_s,
        len(cifar10_test_loader.dataset),
        100. * correct_s / len(cifar10_test_loader.dataset)
    ))

    print('Tea_Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss_t / len(cifar10_test_loader.dataset),
        correct_t,
        len(cifar10_test_loader.dataset),
        100. * correct_t / len(cifar10_test_loader.dataset)
    ))

    print()

    return correct_s.float() / len(cifar10_test_loader.dataset), correct_t.float() / len(cifar10_test_loader.dataset)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-stu', type=str, default="MobileNet_0_5", help='net type')
    parser.add_argument('-tea', type=str, default="MobileNet_1_0", help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=32, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()

    model_s,model_t = get_networks(args)
    print("model load successfully")

    # 计算该网络的参数量
    cal_para_flops(model_s)
    cal_para_flops(model_t)

    # input = torch.rand(1,3,224,224)
    # output,middle_fea_s,middle_output_s = model_s(input)
    # print(output.size())
    # print(middle_fea_s.size())
    # print(middle_output_s.size())


    # 数据集
    cifar10_training_loader = get_cifar10_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    cifar10_test_loader = get_cifar10_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )
    print("data load successfully")

    # 优化器、损失函数、学习率
    iter_per_epoch = len(cifar10_training_loader)
    # 学生
    loss_function_s = nn.CrossEntropyLoss()
    optimizer_s = optim.SGD(model_s.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler_s = optim.lr_scheduler.MultiStepLR(optimizer_s, milestones=settings.MILESTONES,
                                                     gamma=0.2)  # learning rate decay
    warmup_scheduler_s = WarmUpLR(optimizer_s, iter_per_epoch * args.warm)

    # 教师
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_t.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler_t = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES,
                                                     gamma=0.2)  # learning rate decay

    warmup_scheduler_t = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    acc_s_list = []
    acc_t_list = []
    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler_s.step(epoch)
            train_scheduler_t.step(epoch)
        train(epoch,model_s,model_t)
        acc_s,acc_t = eval(model_s,model_t)
        acc_s_list.append(acc_s)
        acc_t_list.append(acc_t)
        print('acc_s_list:',acc_s_list)
        print('acc_t_list:',acc_t_list)




