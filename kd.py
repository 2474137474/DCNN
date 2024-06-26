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




def train(epoch,model_s,model_t):

    model_s.train()
    model_t.train()

    for batch_index, (images, labels) in enumerate(cifar100_training_loader):

        # labels = labels.cuda()
        # images = images.cuda()

        output_s, final_fea_s, \
        middle_fea_s_3, middle_fea_s_4, middle_fea_s_5, \
        middle_output_s_3, middle_output_s_4, middle_output_s_5 = model_s(images)

        output_t, final_fea_t, \
        middle_fea_t_3, middle_fea_t_4, middle_fea_t_5, \
        middle_output_t_3, middle_output_t_4, middle_output_t_5 = model_t(images)

        # output_s,middle_fea_s,middle_out_s = model_s(images)
        # output_t,middle_fea_t,middle_out_t = model_t(images)
        loss_s = loss_function_s(output_s, labels) # 这里CrossEntropyLoss()包含了logsoftmax+NllLoss
        loss_t = loss_function_t(output_t, labels)

        # 教师与学生输出层(未经过softmax)的KL散度Loss
        middle_out_loss = loss_kd_output(output_s,output_t,t=args.t)
        # 教师与学生 全连接层前的特征图 之间的Loss(借鉴自蒸馏损失的写法)
        middle_fea_loss = loss_kd_feature(middle_fea_s_5,middle_fea_t_5)

        # # 总loss，包括L(FL) + L(KL) + L(L2)
        # total_loss = loss_s + loss_t + middle_out_loss + middle_fea_loss

        # 教师loss为loss_t，只来自与标签的损失
        # 学生loss
        loss_s += middle_out_loss + middle_fea_loss

        optimizer_s.zero_grad()
        optimizer_t.zero_grad()
        # total_loss.backward()

        loss_s.backward(retain_graph=True)
        loss_t.backward()

        # 总loss，包括L(FL) + L(KL) + L(L2)
        total_loss = loss_s + loss_t + middle_out_loss + middle_fea_loss

        optimizer_s.zero_grad()
        optimizer_t.zero_grad()
        total_loss.backward()
        optimizer_s.step()
        optimizer_t.step()

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\t'
              'stu_loss:{stu_loss:0.4f}\t tea_loss:{tea_loss:0.4f}\t total_loss:{total_loss:0.4f}\t {LR:0.6f}'.format(
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar100_training_loader.dataset),
            stu_loss = loss_s.item(),
            tea_loss = loss_t.item(),
            # total_loss = loss_s.item()+loss_t.item(),
            total_loss = total_loss.item(),
            LR=optimizer_s.param_groups[0]['lr'],
        ))

        # tensorboard 记录相关信息
        # writer.add_scalar('Train_loss_s', loss_s.item(), n_iter)
        # writer.add_scalar('Train_loss_t', loss_t.item(), n_iter)
        # writer.add_scalar('Train_loss_total', total_loss.item(), n_iter)
        # writer.add_scalar('Middle_out_loss', middle_out_loss.item(), n_iter)
        # writer.add_scalar('Middle_fea_loss', middle_fea_loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler_s.step()
            warmup_scheduler_t.step()





def eval(model_s,model_t):

    start = time.time()
    model_s.eval()
    model_t.eval()

    test_loss_s = 0.0
    test_loss_t = 0.0
    correct_s = 0.0
    correct_t = 0.0

    for (images, labels) in cifar100_test_loader:

        images = images.cuda()
        labels = labels.cuda()

        output_s,middle_fea_s,middle_out_s = model_s(images)
        output_t,middle_fea_t,middle_out_t = model_t(images)
        loss_s = loss_function_s(output_s, labels)
        loss_t = loss_function_t(output_t, labels)
        test_loss_s += loss_s.item()
        test_loss_t += loss_t.item()
        _, preds_s = output_s.max(1)
        _, preds_t = output_t.max(1)
        correct_s += preds_s.eq(labels).sum()
        correct_t += preds_t.eq(labels).sum()

    print('Stu_Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss_s / len(cifar100_test_loader.dataset),
        correct_s,
        len(cifar100_test_loader.dataset),
        100. * correct_s / len(cifar100_test_loader.dataset)
    ))

    print('Tea_Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss_t / len(cifar100_test_loader.dataset),
        correct_t,
        len(cifar100_test_loader.dataset),
        100. * correct_t / len(cifar100_test_loader.dataset)
    ))

    # writer.add_scalar('Test_loss_s', test_loss_s / len(cifar100_test_loader.dataset), epoch)
    # writer.add_scalar('Test_Acc_s', correct_s.float() / len(cifar100_test_loader.dataset), epoch)
    # writer.add_scalar('Test_loss_t', test_loss_t / len(cifar100_test_loader.dataset), epoch)
    # writer.add_scalar('Test_Acc_t', correct_t.float() / len(cifar100_test_loader.dataset), epoch)


    return correct_s.float() / len(cifar100_test_loader.dataset), correct_t.float() / len(cifar100_test_loader.dataset)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-stu', type=str, default="MobileNet_0_5", help='net type')
    parser.add_argument('-tea', type=str, default="MobileNet_1_0", help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-GPU_num', type=int, default=1, help='GPU_num')
    parser.add_argument('-t', type=int, default=2, help='distillation temperature')
    args = parser.parse_args()

    model_s,model_t = get_networks(args)

    # args.device = torch.device(f'cuda:{args.GPU_num}')
    # torch.cuda.set_device(args.device)
    args.device = torch.device('cpu')

    model_s = model_s.to(args.device)
    model_t = model_t.to(args.device)

    print("model load successfully")

    # 计算该网络的参数量
    cal_para_flops(model_s)
    cal_para_flops(model_t)


    # 数据集
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )
    print("data load successfully")

    # 优化器、损失函数、学习率
    iter_per_epoch = len(cifar100_training_loader)
    # 学生
    loss_function_s = nn.CrossEntropyLoss()
    optimizer_s = optim.SGD(model_s.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler_s = optim.lr_scheduler.MultiStepLR(optimizer_s, milestones=settings.MILESTONES,
                                                     gamma=0.2)  # learning rate decay
    warmup_scheduler_s = WarmUpLR(optimizer_s, iter_per_epoch * args.warm)

    # 教师
    loss_function_t = nn.CrossEntropyLoss()
    optimizer_t = optim.SGD(model_t.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler_t = optim.lr_scheduler.MultiStepLR(optimizer_t, milestones=settings.MILESTONES,
                                                     gamma=0.2)  # learning rate decay

    warmup_scheduler_t = WarmUpLR(optimizer_t, iter_per_epoch * args.warm)

    # 获取权重保存路径
    checkpoint_path_s, checkpoint_path_t = get_checkpoint_path(settings, args)
    # 获取tensorboard路径
    # writer = SummaryWriter(log_dir=os.path.join(settings.LOG_DIR, args.stu+'-'+args.tea, settings.TIME_NOW))
    input_tensor = torch.Tensor(1,3,224,224).to(args.device)
    # writer.add_graph(model_s, input_tensor)
    # writer.add_graph(model_t, input_tensor)





    # acc_s_list = []
    # acc_t_list = []
    best_acc_s = 0.0
    best_acc_t = 0.0
    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler_s.step(epoch)
            train_scheduler_t.step(epoch)
        train(epoch,model_s,model_t)
        acc_s,acc_t = eval(model_s,model_t)
        if epoch > 0: # 120epoch之后开始保存
            if acc_s > best_acc_s:
                torch.save(model_s.state_dict(), checkpoint_path_s.format(net=args.stu, epoch=epoch, type='best', acc=acc_s))
                best_acc_s = acc_s
            if acc_t > best_acc_t:
                torch.save(model_t.state_dict(), checkpoint_path_t.format(net=args.tea, epoch=epoch, type='best', acc=acc_t))
                best_acc_t = acc_t


        # acc_s_list.append(acc_s)
        # acc_t_list.append(acc_t)

        print('max_stu_acc:', best_acc_s)
        print('max_tea_acc:', best_acc_t)

    # writer.close()

    




