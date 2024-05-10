import copy
import os
import sys
import argparse

import numpy as np
import torch

from utils import *
from conf import settings
import torch.nn.functional as F

from models.MobileNet_1_0_new import mobilenet_1_0_new


def cal_ev(output):
    return 1 - (np.sum(output * np.log(output)) / np.log(1 / J))


def eval(model, images, labels, choice='stu', empty_flag=False):
    if empty_flag:
        return 0, 0
    model.eval()
    test_loss = 0.0
    correct = 0.0
    if choice == 'stu':
        output,x_transfer = model(images)
    elif choice == 'tea':
        output = model(images)
    _, preds = output.max(1)
    loss = loss_function_s(output, labels)
    test_loss += loss.item()
    correct += preds.eq(labels).sum()
    return correct, test_loss



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-stu', type=str, default="MobileNet_0_5", help='net type')
    parser.add_argument('-tea', type=str, default="MobileNet_1_0", help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-GPU_num', type=int, default=1, help='GPU_num')
    parser.add_argument('-t', type=int, default=2, help='distillation temperature')
    args = parser.parse_args()


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


    model_s,model_t = get_networks(args)
    state_dict_s = torch.load("./checkpoint/MobileNet_0_5-178-best-0.7249.pth", map_location='cpu')
    state_dict_t = torch.load("./checkpoint/MobileNet_1_0-185-best-0.7139.pth", map_location='cpu')

    # 对于state_dict中存在的key，但是model中不存在的key，需要删除
    new_state_dict = copy.deepcopy(state_dict_s)
    for key in state_dict_s:
        if key not in model_s.state_dict():
            new_state_dict.pop(key)
    model_s.load_state_dict(new_state_dict)

    new_state_dict = copy.deepcopy(state_dict_t)
    for key in state_dict_t:
        if key not in model_t.state_dict():
            new_state_dict.pop(key)
    model_t.load_state_dict(new_state_dict)


    args.device = torch.device('cpu')
    model_s = model_s.to(args.device)
    model_t = model_t.to(args.device)
    loss_function_s = nn.CrossEntropyLoss()
    loss_function_t = nn.CrossEntropyLoss()

    # model.stem[0]   model.stem[0].conv1
    # stem,conv1这些是模型这个类的属性，stem[0]这个[0]是sequential的第1个元素，所以用[]来索引
    # 内部的stem[0].conv是因为这个conv是BasicConv2d的属性，所以用.来索引


    old_state_dict = model_t.state_dict()
    model_t_new = mobilenet_1_0_new()
    # 把模型的参数复制到新模型
    model_t_new.load_state_dict(old_state_dict)

    # acc_s, acc_t = eval(model_s,model_t_new)
    # print(acc_s, acc_t)

    J = 100
    threshold = 0.65  # 置信度超过这个值，则直接分类，不进行传递

    # 学生进行分类的总数,正确数
    s_num = 0
    t_num = 0
    correct_s = 0
    correct_t = 0
    # 准确率、损失
    correct = 0.0
    test_loss = 0.0

    # 取cifar100的一个batch
    for i, (images, labels) in enumerate(cifar100_training_loader):
        if i == 5:
            break
        correct_s_bs = 0; correct_t_bs = 0; test_loss_s = 0.0; test_loss_t = 0.0
        image_s = []; label_s = [] # 直接分类
        image_t = []; label_t = [] # 传递给教师网络

        # if i != 2:
        #     images = images.to(args.device)
        # else:
        #     break
        # debug测试提取到的特征图
        output_s, x_transfer = model_s(images)
        output_t = model_t_new(images)


        output_s, x_transfer = model_s(images)
        output_s = F.softmax(output_s, dim=1)
        output_s = output_s.detach().numpy()
        # 处理一个batch的输入和输出
        for image,label,output,feature_map in zip(images,labels,output_s,x_transfer):
            e_v = cal_ev(output)
            # print(f"学生网络的置信度为{e_v}")
            if e_v > threshold:
                print("直接分类")
                image_s.append(image)
                label_s.append(label)
            else:
                print("传递给教师网络")
                image_t.append(feature_map)
                label_t.append(label)

        empty_flag_s = False
        empty_flag_t = False
        # 如果全部由学生或全部由教师，额外处理一下
        if len(image_s) == 0:
            empty_flag_s = True
        else:
            image_s = torch.stack(image_s)

        if len(image_t) == 0:
            empty_flag_t = True
        else:
            image_t = torch.stack(image_t)

        label_s = torch.tensor(label_s)
        label_t = torch.tensor(label_t)

        correct_s_bs, test_loss_s = eval(model_s, image_s, label_s, 'stu',empty_flag_s)
        correct_t_bs, test_loss_t = eval(model_t_new, image_t,label_t, 'tea',empty_flag_t)

        s_num += len(label_s)
        t_num += len(label_t)
        correct_s += correct_s_bs
        correct_t += correct_t_bs

        correct += correct_s_bs + correct_t_bs
        test_loss += test_loss_s + test_loss_t

    print(f"学生进行分类的总数为{s_num}\t正确数为{correct_s}\t准确率为{correct_s/s_num}")
    print(f"传递给教师网络的总数为{t_num}\t正确数为{correct_t}\t准确率为{correct_t/t_num}")
    print(f"总分类数为{s_num+t_num}\t正确数为{correct}\t"
          f"总的准确率为{correct/(s_num+t_num)}\t"
          f"总的损失为{test_loss/(s_num+t_num)}")
