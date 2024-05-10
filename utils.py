import os
import sys
import re
import datetime

import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

def get_networks(args):
    if args.stu == 'MobileNet_0_5' and args.tea == 'MobileNet_1_0':
        from models import mobilenet_0_5
        from models import mobilenet_1_0
        net_stu = mobilenet_0_5(class_num=100)
        net_tea = mobilenet_1_0(class_num=100)
        return net_stu, net_tea
    if args.stu == "MobileNet_1_0" and args.tea == "MobileNet_1_0":
        from models import mobilenet_1_0
        net_stu = mobilenet_1_0(class_num=100)
        net_tea = mobilenet_1_0(class_num=100)
        return net_stu



def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),

    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Resize((224, 224))
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader





def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]

def cal_para_flops(model):
    from thop import profile
    from thop import clever_format
    input = torch.rand(1, 3, 224, 224).to("cpu")
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)

def loss_kd_output(output_s,output_t, t): # 教师用softmax软化label，学生用log_softmax，因为nn.KLDivLoss接受的是
                                          #一个对数概率，一个概率
    """
    Compute the knowledge-distillation (KD) loss given outputs, where
    `output_s` and `output_t` are the outputs of the student and teacher
    models, respectively.
    计算经过softmax之后的KL散度损失
    """
    loss = nn.KLDivLoss()(F.log_softmax(output_s / t, dim=1),
                          F.softmax(output_t / t, dim=1)) * (t ** 2)
    return loss

# def loss_kd_output(output_s, output_t, t): # 自蒸馏论文的写法
#     output_s /= t
#     output_t /= t
#     output_s = torch.log_softmax(output_s, dim=1)
#     output_t = torch.softmax(output_t, dim=1)
#     # 如果两者比较接近，则值越大，加了负号就越小
#     loss = -torch.mean(torch.sum(output_t * output_s, dim=1)) * (t ** 2)
#     return loss

def loss_kd_feature(fea_s, fea_t):
    '''
    计算经过最后的全连接层之前的 特征图之间的 Loss
    '''
    loss = (fea_s-fea_t)**2 * ((fea_s>0) | (fea_t>0)).float()
    return torch.abs(loss).sum()

def get_checkpoint_path(settings, args):
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, settings.TIME_NOW)
    checkpoint_path_s = os.path.join(checkpoint_path, args.stu)
    checkpoint_path_t = os.path.join(checkpoint_path, args.tea)
    if not os.path.exists(checkpoint_path_s):
        os.makedirs(checkpoint_path_s)
    if not os.path.exists(checkpoint_path_t):
        os.makedirs(checkpoint_path_t)
    checkpoint_path_s = os.path.join(checkpoint_path_s, '{net}-{epoch}-{type}-{acc:0.2f}.pth')
    checkpoint_path_t = os.path.join(checkpoint_path_t, '{net}-{epoch}-{type}-{acc:0.2f}.pth')
    return checkpoint_path_s, checkpoint_path_t

# if __name__ == '__main__':
#     # output_s = torch.tensor([[0.9,0.05, 0.8]]) * 10
#     # output_t = torch.tensor([[0.1, 0.1, 0.8]]) * 10
#     # loss_kd_output = loss_kd_output(output_s, output_t, 3)
#     # print(loss_kd_output)
#
#     # 生成两个有正有负的tensor
#     fea_s = torch.randn(7,7,512)
#     fea_t = torch.randn(7,7,512)
#     loss_kd_feature = loss_kd_feature(fea_s, fea_t)
#     print(loss_kd_feature)

