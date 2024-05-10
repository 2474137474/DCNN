"""mobilenet in pytorch



[1] Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam

    MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
    https://arxiv.org/abs/1704.04861
"""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class sa_layer(nn.Module):
    """Constructs a Channel Spatial Group module.

    Args:
        k_size: Adaptive selection of kernel size
    """
    '''这里的channel是groups的4倍'''
    def __init__(self, channel, groups=32):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out


class Scale(nn.Module):
    '''这里的channel是第一个深度可分离卷积输入的channel'''
    def __init__(self, channel):
        super(Scale, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, stride=2, groups=channel, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(channel)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, groups=channel, padding=1, bias=False)
        self.conv4 = nn.Conv2d(channel, channel*2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(channel*2)
        self.relu4 = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        return x

class SE(nn.Module):
    '''这里的channel就是输入特征图的channel'''
    def __init__(self, channel,reduction=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(-1, c, 1, 1)
        # return y.expand_as(x)
        return y

def channel_split(x, split):
    # 原论文没有使用随机分，而是顺序的torch.split，这里可以考虑先生成一个随机的索引表，随后随机split
    """x是输入特征图，n是要分割的通道数"""
    assert x.size(1) == split*4
    return torch.split(x, split, dim=1)

class PSC(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size, **kwargs):
        super(PSC, self).__init__()
        self.psc = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1,**kwargs),
            nn.BatchNorm2d(channel_in),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.psc(x)
        return x

class MSFF(nn.Module):
    def __init__(self, channel_in, **kwargs):
        super(MSFF, self).__init__()
        # PSC用的是深度可分离卷积
        self.psc1 = PSC(channel_in//4, channel_in//4, kernel_size=3, padding=1, groups=channel_in//4)
        self.psc2 = PSC(channel_in//4, channel_in//4, kernel_size=5, padding=2, groups=channel_in//4)
        self.psc3 = PSC(channel_in//4, channel_in//4, kernel_size=7, padding=3, groups=channel_in//4)
        self.psc4 = PSC(channel_in//4, channel_in//4, kernel_size=9, padding=4, groups=channel_in//4)
        self.se1 = SE(channel_in//4)
        self.se2 = SE(channel_in//4)
        self.se3 = SE(channel_in//4)


    def forward(self, x):
        x = channel_split(x, x.size(1)//4)
        x1 = self.psc1(x[0]) # S0'
        x2 = self.psc2(x[1]) # S1'
        x3 = self.psc3(x[2]) # S2'
        x4 = self.psc4(x[3]) # S3'
        x1 = x2 * self.se1(x1) # Z0'
        x2 = x3 * self.se2(x2) # Z1'
        x3 = x4 * self.se3(x3) # Z2'
        x4 = x4 # Z3'
        return torch.cat([x1, x2, x3, x4], 1)

class BasicConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(
            input_channels, output_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class DCNN(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.stem = nn.Sequential(
            BasicConv2d(3,64,kernel_size=7,padding=2,stride=4),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.sa1 = sa_layer(128)
        self.sa2 = sa_layer(128)
        self.sa3 = sa_layer(128)
        self.block1_1 = nn.Sequential(
            BasicConv2d(64, 64, 1, padding=0),
            MSFF(64),
            BasicConv2d(64, 128, 1, padding=0),
        )
        self.block1_2 = nn.Sequential(
            BasicConv2d(128, 64, 1, padding=0),
            MSFF(64),
            BasicConv2d(64, 128, 1, padding=0),
        )
        self.block1_3 = nn.Sequential(
            BasicConv2d(128, 64, 1, padding=0),
            MSFF(64),
            BasicConv2d(64, 128, 1, padding=0),
        )
        self.sc_t1 = Scale(128)
        self.sc_t2 = Scale(256)
        self.sc_t3 = Scale(512)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, num_classes)
        )
    def forward(self, x):
        '7x7 Conv & MaxPool'
        x = self.stem(x)
        'Block-1'
        x = self.block1_1(x)
        x = self.sa1(x)
        x = self.block1_2(x)
        x = self.sa2(x)
        x = self.block1_3(x)
        x = self.sa3(x)
        'SC-T1~3'
        x = self.sc_t1(x)
        x = self.sc_t2(x)
        x = self.sc_t3(x)
        'Classifier'
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    input = torch.randn(1, 3, 256, 256)
    model = DCNN()
    output = model(input)
    print(output.shape)

    from thop import profile
    from thop import clever_format
    macs, params = profile(model, inputs=(input, ))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)

