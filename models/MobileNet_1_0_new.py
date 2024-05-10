"""mobilenet in pytorch



[1] Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam

    MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
    https://arxiv.org/abs/1704.04861
"""

import torch
import torch.nn as nn


# 自蒸馏模块的bottleneck
def branchBottleNeck(channel_in, channel_out, kernel_size, **kwargs):
    middle_channel = channel_out // 4
    return nn.Sequential(
        nn.Conv2d(channel_in, middle_channel, kernel_size=1, stride=1),
        nn.BatchNorm2d(middle_channel),
        nn.ReLU(),

        nn.Conv2d(middle_channel, middle_channel, kernel_size=kernel_size, stride=kernel_size, **kwargs),
        nn.BatchNorm2d(middle_channel),
        nn.ReLU(),

        nn.Conv2d(middle_channel, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(),
    )


class DepthSeperabelConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                input_channels,
                input_channels,
                kernel_size,
                groups=input_channels,
                **kwargs),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True)
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x


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


class MobileNet(nn.Module):

    """
    Args:
        width multipler: The role of the width multiplier α is to thin
                         a network uniformly at each layer. For a given
                         layer and width multiplier α, the number of
                         input channels M becomes αM and the number of
                         output channels N becomes αN.
    """

    def __init__(self, width_multiplier=1, class_num=100):
       super().__init__()

       alpha = width_multiplier
       self.stem = nn.Sequential(
           BasicConv2d(3, int(32 * alpha), 3, padding=1, bias=False, stride=2),
           DepthSeperabelConv2d(
               int(32 * alpha),
               int(64 * alpha),
               3,
               padding=1,
               bias=False
           )
       )

       #downsample
       self.conv1 = nn.Sequential(
           DepthSeperabelConv2d(
               int(64 * alpha),
               int(128 * alpha),
               3,
               stride=2,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(128 * alpha),
               int(128 * alpha),
               3,
               padding=1,
               bias=False
           )
       )

       #downsample
       self.conv2 = nn.Sequential(
           DepthSeperabelConv2d(
               int(128 * alpha),
               int(256 * alpha),
               3,
               stride=2,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(256 * alpha),
               int(256 * alpha),
               3,
               padding=1,
               bias=False
           )
       )

       #downsample
       self.conv3 = nn.Sequential(
           DepthSeperabelConv2d(
               int(256 * alpha),
               int(512 * alpha),
               3,
               stride=2,
               padding=1,
               bias=False
           ),

           DepthSeperabelConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               padding=1,
               bias=False
           )
       )

       #downsample
       self.conv4 = nn.Sequential(
           DepthSeperabelConv2d(
               int(512 * alpha),
               int(1024 * alpha),
               3,
               stride=2,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(1024 * alpha),
               int(1024 * alpha),
               3,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(1024 * alpha),
               int(1024 * alpha),
               3,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(1024 * alpha),
               int(1024 * alpha),
               3,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(1024 * alpha),
               int(1024 * alpha),
               3,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(1024 * alpha),
               int(1024 * alpha),
               3,
               padding=1,
               bias=False
           )
       )
       self.conv5 = nn.Sequential(
           DepthSeperabelConv2d(
               int(1024 * alpha),
               int(2048 * alpha),
               3,
               stride=2,
               padding=1,
               bias=False
           ),
          DepthSeperabelConv2d(
            int(2048 * alpha),
            int(2048 * alpha),
            3,
            padding=1,
            bias=False
          ),
          DepthSeperabelConv2d(
            int(2048 * alpha),
            int(2048 * alpha),
            3,
            padding=1,
            bias=False
          ),
          DepthSeperabelConv2d(
            int(2048 * alpha),
            int(2048 * alpha),
            3,
            padding=1,
            bias=False
          ),
       )

       self.fc = nn.Linear(int(2048 * alpha), class_num)
       self.avg = nn.AdaptiveAvgPool2d(1)

       self.avgpool1 = nn.AdaptiveAvgPool2d(1)
       self.avgpool2 = nn.AdaptiveAvgPool2d(1)
       self.avgpool3 = nn.AdaptiveAvgPool2d(1)
       self.avgpool4 = nn.AdaptiveAvgPool2d(1)
       self.avgpool5 = nn.AdaptiveAvgPool2d(1)
       self.bottleneck1 = branchBottleNeck(int(128 * alpha), int(1024 * alpha), 3, padding=0) # 64,512
       self.bottleneck2 = branchBottleNeck(int(256 * alpha), int(1024 * alpha), 3, padding=0) # 128,512
       self.bottleneck3 = branchBottleNeck(int(512 * alpha), int(4096 * alpha), 8, padding=1) # 256,2048
       self.bottleneck4 = branchBottleNeck(int(1024 * alpha), int(4096 * alpha), 4, padding=1) # 512,2048
       self.bottleneck5 = branchBottleNeck(int(2048 * alpha), int(4096 * alpha), 2, padding=0) # 1024,2048
       self.middle_fc1 = nn.Linear(512,class_num)
       self.middle_fc2 = nn.Linear(512,class_num)
       self.middle_fc3 = nn.Linear(2048,class_num)
       self.middle_fc4 = nn.Linear(2048,class_num)
       self.middle_fc5 = nn.Linear(2048,class_num)


    def forward(self, x):
        import matplotlib.pyplot as plt
        plt.subplot(1,2,1)
        x_batch = x[0,:,:,:].permute(1,2,0)
        plt.imshow(x_batch)
        plt.show()


        x = self.stem(x)
        #
        x = self.conv1(x)
        x = self.conv2(x)

        # for i in range(2,len(self.conv3)):
        #     x = self.conv3[i](x)
        x = self.conv3(x)

        images_batch = x.detach()
        images = images_batch[0, 0:3, :].permute(1, 2, 0)
        plt.subplot(1, 2, 2)
        plt.imshow(images)
        plt.show()

        # middle_output3 = self.bottleneck3(x)
        # middle_output3 = self.avgpool3(middle_output3)
        # middle_fea_3 = middle_output3
        # middle_output3 = torch.flatten(middle_output3, 1)
        # middle_output3 = self.middle_fc3(middle_output3)

        x = self.conv4(x)
        # middle_output4 = self.bottleneck4(x)
        # middle_output4 = self.avgpool4(middle_output4)
        # middle_fea_4 = middle_output4
        # middle_output4 = torch.flatten(middle_output4, 1)
        # middle_output4 = self.middle_fc4(middle_output4)

        x = self.conv5(x)
        # middle_output5 = self.bottleneck5(x)
        # middle_output5 = self.avgpool5(middle_output5)
        # middle_fea_5 = middle_output5
        # middle_output5 = torch.flatten(middle_output5, 1)
        # middle_output5 = self.middle_fc5(middle_output5)

        x = self.avg(x)
        # final_fea = x

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # return x, final_fea, middle_fea_3,middle_fea_4,middle_fea_5, middle_output3,middle_output4,middle_output5
        return x

def mobilenet_1_0_new(alpha=0.5, class_num=100):
    return MobileNet(alpha, class_num)

if __name__ == "__main__":
    model = mobilenet_1_0_new()
    print(model)
    input = torch.rand(1,3,224,224)
    output = model(input)
    print(output.size())

    # 计算该网络的参数量
    from thop import profile
    from thop import clever_format
    flops, params = profile(model, inputs=(input, ))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)

