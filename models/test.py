import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        # dw
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))

        # pw-linear
        layers.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(oup))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


# 绘制 MobileNetV2 的倒残差结构
fig, ax = plt.subplots(figsize=(8, 6))

# 绘制输入
ax.text(0, 0, "Input", ha="center", va="center", bbox=dict(facecolor="lightgray", edgecolor="black", boxstyle="round"))

# 绘制第一个 Inverted Residual 块
ax.text(1, 0, "Expand", ha="center", va="center", bbox=dict(facecolor="lightgray", edgecolor="black", boxstyle="round"))
ax.text(2, 0, "Depthwise\nConv", ha="center", va="center",
        bbox=dict(facecolor="lightgray", edgecolor="black", boxstyle="round"))
ax.text(3, 0, "Project", ha="center", va="center",
        bbox=dict(facecolor="lightgray", edgecolor="black", boxstyle="round"))
ax.arrow(0.5, 0, 0.5, 0, head_width=0.1, head_length=0.1, fc="k", ec="k")
ax.arrow(1.5, 0, 0.5, 0, head_width=0.1, head_length=0.1, fc="k", ec="k")
ax.arrow(2.5, 0, 0.5, 0, head_width=0.1, head_length=0.1, fc="k", ec="k")

# 绘制残差连接
ax.arrow(0, 0, 3, 0, head_width=0.1, head_length=0.1, fc="k", ec="k", linestyle="--")

# 设置坐标轴和标题
ax.set_xlim(0, 4)
ax.set_ylim(-1, 1)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("MobileNetV2 Inverted Residual Block")

plt.show()