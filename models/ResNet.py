import torch.nn as nn
import torch


# 首先定义34层残差结构
class BasicBlock(nn.Module):
    expansion = 1  # 对应主分支中卷积核的个数有没有发生变化

    # 定义初始化函数（输入特征矩阵的深度，输出特征矩阵的深度（主分支上卷积核的个数），不惧默认设置为1，下采样参数设置为None）
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    # 定义正向传播的过程
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)  # 将输入传入下采样函数得到捷径分支的输出
        # 主分支上的输出
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # 将主分支上的输出加上捷径分支上的输出
        out = self.relu(out)

        return out  # 得到残差结构的最终输出


# 定义50层、101层、152层的残差结构，在这个网络上进行修改得到ResNext网络
class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4  # 残差结构所使用卷积核的一个变化

    # 定义初始化函数
    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 # 相比resnet网络多传入了两个参数groups=1, width_per_group=64，
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        # 输入、输出特征矩阵的channel设置为width
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    # 定义正向传播过程
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


# 定义ResNet网络模型
class ResNet(nn.Module):

    def __init__(self,
                 block,  # 对应的就是残差结构
                 blocks_num,  # 所使用残差结构的数目
                 num_classes=1000,  # 训练集的分类个数
                 include_top=True,  # 是为了在ResNet网络上搭建更复杂的网络
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top  # 传入类变量之中
        self.in_channel = 64  # 输入特征矩阵的深度

        self.groups = groups
        self.width_per_group = width_per_group

        # 定义第一层的卷积层，3表示输入矩阵的深度
        # TODO:修改输入通道为1
        self.conv1 = nn.Conv2d(1, self.in_channel, kernel_size=7, stride=2,padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 最大池化下采样操作
        self.layer1 = self._make_layer(block, 64, blocks_num[0])  # 一系列残差结构
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应的平均池化下采样output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)  # 全连接层

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    # （哪一个残差结构，残差结构中第一卷积层所使用卷积核的个数，该层包含了几个残差结构，step为1）
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None  # 定义下采样
        if stride != 1 or self.in_channel != channel * block.expansion:  # 对于18层和34层的残差结构，就会跳过if语句;
            downsample = nn.Sequential(  # 生成下采样函数
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []  # 定义空的列表
        # 将第一层的残差结构传进去
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion
        # 实现实线部分
        for _ in range(1, block_num):  # 表示从一开始遍历，不写则默认是0层开始
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)  # 非关键字参数的方式传入nn.squential函数

    # 进行正向传播过程
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


# 没有加载ImageNet预训练参数
def resnet18(num_classes=10, include_top=True):
    # https://download.pytorch.org/models/resnet18-5c106cde.pth
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)


def resnet34(num_classes=10, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=10, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=10, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)
