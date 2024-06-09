"""
modules.py - This file stores the rathering boring network blocks.
"""

from numpy.lib.arraysetops import isin
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from models import mod_resnet


class ResBlock(nn.Module):
    """这段代码定义了一个名为ResBlock的类，它是一个残差块（Residual Block），
    常用于深度神经网络中，特别是在卷积神经网络（CNN）的构建中。
    这种残差块最初由He等人在2015年提出，用于解决深度神经网络训练过程中的梯度消失或梯度爆炸问题，
    允许模型能够成功训练更深的网络结构。以下是ResBlock类的具体功能和组成：
    初始化（__init__）:
    输入维度（indim）：输入特征图的通道数。
    输出维度（outdim）：输出特征图的通道数。如果没有指定，输出维度将与输入维度相同。
    卷积层：该类定义了两个卷积层，都是使用3x3的卷积核和padding=1（填充为1）。这种设置使得卷积操作不改变特征图的空间尺寸（高度和宽度）。
    下采样层（self.downsample）：如果输入和输出维度不同，使用一个卷积层来调整维度，确保可以进行残差连接。这个卷积层也用3x3卷积核，padding=1。
    前向传播（forward）:
    输入（x）：输入特征图。
    经过卷积层和ReLU激活函数处理：首先通过第一个卷积层和ReLU激活函数处理输入，然后结果再次通过第二个卷积层和ReLU激活函数。
    残差连接：如果定义了下采样层，先对输入特征图x进行下采样以匹配维度；最后，将这个下采样或原始的输入与第二个卷积层的输出相加，
    形成残差连接。
    输出：返回残差连接的结果。
    这种残差连接的关键思想是允许网络学习输入和输出之间的差异（即残差），而不是直接学习未处理输入的映射。这有助于信息在网络中的传播，
    从而可以训练更深的网络。

    在实际应用中，这样的残差块可以大量堆叠，形成如ResNet这样的深度网络，广泛应用于图像识别、对象检测和许多其他视觉相关任务中。"""
    def __init__(self, indim, outdim=None):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
 
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)
 
    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
        
        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class MaskRGBEncoderSO(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = mod_resnet.resnet50(pretrained=True, extra_chan=1)
        #参数extra_chan=1表示在标准的RGB输入（3个通道）之外，还将有一个额外的通道。
        #这通常用于将二值掩码或其他类型的单通道数据作为网络的输入之一
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1 # 1/4, 256
        self.layer2 = resnet.layer2 # 1/8, 512
        self.layer3 = resnet.layer3 # 1/16, 1024

    def forward(self, f, m):
        #RGB图像和掩码图像融合
        #通过修改过的ResNet50模型进行特征提取
    
        f = torch.cat([f, m], 1)

        x = self.conv1(f)
        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        x = self.layer1(x)   # 1/4, 256
        x = self.layer2(x) # 1/8, 512
        x = self.layer3(x) # 1/16, 1024
        #通过将掩码和图像数据结合，网络可以更好地理解图像的哪些部分是重点区域，从而改进模型的预测性能，
        return x


class MaskRGBEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = mod_resnet.resnet50(pretrained=True, extra_chan=2)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1 # 1/4, 256
        self.layer2 = resnet.layer2 # 1/8, 512
        self.layer3 = resnet.layer3 # 1/16, 1024

    def forward(self, f, m, o):#o: other_mask

        f = torch.cat([f, m, o], 1)

        x = self.conv1(f)
        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        x = self.layer1(x)   # 1/4, 256
        x = self.layer2(x) # 1/8, 512
        x = self.layer3(x) # 1/16, 1024

        return x
 

class RGBEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # resnet = models.resnet50(pretrained=True)
        resnet = mod_resnet.resnet50(pretrained=True)     #use mod_resnet as backbone
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.layer2 = resnet.layer2 # 1/8, 512
        self.layer3 = resnet.layer3 # 1/16, 1024

    def forward(self, f):
        x = self.conv1(f) 
        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        f4 = self.res2(x)   # 1/4, 256 torch.Size([1, 256, 72, 128])
        f8 = self.layer2(f4) # 1/8, 512 torch.Size([1, 512, 72, 128])
        f16 = self.layer3(f8) # 1/16, 1024 torch.Size([1, 1024, 72, 128])

        return f16, f8, f4


class UpsampleBlock(nn.Module):
    """
    上采样模块，用于将输入特征图进行上采样并融合，然后输出特定尺度和通道数的特征图。

    参数:
    - skip_c: 跳过连接的通道数。
    - up_c: 上采样层的通道数。
    - out_c: 输出特征图的通道数。
    - scale_factor: 上采样的缩放因子，默认为2。
    """
    def __init__(self, skip_c, up_c, out_c, scale_factor=2):
        super().__init__()
        self.skip_conv1 = nn.Conv2d(skip_c, up_c, kernel_size=3, padding=1)
        self.skip_conv2 = ResBlock(up_c, up_c)
        self.out_conv = ResBlock(up_c, out_c)
        self.scale_factor = scale_factor

    def forward(self, skip_f, up_f):
        """
        前向传播函数，融合跳过连接的特征图和上采样的特征图，然后输出。

        参数:valdimrgb
        - skip_f: 跳过连接的特征图。
        - up_f: 上采样的特征图。

        返回:
        - x: 经过处理后的输出特征图。
        """
        x = self.skip_conv2(self.skip_conv1(skip_f)) # 对跳过连接的特征图进行卷积处理
        x = x + F.interpolate(up_f, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        x = self.out_conv(x)
        return x


class KeyValue(nn.Module):
    """
    KeyValue 类用于创建一个键值对模块，该模块包含两个卷积层，用于将输入数据投影到键空间和值空间。
    
    参数:
    - indim (int): 输入维度，即输入数据的通道数。
    - keydim (int): 键的维度，即键空间的通道数。
    - valdim (int): 值的维度，即值空间的通道数。
    """
    def __init__(self, indim, keydim, valdim):
        """
        初始化KeyValue模块，创建键投影层和值投影层。
        """
        super().__init__()
        self.key_proj = nn.Conv2d(indim, keydim, kernel_size=3, padding=1)
        self.val_proj = nn.Conv2d(indim, valdim, kernel_size=3, padding=1)
 
    def forward(self, x):  
        """
        前向传播函数，将输入数据分别投影到键空间和值空间。
        
        参数:
        - x (Tensor): 输入的张量。
        
        返回:
        - Tuple[Tensor, Tensor]: 键空间的张量和值空间的张量。
        """
        return self.key_proj(x), self.val_proj(x)

class Score(nn.Module):
    ''' 这段代码定义了一个名为Score的神经网络模块，这是一个基于PyTorch框架的评分或者回归模型。
        这个模型使用卷积层来处理输入特征图，
        然后通过全连接层输出一个单一的数值，
        通常用于评分或者预测任务

        初始化方法（__init__）:
        输入通道（input_chan）: 模型第一个卷积层的输入通道数。
        卷积层: 定义了四个卷积层，每个层的输出通道数均为256，
        使用3x3的卷积核，步长（stride）为1或2，以及填充（padding）为1。
        这些卷积层主要用于提取和转换输入特征图的特征。
        全连接层: 两个全连接层，第一个全连接层将卷积层输出的特征图转换成1024个特征的一维向量，
        第二个全连接层将这1024个特征转换成一个单一的输出值，适用于回归或评分任务。

        初始化权重和偏置：使用He初始化（kaiming_normal_和kaiming_uniform_）
        来初始化卷积层和全连接层的权重，偏置初始化为0。He初始化有利于ReLU激活函数后的模型收敛。
        前向传播方法（forward）:

        输入x: 传入的特征图。
        通过四个卷积层和ReLU激活函数，逐步提取和转换特征。
        第四个卷积层使用了步长为2，这有助于减少特征图的空间维度。
        将卷积层输出的多维特征图展平（flatten）成一维向量。
        通过两个全连接层继续处理特征，第二个全连接层输出最终的评分或预测结果。
        此外，代码中注释掉的部分提到了使用全局平均池化层（nn.AdaptiveAvgPool2d）
        和一个全连接层来替代多个全连接层的方案。这种方案可以减少模型的参数数量，同时保持全局特征的使用。

        整体而言，Score类是一个用于特征图评分的卷积神经网络模型，
        可以用于多种应用场景，如图像质量评估、图像中的物体评分等，
        依赖于提取深层次的特征并将其转换为实际的评分输出。'''
    def __init__(self, input_chan):
        super(Score, self).__init__()
        input_channels = input_chan
        self.conv1 = nn.Conv2d(input_channels, 256, 3, 1, 1)
        self.conv2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv4 = nn.Conv2d(256, 256, 3, 2, 1)
        
        
        
   
        self.fc1   = nn.Linear(256*10*18, 1024)       #fc layers
        self.fc2   = nn.Linear(1024, 1)

        # self.gav = nn.AdaptiveAvgPool2d(1)        #global average pooling layers
        # self.fc = nn.Linear(256, 1)

        for i in [self.conv1, self.conv2, self.conv3, self.conv4]:
            nn.init.kaiming_normal_(i.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(i.bias, 0)

        for i in [self.fc1, self.fc2]:
            nn.init.kaiming_uniform_(i.weight, a=1)
            nn.init.constant_(i.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        
        x_1 = x.size(1)
        
        
        
        
        #self.fc111  = nn.Linear(x_1, 1024).to(x.device)  
        in_features = x.size(1)
        self.fc11 = nn.Linear(in_features, 1024).to(x.device)
 
        x = F.relu(self.fc11(x))
        x = F.relu(self.fc2(x))
      
        # x = self.gav(x)               #the method of gav
        # x = x.view(x.size(0), -1)
        # x = F.leaky_relu(self.fc(x))
        return x

class _ASPPModule(nn.Module):
    """空洞卷积（Atrous Convolution 或 Dilated Convolution）是一种特殊类型的卷积操作，
    用于扩大卷积核的感受野而不增加参数数量或计算量。它通过在标准卷积核的元素之间插入空间来实现，这种操作使得卷积核能够覆盖更大的输入区域。

    空洞卷积的工作原理
    传统的卷积操作是通过在输入特征图上滑动一个卷积核并进行点乘操作来实现的，
    每次移动一个像素步长。而在空洞卷积中，卷积核的每个元素之间会有一定的空间（称为膨胀率dilation rate），
    这样使得卷积核覆盖更广的输入区域，但实际上使用的参数并没有增加。

    例如，一个3x3的卷积核，在膨胀率为2的情况下，其覆盖的实际区域不再是3x3的区域，
    而是5x5的区域，其中卷积核的元素被扩展了，空出的位置为0（不参与计算）。
    这样可以使得输出特征图在保持相同分辨率的同时，获得更大的感受野。"""
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, 
        padding=padding, dilation=dilation, bias=False)

        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)      
        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    """
    这段代码定义了两个类 _ASPPModule 和 ASPP，
    它们是为实现深度卷积神经网络中的一个高级特性——空洞空间金字塔池化（Atrous Spatial Pyramid Pooling, ASPP）设计的。
    这种结构在图像分割任务中特别常见，如在DeepLab模型中广泛使用，以有效地捕获多尺度的上下文信息。以下是对这两个类的功能和结构的详细解释：

    _ASPPModule 类
    这个类定义了一个空洞卷积（Atrous Convolution）模块，它可以增加感受野，而不增加参数数量或计算复杂度。

    构造函数：接收输入和输出的通道数（inplanes 和 planes），卷积核大小（kernel_size），填充（padding），膨胀率（dilation），
    和批量归一化层的类型（BatchNorm）。它创建了一个空洞卷积层，后面接批量归一化和ReLU激活函数。

    前向传播：输入通过空洞卷积层、批量归一化和ReLU激活函数，顺序处理后输出。
    权重初始化：使用He初始化方法（kaiming_normal_）初始化卷积层，批量归一化层的权重设为1，偏置设为0。


    ASPP 类
    这个类实现了空洞空间金字塔池化（ASPP）结构，用于聚合不同尺度的上下文信息。
    构造函数：接收输入通道数（inplanes），输出步长（output_stride，用于确定膨胀率），和批量归一化层的类型（BatchNorm）。基于输出步长，它会设置不同的膨胀率数组。
    它创建了四个不同膨胀率的_ASPPModule实例以及一个全局平均池化层，后者通过1x1的卷积将全局信息压缩到256个通道。
    前向传播：分别处理输入通过上述五个模块，然后将这些特征图在通道维度上拼接。拼接后的特征图通过一个1x1卷积，再经过批量归一化和ReLU激活函数，最后应用dropout防止过拟合。
    权重初始化：与_ASPPModule类似，使用He初始化方法初始化卷积层，批量归一化层的权重设为1，偏置设为0。
    总结
    通过使用不同膨胀率的空洞卷积，ASPP能够在不同尺度上捕获图像特征，这对于需要精确像素级预测的图像分割任务非常有帮助。
    全局平均池化层则帮助模型捕获全局上下文信息，这些特性使ASPP成为深度学习模型中处理图像分割问题的强大工具。
    """
    def __init__(self, inplanes, output_stride=16, BatchNorm=nn.InstanceNorm2d):
        super(ASPP, self).__init__()
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError
        
        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                     nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(1280, 1024, 1, bias=False)
        self.bn1 = BatchNorm(1024)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)#torch.Size([1, 1024, 18, 32])


        return self.dropout(x)
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
