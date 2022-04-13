import math
import torch
from torch import nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)
        block2_1 = RSBU_CW(in_channels=4, out_channels=4, kernel_size=3, down_sample=True)(block1)
        block2 = AttentionLayer(512)(block2_1)
        block3_1 = RSBU_CW(in_channels=4, out_channels=4, kernel_size=3, down_sample=False)(block2)
        block3 = AttentionLayer(512)(block3_1)
        block4_1 = RSBU_CW(in_channels=4, out_channels=4, kernel_size=3, down_sample=True)(block3)
        block4 = AttentionLayer(512)(block4_1)
        block5_1 = RSBU_CW(in_channels=4, out_channels=4, kernel_size=3, down_sample=False)(block4)
        block5 = AttentionLayer(512)(block5_1)
        block6_1 = RSBU_CW(in_channels=4, out_channels=4, kernel_size=3, down_sample=True)(block5)
        block6 = AttentionLayer(512)(block6_1)

        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (torch.tanh(block8) + 1) / 2


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            AttentionLayer(512),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            AttentionLayer(512),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            AttentionLayer(512),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            AttentionLayer(512),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            AttentionLayer(512),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            AttentionLayer(512),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            AttentionLayer(512),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual

class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

class RSBU_CW(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, down_sample=False):
        super().__init__()
        self.down_sample = down_sample
        self.in_channels = in_channels
        self.out_channels = out_channels
        stride = 1
        if down_sample:
            stride = 2
        self.BRC = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                      padding=1)
        )
        self.global_average_pool = nn.AdaptiveAvgPool1d(1)
        self.FC = nn.Sequential(
            nn.Linear(in_features=out_channels, out_features=out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=out_channels, out_features=out_channels),
            nn.Sigmoid()
        )
        self.flatten = nn.Flatten()
        self.average_pool = nn.AvgPool1d(kernel_size=1, stride=2)

    def forward(self, input):
        x = self.BRC(input)
        x_abs = torch.abs(x)
        gap = self.global_average_pool(x_abs)
        gap = self.flatten(gap)
        alpha = self.FC(gap)
        threshold = torch.mul(gap, alpha)
        threshold = torch.unsqueeze(threshold, 2)
        # 软阈值化
        sub = x_abs - threshold
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x), n_sub)  
        if self.down_sample:  # 如果是下采样，则对输入进行平均池化下采样
            input = self.average_pool(input)
        if self.in_channels != self.out_channels:  # 如果输入的通道和输出的通道不一致，则进行padding,直接通过复制拼接矩阵进行padding,原代码是通过填充0
            zero_padding=torch.zeros(input.shape).cuda()
            input = torch.cat((input, zero_padding), dim=1)

        result = x + input
        return result

class AttentionLayer(nn.Module):
    """
    A2-Nets: Double Attention Networks. NIPS 2018
    """
    def __init__(self,in_c):
        """
        :param
        in_c: 进行注意力refine的特征图的通道数目；
        原文中的降维和升维没有使用
        """
        super(AttentionLayer,self).__init__()
        self.in_c = in_c
        """
        以下对同一输入特征图进行卷积，产生三个尺度相同的特征图，即为文中提到A, B, V
        """
        self.convA = nn.Conv2d(in_c,in_c,kernel_size=1)
        self.convB = nn.Conv2d(in_c,in_c,kernel_size=1)
        self.convV = nn.Conv2d(in_c,in_c,kernel_size=1)
    def forward(self,input):

        feature_maps = self.convA(input)
        atten_map = self.convB(input)
        b, _, h, w = feature_maps.shape

        feature_maps = feature_maps.view(b, 1, self.in_c, h*w) # 对 A 进行reshape
        atten_map = atten_map.view(b, self.in_c, 1, h*w)       # 对 B 进行reshape 生成 attention_aps
        global_descriptors = torch.mean((feature_maps * F.softmax(atten_map, dim=-1)),dim=-1) # 特征图与attention_maps 相乘生成全局特征描述子

        v = self.convV(input)
        atten_vectors = F.softmax(v.view(b, self.in_c, h*w), dim=-1) # 生成 attention_vectors
        out = torch.bmm(atten_vectors.permute(0,2,1), global_descriptors).permute(0,2,1) # 注意力向量左乘全局特征描述子

        return out.view(b, _, h, w)