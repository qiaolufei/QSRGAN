import torch
from torch import nn
from torchvision.models.vgg import vgg16


class GeneratorLoss(nn.Module):  # 生成损失函数
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        # ---VGG---
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        # ---VGG--- 最后为loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss 对抗损失函数
        adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss 感知VGG损失函数  LOSS神经网络就是VGG网络
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss MSE损失函数
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss


class TVLoss(nn.Module):  # TV损失函数
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()  # 继承
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]  # 取x的第一个数
        h_x = x.size()[2]  # 第三个数
        w_x = x.size()[3]  # 最后一个数
        count_h = self.tensor_size(x[:, :, 1:, :])  # H-X的最大值就是他
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()  # 相减然后再平方，再求和
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
