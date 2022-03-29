import argparse
import os
from math import log10

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm  # 进度条

import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from loss import GeneratorLoss
from model import Generator, Discriminator

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=88, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    opt = parser.parse_args()
    
    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs

    # 加载数据集
    train_set = TrainDatasetFromFolder('data/VOC2012/train', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    val_set = ValDatasetFromFolder('data/VOC2012/val', upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)
    # 加载网络模型
    netG = Generator(UPSCALE_FACTOR)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    netD = Discriminator()
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

    # 加载loss函数
    generator_criterion = GeneratorLoss()
    # 判断GPU加速
    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()

    # 定义Adam优化器
    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())
    # 定义结果保存的字典，值为列表
    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
    
    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)  # 生成进度条>>>>>>>>
        # 定义字典统计相关超参数
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
    
        netG.train()
        netD.train()
        for data, target in train_bar:
            g_update_first = True
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size
    
            ############################
            # data/z：由target下采样的低分辨率图像 -->  G --> fake_img --> D --> fake_out(label)
            # target/real_img：高分辨率图像（原图） --> D --> real_out(label)
            ###########################
            real_img = Variable(target)  # torch数据类型的标签图像real_img
            if torch.cuda.is_available():
                real_img = real_img.cuda()
            z = Variable(data)  # torch数据类型的输入图像z
            if torch.cuda.is_available():
                z = z.cuda()
            fake_img = netG(z)  # 生成的超分辨图像
    
            netD.zero_grad()  # 判别网络的梯度归零
            real_out = netD(real_img).mean()  # 判别网络对于标签图像的输出的均值real_out
            fake_out = netD(fake_img).mean()  # 判别网络对于fake_img的输出的均值fake_out
            d_loss = 1 - real_out + fake_out  # d_loss = - [D(z)-1-D(G(z))],所以最小化d_loss，则后一项的最大化
            d_loss.backward(retain_graph=True)  # 反向传播
            optimizerD.step()  # 梯度优化
    
            ############################
            # (2) 更新生成网络: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            netG.zero_grad()   # 生成网络梯度归零
            ## The two lines below are added to prevent runetime error in Google Colab ##
            fake_img = netG(z)  # 生成网络的的输出图像fake_img
            fake_out = netD(fake_img).mean()  # 判别网络对于fake_img的输出的均值fake_out
            ##
            g_loss = generator_criterion(fake_out, fake_img, real_img)  # 生成网络loss计算
            g_loss.backward()
            
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()
            
            
            optimizerG.step()

            # loss for current batch before optimization 
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size  # d_loss real/fake通过判别器的差距
            running_results['d_score'] += real_out.item() * batch_size  # real通过判别器的值
            running_results['g_score'] += fake_out.item() * batch_size  # fake通过判别器的值
            # 描述进度和损失函数，得分函数的平均值
            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))

        # ################## 进入eval模式 （测试模式参数固定，只有前向传播）用测试集###############
        netG.eval()
        out_path = 'training_results/SRF_' + str(UPSCALE_FACTOR) + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
        with torch.no_grad():
            val_bar = tqdm(val_loader)  # 用验证集进行计算MSE，输出图像
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size
                lr = val_lr
                hr = val_hr
                if torch.cuda.is_available():
                    lr = lr.cuda()
                    hr = hr.cuda()
                sr = netG(lr)  # 直接输出结果，没有参数优化的过程
        
                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                valing_results['ssims'] += batch_ssim * batch_size
                valing_results['psnr'] = 10 * log10((hr.max()**2) / (valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        valing_results['psnr'], valing_results['ssim']))
                # 通过extend把三张图连在一起
                val_images.extend(
                    [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                     display_transform()(sr.data.cpu().squeeze(0))])
            val_images = torch.stack(val_images)  # 按行拼接，按列拼接
            val_images = torch.chunk(val_images, val_images.size(0) // 15)
            val_save_bar = tqdm(val_images, desc='[saving training results]')  #传入str类型，作为进度条标题（类似于说明）
            index = 1
            for image in val_save_bar:
                # 每一行显示三个图像
                image = utils.make_grid(image, nrow=3, padding=5)
                utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                index += 1
    
        # save model parameters
        torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        torch.save(netD.state_dict(), 'epochs/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        # save loss\scores\psnr\ssim
        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])
    
        if epoch % 10 == 0 and epoch != 0:
            out_path = 'statistics/'
            data_frame = pd.DataFrame(
                data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                      'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(1, epoch + 1))
            data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_train_results.csv', index_label='Epoch')
