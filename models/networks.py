import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
from timm.scheduler.cosine_lr import CosineLRScheduler
import numpy as np
from torchvision import models
import numpy as np
from util.util import range_compressor_cuda


operation_seed_counter = 0

def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device="cpu")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator


class AugmentNoise(object):
    def __init__(self, style):
        if style.startswith('gauss'):
            self.params = [
                float(p) / 255.0 for p in style.replace('gauss', '').split('_')
            ]
            if len(self.params) == 1:
                self.style = "gauss_fix"
            elif len(self.params) == 2:
                self.style = "gauss_range"
        elif style.startswith('poisson'):
            self.params = [
                float(p) for p in style.replace('poisson', '').split('_')
            ]
            if len(self.params) == 1:
                self.style = "poisson_fix"
            elif len(self.params) == 2:
                self.style = "poisson_range"
        print('Noise Style: %s, Noise Params: %s' % (self.style, str(self.params)))

    def add_train_noise(self, x):
        shape = x.shape # [0,1]
        if self.style == "gauss_fix":
            std = self.params[0]
            noise = torch.normal(mean=0, std=std, size=(shape[0], shape[1],shape[2],shape[3]), 
                                 generator=get_generator(), dtype=torch.float)
            return x + noise.to(x.device)
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = np.random.rand(shape[0]) * (max_std - min_std) + min_std
            noise = torch.cat([torch.normal(mean=0, 
                                            std=std[i], 
                                            size=(1, shape[1],shape[2],shape[3]), 
                                            generator=get_generator(),
                                            dtype=torch.float) 
                               for i in range(shape[0])], 0)
            return x + noise.to(x.device)


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            return 1 - max(0, epoch-opt.niter) / max(1, float(opt.niter_decay))
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=opt.lr_decay_iters,
                                        gamma=0.5)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.2,
                                                   threshold=0.01,
                                                   patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=opt.niter,
                                                   eta_min=1e-5)
    elif opt.lr_policy == 'cosine_warmup':
        scheduler = CosineLRScheduler(optimizer, 
                                      t_initial=opt.niter, 
                                      lr_min=1e-5,
                                      warmup_t=4,
                                      warmup_lr_init=1e-5)
    else:
        return NotImplementedError('lr [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 \
                or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == 'uniform':
                init.uniform_(m.weight.data, b=init_gain)
            else:
                raise NotImplementedError('[%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='default', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    if init_type != 'default' and init_type is not None:
        init_weights(net, init_type, init_gain=init_gain)
    return net


class L1MuLoss(nn.Module):
    def __init__(self, mu=5000):
        super(L1MuLoss, self).__init__()
        self.mu = mu
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, label, mask=1.):
        mu_pred = range_compressor_cuda(pred, self.mu) * mask
        mu_label = range_compressor_cuda(label, self.mu) * mask
        return self.l1_loss(mu_pred, mu_label)


def normalize_batch(batch, mask=1.):
    batch = range_compressor_cuda(batch) * mask
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std


class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(weights='DEFAULT').features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = nn.L1Loss()

    def forward(self, img1, img2, mask=1):
        x = normalize_batch(img1, mask)
        y = normalize_batch(img2, mask)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0

        content_loss += self.criterion(x_vgg['relu3_2'], y_vgg['relu3_2']) * 1
        content_loss += self.criterion(x_vgg['relu4_2'], y_vgg['relu4_2']) * 1
        content_loss += self.criterion(x_vgg['relu5_2'], y_vgg['relu5_2']) * 2

        return content_loss / 4.
