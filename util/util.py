"""This module contains simple helper functions """
from __future__ import print_function
import numpy as np
import os, glob
import cv2
import math
import imageio
from math import log10
import random
import torch
import torch.nn as nn
import torch.nn.init as init
from skimage.metrics import peak_signal_noise_ratio
imageio.plugins.freeimage.download()


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)

def print_numpy(x, val=True, shp=True):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, mid = %3.3f, std=%3.3f'
              % (np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))

def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def prompt(s, width=66):
    print('='*(width+4))
    ss = s.split('\n')
    if len(ss) == 1 and len(s) <= width:
        print('= ' + s.center(width) + ' =')
    else:
        for s in ss:
            for i in split_str(s, width):
                print('= ' + i.ljust(width) + ' =')
    print('='*(width+4))

def split_str(s, width):
    ss = []
    while len(s) > width:
        idx = s.rfind(' ', 0, width+1)
        if idx > width >> 1:
            ss.append(s[:idx])
            s = s[idx+1:]
        else:
            ss.append(s[:width])
            s = s[width:]
    if s.strip() != '':
        ss.append(s)
    return ss

def augment_func(img, hflip, vflip, rot90):  # CxHxW
    if hflip:   img = img[:, :, ::-1]
    if vflip:   img = img[:, ::-1, :]
    if rot90:   img = img.transpose(0, 2, 1)
    return np.ascontiguousarray(img)

def augment(*imgs):  # CxHxW
    hflip = random.random() < 0.5
    vflip = random.random() < 0.5
    rot90 = random.random() < 0.5
    return (augment_func(img, hflip, vflip, rot90) for img in imgs)

def gamma_correction(img, r=1/2.2):
    img = np.maximum(img, 0)
    img = np.power(img, r)
    return img

def weight_0(img, mask, ref):
    return (1 - mask) * (2 * ref - 1) * img, (1 - mask) * (2 * ref - 1)

def weight_1(img, mask, ref):
    return mask * (2 * ref) * img + (1 - mask) * (- 2 * ref + 2) * img, mask * (2 * ref) + (1 - mask) * (- 2 * ref + 2)

def weight_2(img, mask, ref):
    return mask * (1 - 2 * ref) * img, mask * (1 - 2 * ref)

def syn_hdr(in_0, in_1, in_2, rgb_1):
    mask = (rgb_1 < 0.5).float()
    img_out = weight_0(in_0, mask, rgb_1)[0] + weight_1(in_1, mask, rgb_1)[0] + weight_2(in_2, mask, rgb_1)[0]
    return img_out

def list_all_files_sorted(folder_name, extension=""):
    return sorted(glob.glob(os.path.join(folder_name, "*" + extension)))

def read_expo_times(file_name):
    return np.power(2, np.loadtxt(file_name))

def read_label(file_path):
    label = imageio.imread(file_path, 'hdr')
    label = label[:, :, [2, 1, 0]]  ##cv2
    return label

def ldr_to_hdr(imgs, expo, gamma):
    return (imgs ** gamma) / (expo + 1e-8)

def inv_gamma(imgs, gamma):
    return imgs ** gamma

def range_compressor(x):
    return (np.log(1 + 5000 * x)) / np.log(1 + 5000)

def range_compressor_cuda(hdr_img, mu=5000):
    return (torch.log(1 + mu * hdr_img)) / math.log(1 + mu)

def range_compressor_tensor(x, device):
    a = torch.tensor(1.0, device=device, requires_grad=False)
    mu = torch.tensor(5000.0, device=device, requires_grad=False)
    return (torch.log(a + mu * x)) / torch.log(a + mu)

def calc_psnr(x, target):
    sqrdErr = np.mean((x - target) ** 2)
    return 10 * log10(1/sqrdErr)

def batch_psnr(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    psnr = 0
    for i in range(Img.shape[0]):
        psnr += peak_signal_noise_ratio(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (psnr/Img.shape[0])

def batch_psnr_mu(img, imclean, data_range):
    img = range_compressor_cuda(img)
    imclean = range_compressor_cuda(imclean)
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    psnr = 0
    for i in range(Img.shape[0]):
        psnr += peak_signal_noise_ratio(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (psnr/Img.shape[0])

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr * (0.5 ** (epoch // args.lr_decay_interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def init_parameters(net):
    """Init layer parameters"""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

def set_random_seed(seed=0):
    """Set random seed for reproduce"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2):
    """
    calculate SSIM

    :param img1: [0, 255]
    :param img2: [0, 255]
    :return:
    """
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def radiance_writer(out_path, image):
    with open(out_path, "wb") as f:
        f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(b"-Y %d +X %d\n" %(image.shape[0], image.shape[1]))
        brightest = np.maximum(np.maximum(image[...,0], image[...,1]), image[...,2])
        mantissa = np.zeros_like(brightest)
        exponent = np.zeros_like(brightest)
        np.frexp(brightest, mantissa, exponent)
        scaled_mantissa = mantissa * 255.0 / brightest
        rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        rgbe[...,0:3] = np.around(image[...,0:3] * scaled_mantissa[...,None])
        rgbe[...,3] = np.around(exponent + 128)

        rgbe.flatten().tofile(f)

def save_hdr(path, image):
    return radiance_writer(path, image)




