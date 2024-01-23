import os
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from tqdm import tqdm
# from util.util import calc_psnr as calc_psnr
import time
import numpy as np
from collections import OrderedDict as odict
from copy import deepcopy
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from util.util import range_compressor, calculate_ssim, save_hdr
import cv2


if __name__ == '__main__':
    opt = TestOptions().parse()

    if not isinstance(opt.load_iter, list):
        load_iters = [opt.load_iter]
    else:
        load_iters = deepcopy(opt.load_iter)

    if not isinstance(opt.dataset_name, list):
        dataset_names = [opt.dataset_name]
    else:
        dataset_names = deepcopy(opt.dataset_name)
    datasets = odict()
    for dataset_name in dataset_names:
        dataset = create_dataset(dataset_name, 'test', opt)
        datasets[dataset_name] = tqdm(dataset)

    for load_iter in load_iters:
        opt.load_iter = load_iter
        model = create_model(opt)
        model.setup(opt)
        model.eval()
        log_dir = '%s/%s/log_epoch_%d.txt' % (
                opt.checkpoints_dir, opt.name, load_iter)
        os.makedirs(os.path.split(log_dir)[0], exist_ok=True)
        f = open(log_dir, 'a')

        for dataset_name in dataset_names:
            opt.dataset_name = dataset_name
            tqdm_val = datasets[dataset_name]
            dataset_test = tqdm_val.iterable
            dataset_size_test = len(dataset_test)

            print('='*80)
            print(dataset_name + ' dataset')
            tqdm_val.reset()

            psnr_l = [0.0] * dataset_size_test
            psnr_mu = [0.0] * dataset_size_test
            ssim_l = [0.0] * dataset_size_test
            ssim_mu = [0.0] * dataset_size_test

            time_val = 0
            for i, data in enumerate(tqdm_val):
                torch.cuda.empty_cache()
                model.set_input(data)
                torch.cuda.synchronize()
                time_val_start = time.time()
                model.test()
                torch.cuda.synchronize()
                time_val += time.time() - time_val_start
                res = model.get_current_visuals()

                output = model.data_out[0].detach().cpu().numpy().astype(np.float32)# [::-1,:,:]
                gt = model.data_color_label[0].detach().cpu().numpy().astype(np.float32)
               
                if opt.calc_metrics:
                    # psnr-l and psnr-\mu
                    psnr_l[i] = compare_psnr(gt, output, data_range=1.0)
                    label_mu = range_compressor(gt)
                    output_mu = range_compressor(output)
                    psnr_mu[i] = compare_psnr(label_mu, output_mu, data_range=1.0)
                    # ssim-l
                    output_l = np.clip(output * 255.0, 0., 255.).transpose(1, 2, 0)
                    label_l = np.clip(gt * 255.0, 0., 255.).transpose(1, 2, 0)
                    ssim_l[i] = calculate_ssim(output_l, label_l)
                    # ssim-\mu
                    output_mu = np.clip(output_mu * 255.0, 0., 255.).transpose(1, 2, 0)
                    label_mu = np.clip(label_mu * 255.0, 0., 255.).transpose(1, 2, 0)
                    ssim_mu[i] = calculate_ssim(output_mu, label_mu)

                if opt.save_imgs:
                    folder_dir = '%s/%s/output_%d' % (opt.checkpoints_dir, opt.name, load_iter)  
                    os.makedirs(folder_dir, exist_ok=True)
                    save_dir = '%s/%s.hdr' % (folder_dir, data['fname'][0])
                    save_hdr(save_dir, output.transpose(1, 2, 0)[..., ::-1])

                    folder_dir = '%s/%s/output_vis_%d' % (opt.checkpoints_dir, opt.name, load_iter)  
                    os.makedirs(folder_dir, exist_ok=True)
                    save_dir = '%s/%s.png' % (folder_dir, data['fname'][0])
                    out_vis = res['data_out'][0].cpu().numpy().transpose(1, 2, 0)
                    cv2.imwrite(save_dir, np.array(out_vis).astype(np.uint8))

            avg_psnr_l = '%.2f'%np.mean(psnr_l)
            avg_psnr_mu = '%.2f'%np.mean(psnr_mu)
            avg_ssim_l = '%.4f'%np.mean(ssim_l)
            avg_ssim_mu = '%.4f'%np.mean(ssim_mu)

            f.write('AVG Time: %.3f ms \n avg_psnr_l: %s, avg_psnr_mu: %s \n avg_ssim_l: %s, avg_ssim_mu: %s \n' 
            % (time_val/dataset_size_test*1000, avg_psnr_l, avg_psnr_mu, avg_ssim_l, avg_ssim_mu))
            print('AVG Time: %.3f ms \n avg_psnr_l: %s, avg_psnr_mu: %s \n avg_ssim_l: %s, avg_ssim_mu: %s \n' 
            % (time_val/dataset_size_test*1000, avg_psnr_l, avg_psnr_mu, avg_ssim_l, avg_ssim_mu))
            f.flush()
            f.write('\n')
        f.close()
    for dataset in datasets:
        datasets[dataset].close()
