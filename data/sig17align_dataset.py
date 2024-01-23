import numpy as np
from data.base_dataset import BaseDataset
from util.util import *
import glob
from multiprocessing.dummy import Pool
from tqdm import tqdm
import os


# Our data loader for self-supervised models
class SIG17AlignDataset(BaseDataset):
    def __init__(self, opt, split='train', dataset_name='Sig17Align'):
        super(SIG17AlignDataset, self).__init__(opt, split, dataset_name)
        
        self.root = opt.dataroot
        self.batch_size = opt.batch_size
        self.patch_size = opt.patch_size
        self.split = split

        if split == 'train':
            self.image_list = self._get_image_dir(self.root, 'Training', type=split)
            self._getitem = self._getitem_train
            self.len_data = 1200 * opt.batch_size  

        elif split == 'val':
            self.image_list = self._get_image_dir(self.root, 'Test', type=split)
            self._getitem = self._getitem_val
            self.len_data = len(self.image_list)

        elif split == 'test':
            if self.opt.stage0_inference:
                self.image_list = self._get_image_dir(self.root, 'Training_align', type=split) 
            else:
                if self.opt.network == 'FSHDR':
                    self.image_list = self._get_image_dir(self.root, 'Test_align', type=split) 
                else:
                    self.image_list = self._get_image_dir(self.root, 'Test', type=split) 
            self._getitem = self._getitem_test
            self.len_data = len(self.image_list)

        else:
            raise ValueError
               
        self.expoTimes = [0] * len(self.image_list)
        self.input0 = [0] * len(self.image_list)
        self.input1 = [0] * len(self.image_list)
        self.input2 = [0] * len(self.image_list)
        self.color_label = [0] * len(self.image_list)
        self.stru_label = [0] * len(self.image_list)

        read_images(self)

    def __getitem__(self, index):
        return self._getitem(index)

    def __len__(self):
        return self.len_data

    def _getitem_train(self, idx):
        idx = idx % len(self.image_list)

        color_label, stru_label, img0, img1, img2 = self._crop_patch(
            [self.color_label[idx], 
             self.stru_label[idx], 
             self.input0[idx], 
             self.input1[idx], 
             self.input2[idx]],
             p=self.patch_size)  # CxHxW
        
        # label, img0, img1, img2 = augment(label, img0, img1, img2)

        color_label = torch.from_numpy(np.float32(color_label))
        stru_label = torch.from_numpy(np.float32(stru_label))
        img0 = self._get_input(img0, self.expoTimes[idx][0])
        img1 = self._get_input(img1, self.expoTimes[idx][1])
        img2 = self._get_input(img2, self.expoTimes[idx][2])

        return {'input0': img0, 
                'input1': img1, 
                'input2': img2, 
                'label': color_label,
                'other_label': stru_label,
                'expo': self.expoTimes[idx][1],
                'fname': self.image_list[idx][3]}

    def _getitem_val(self, idx):
        label = torch.from_numpy(np.float32(self.color_label[idx]).transpose(2, 0, 1))
        img0 = self._get_input(self.input0[idx], self.expoTimes[idx][0], True)
        img1 = self._get_input(self.input1[idx], self.expoTimes[idx][1], True)
        img2 = self._get_input(self.input2[idx], self.expoTimes[idx][2], True)

        return {'input0': img0[:, 0:512, 0:512], 
                'input1': img1[:, 0:512, 0:512], 
                'input2': img2[:, 0:512, 0:512], 
                'label': label[:, 0:512, 0:512],
                'other_label': label[:, 0:512, 0:512],
                'expo': self.expoTimes[idx][1],
                'fname': self.image_list[idx][3]}

    def _getitem_test(self, idx):
        label = torch.from_numpy(np.float32(self.color_label[idx]).transpose(2, 0, 1))
        img0 = self._get_input(self.input0[idx], self.expoTimes[idx][0], True)
        img1 = self._get_input(self.input1[idx], self.expoTimes[idx][1], True)
        img2 = self._get_input(self.input2[idx], self.expoTimes[idx][2], True)

        return {'input0': img0, 
                'input1': img1, 
                'input2': img2, 
                'label': label,
                'other_label': label,
                'expo': self.expoTimes[idx][1],
                'fname': self.image_list[idx][3]}

    def _get_input(self, img, expotime, trans=False):
        if trans:
            img = img.transpose(2, 0, 1)  # CxHxW
        img = np.float32(img / 2 ** 16).clip(0, 1)
        pre_img = ldr_to_hdr(img, expotime, 2.2)
        pre_img = np.concatenate((pre_img, img), 0)
        return torch.from_numpy(pre_img)

    def _crop_patch(self, list_img, p):
        imgs = []  # HxWxC
        ih, iw = list_img[0].shape[0:2]
        ph = random.randrange(0, ih - p + 1)
        pw = random.randrange(0, iw - p + 1)
        for img in list_img:
            imgs.append(img[ph:ph+p, pw:pw+p, :].transpose(2, 0, 1))
        return imgs  # CxHxW

    def _get_image_dir(self, dataroot, name, type='train'):
        label_name = 'HDRImg.hdr'
        
        if '_align' in name:
            ori_name = name.split('_')[0]
        else:
            ori_name = name
            
        scenes_dir = os.path.join(dataroot, name)
        ori_dir = os.path.join(dataroot, ori_name)

        scenes_list = sorted(os.listdir(scenes_dir))
        image_list = []

        for scene in scenes_list:
            exposure_file_path = os.path.join(ori_dir, scene, 'exposure.txt')
            ldr_file_path = list_all_files_sorted(os.path.join(scenes_dir, scene), '.tif')
            if type == 'train':
                color_label_path = os.path.join(dataroot, 'Training_align', scene, label_name)
                if self.opt.stru_lable_path == '':
                    stru_label_path = color_label_path
                else:
                    stru_label_path = os.path.join(self.opt.stru_lable_path, scene + '.hdr')
            else:
                color_label_path = os.path.join(ori_dir, scene, label_name)
                stru_label_path = []
            image_list += [[exposure_file_path, ldr_file_path, color_label_path, scene, stru_label_path]]

        return image_list
    
def iter_obj(num, objs):
    for i in range(num):
        yield (i, objs)

def imreader(arg):
    i, obj = arg
    for _ in range(3):
        try:
            obj.expoTimes[i] = read_expo_times(obj.image_list[i][0])
            obj.input0[i] = cv2.imread(obj.image_list[i][1][0], -1)
            obj.input1[i] = cv2.imread(obj.image_list[i][1][1], -1)
            obj.input2[i] = cv2.imread(obj.image_list[i][1][2], -1)
            obj.color_label[i] = read_label(obj.image_list[i][2])

            if obj.split == 'train':
                obj.stru_label[i] = read_label(obj.image_list[i][4])
            
            failed = False
            break
        except:
            failed = True
    if failed: print('%s fails!' % obj.image_list[i])

def read_images(obj):
    # may use `from multiprocessing import Pool` instead, but less efficient and
    # NOTE: `multiprocessing.Pool` will duplicate given object for each process.
    print('Starting to load images via multiple imreaders')
    pool = Pool() # use all threads by default
    for _ in tqdm(pool.imap(imreader, iter_obj(len(obj.image_list), obj)), total=len(obj.image_list)):
        pass
    pool.close()
    pool.join()

