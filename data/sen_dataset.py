import numpy as np
from data.base_dataset import BaseDataset
from util.util import *
import glob
from multiprocessing.dummy import Pool
from tqdm import tqdm
import os



# Data loader only for testing
class SENDataset(BaseDataset):
    def __init__(self, opt, split='train', dataset_name='Sen'):
        super(SENDataset, self).__init__(opt, split, dataset_name)

        self.root = opt.dataroot
        self.batch_size = opt.batch_size
        self.patch_size = opt.patch_size
        self.split = split
        self.scenes_list = sorted(os.listdir(os.path.join(self.root, 'Scenes')))

        if self.opt.network == 'FSHDR':
            self._getitem = self._getitem_test_align
        else:
            self._getitem = self._getitem_test

        self.len_data = len(self.scenes_list)

    def __getitem__(self, index):
        return self._getitem(index)

    def __len__(self):
        return self.len_data

    def _getitem_test(self, idx):
        ldr_file_path = list_all_files_sorted(os.path.join(self.root, 'Scenes', self.scenes_list[idx]), '.tif')
        expoTimes = read_expo_times(os.path.join(self.root, 'Scenes', self.scenes_list[idx], 'exposures.txt'))
        img_len = (len(ldr_file_path) + 2) // 2 - 1

        input0 = cv2.imread(ldr_file_path[img_len-1], -1)
        input1 = cv2.imread(ldr_file_path[img_len], -1)
        input2 = cv2.imread(ldr_file_path[img_len+1], -1)

        label = torch.from_numpy(np.float32(input1).transpose(2, 0, 1))

        img0 = self._get_input(input0, expoTimes[img_len-1]/expoTimes[img_len-1], True)
        img1 = self._get_input(input1, expoTimes[img_len]/expoTimes[img_len-1], True)
        img2 = self._get_input(input2, expoTimes[img_len+1]/expoTimes[img_len-1], True)

        return {'input0': img0, 
                'input1': img1, 
                'input2': img2, 
                'label': label,
                'other_label': label,
                'expo': expoTimes[img_len]/expoTimes[img_len-1],
                'fname': self.scenes_list[idx]}

    def _getitem_test_align(self, idx):
        ldr_file_path = list_all_files_sorted(os.path.join(self.root, 'Scenes_align', self.scenes_list[idx]), '.tif')
        expoTimes = read_expo_times(os.path.join(self.root, 'Scenes', self.scenes_list[idx], 'exposures.txt'))
        img_len = (len(ldr_file_path) + 2) // 2 - 1

        input0 = cv2.imread(ldr_file_path[0], -1)
        input1 = cv2.imread(ldr_file_path[1], -1)
        input2 = cv2.imread(ldr_file_path[2], -1)

        label = torch.from_numpy(np.float32(input1).transpose(2, 0, 1))

        img0 = self._get_input(input0, expoTimes[img_len-1]/expoTimes[img_len-1], True)
        img1 = self._get_input(input1, expoTimes[img_len]/expoTimes[img_len-1], True)
        img2 = self._get_input(input2, expoTimes[img_len+1]/expoTimes[img_len-1], True)

        return {'input0': img0, 
                'input1': img1, 
                'input2': img2, 
                'label': label,
                'other_label': label,
                'expo': expoTimes[img_len]/expoTimes[img_len-1],
                'fname': self.scenes_list[idx]}

    def _get_input(self, img, expotime, trans=False):
        if trans:
            img = img.transpose(2, 0, 1)  # CxHxW
        img = np.float32(img / 2 ** 16).clip(0, 1)
        pre_img = ldr_to_hdr(img, expotime, 2.2)
        pre_img = np.concatenate((pre_img, img), 0)
        return torch.from_numpy(pre_img)
