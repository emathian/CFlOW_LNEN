import os
from PIL import Image
import numpy as np
import torch
from torchvision.io import read_video, write_jpeg
from torch.utils.data import Dataset
from torchvision import transforms as T


__all__ = ('MVTecDataset')


class TumorNormalDataset(Dataset):
    def __init__(self, c, is_train=True):
        self.dataset_path = c.data_path
        self.class_name = c.class_name
        self.is_train = is_train
        self.cropsize = c.crp_size
        self.infer_train = c.infer_train
        self.list_file_test = c.list_file_test
        self.list_file_train = c.list_file_train
        # load dataset
        self.x, self.y, self.mask  = self.load_dataset_folder()
        # set transforms
        if is_train:
            self.transform_x = T.Compose([
                T.Resize(c.img_size, Image.ANTIALIAS),
                T.RandomRotation(5),
                T.CenterCrop(c.crp_size),
                T.ToTensor()])
        # test:
        else:
            self.transform_x = T.Compose([
                T.Resize(c.img_size, Image.ANTIALIAS),
                T.CenterCrop(c.crp_size),
                T.ToTensor()])
        # mask
        self.transform_mask = T.Compose([
            T.Resize(c.img_size, Image.NEAREST),
            T.CenterCrop(c.crp_size),
            T.ToTensor()])

        self.normalize = T.Compose([T.Normalize(c.norm_mean, c.norm_std)])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        #x = Image.open(x).convert('RGB')
        filepath = x
        x = Image.open(x)
        x = self.normalize(self.transform_x(x))
        mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])

        return x, y, mask, filepath

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        if self.is_train :
            phase = 'train' 
            list_file =  self.list_file_train #'TrainTumorNormal.txt'
        else:
            if not self.infer_train:
                phase = 'test'
                list_file = self.list_file_test #'TestTumorNormal.txt'
            else:
                phase = 'train'
                list_file =   self.list_file_train #'TrainTumorNormal.txt'
        x, y, mask = [], [], []
        img_dir = os.path.join(self.dataset_path, list_file)
        with open(img_dir, 'r') as f:
            content =  f.readlines()
        files_list = []
        for l in content:
            l =  l.strip()
            if l.find('Tumor') != -1:
                y.append(1)
            else:
                y.append(0)
            files_list.append(l)
        files_list = sorted(files_list)
        x.extend(files_list)
        mask.extend([None] * len(files_list))

        return list(x), list(y), list(mask)

    
    
class TCACDataset(Dataset):
    def __init__(self, c, is_train=True):
        self.dataset_path = c.data_path
        self.class_name = c.class_name
        self.is_train = is_train
        self.cropsize = c.crp_size
        self.infer_train = c.infer_train
        self.list_file_test = c.list_file_test
        self.list_file_train = c.list_file_train
        # load dataset
        self.x, self.y, self.mask  = self.load_dataset_folder()
        # set transforms
        if is_train:
            self.transform_x = T.Compose([
                T.Resize(c.img_size, Image.ANTIALIAS),
                T.RandomRotation(5),
                T.CenterCrop(c.crp_size),
                T.ToTensor()])
        # test:
        else:
            self.transform_x = T.Compose([
                T.Resize(c.img_size, Image.ANTIALIAS),
                T.CenterCrop(c.crp_size),
                T.ToTensor()])
        # mask
        self.transform_mask = T.Compose([
            T.Resize(c.img_size, Image.NEAREST),
            T.CenterCrop(c.crp_size),
            T.ToTensor()])

        self.normalize = T.Compose([T.Normalize(c.norm_mean, c.norm_std)])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        #x = Image.open(x).convert('RGB')
        filepath = x
        x = Image.open(x)
        x = self.normalize(self.transform_x(x))
        mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])

        return x, y, mask, filepath

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        if self.is_train :
            phase = 'train' 
            list_file =  self.list_file_train #'TrainTumorNormal.txt'
        else:
            if not self.infer_train:
                phase = 'test'
                list_file = self.list_file_test #'TestTumorNormal.txt'
            else:
                phase = 'train'
                list_file =   self.list_file_train #'TrainTumorNormal.txt'
        x, y, mask = [], [], []
        img_dir = os.path.join(self.dataset_path, list_file)
        with open(img_dir, 'r') as f:
            content =  f.readlines()
        files_list = []
        for l in content:
            l =  l.strip()
            y.append(0)
            files_list.append(l)
        files_list = sorted(files_list)
        x.extend(files_list)
        mask.extend([None] * len(files_list))

        return list(x), list(y), list(mask)
