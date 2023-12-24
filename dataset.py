import torch
import torchvision.transforms as transforms
from torch.utils import data as data
import torch.nn.functional
import os
import glob
from natsort import natsorted
import pandas as pd
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt


class multiDataset(data.Dataset):
    def __init__(self, videolist_file, img_dir, label_dir, n, transform=None, target_transform=None):
        video_list = pd.read_csv(videolist_file, header=None)
        self.transform = transform
        self.target_transform = target_transform
        self.n = n

        video_name_list = video_list[0].values
        
        img_list = []
        labels_list = []
        
        for video_name in video_name_list:
            img_files = glob.glob(os.path.join(img_dir, video_name, '*.png'))
            img_files = natsorted(img_files)
            
            labels_csv = os.path.join(label_dir, video_name + '.csv')
            labels_pd = pd.read_csv(labels_csv, header=None)
            labels_np = labels_pd.to_numpy()
            labels = labels_np[:,0]
            
            if len(img_files) % self.n != 0:
                labels = labels[:(-1) * (len(img_files) % self.n)]
                img_files = img_files[:(-1) * (len(img_files) % self.n)]
                
            #print(len(img_files))
            #print(len(labels))
            
            # n枚の画像で1つのデータを構成する
            img_files = [img_files[i:i+n] for i in range(0, len(img_files), n)]
            labels = [list(map(int, labels[i:i+n])) for i in range(0, len(labels), n)]
            
            #print(labels)
            #print(img_files)

            img_list.extend(img_files)   
            labels_list.extend(labels)
        
        self.img_list = img_list
        self.labels_list = labels_list

        #print(len(img_list))
        #print(len(labels_list))        
        #print([len(v) for v in img_list])
        
    def __len__(self):
        # サブセットの数
        return len(self.img_list)
        
    def __getitem__(self, idx):
        img_files = self.img_list[idx]
        labels = self.labels_list[idx]
        #print(img_files)
        #print(labels)
        # n枚の画像を読み込んで1つのデータを構成する
        img = []
        for img_file in img_files:
            img_ = cv2.imread(img_file)
            img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
            img_ = self.transform(image=img_)['image']
            img.append(img_)
        img = torch.stack(img, dim=0)
        
        labels = self.labels_list[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)
            
        #print(img.shape)
        #print(labels)
            
        return img, labels

    
class simpleDataset(data.Dataset):
    def __init__(self, videolist_file, img_dir, label_dir, n, transform=None, target_transform=None):
        video_list = pd.read_csv(videolist_file, header=None)
        self.transform = transform
        self.target_transform = target_transform
        self.n = n

        video_name_list = video_list[0].values
        
        img_list = []
        labels_list = np.empty(0, dtype=int)
        
        for video_name in video_name_list:
            img_files = glob.glob(os.path.join(img_dir, video_name, '*.png'))
            img_files = natsorted(img_files)
            
            labels_csv = os.path.join(label_dir, video_name + '.csv')
            labels_pd = pd.read_csv(labels_csv, header=None)
            labels_np = labels_pd.to_numpy()
            labels = labels_np[:,1]
            
            if len(img_files) % self.n != 0:
                labels = labels[:(-1) * (len(img_files) % self.n)]
                img_files = img_files[:(-1) * (len(img_files) % self.n)]
            
            img_list.append(img_files)   
            labels_list = np.append(labels_list, labels)
        
        img_list = sum(img_list, [])
        self.img_list = img_list
        self.labels_list = labels_list
        
        print(len(img_list))
        print(labels_list)
        #print(len(v) for v in img_list)
        
    def __len__(self):
        # サブセットの数
        return len(self.img_list)
        
    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx])
        #print(self.img_list[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)['image']

        label = self.labels_list[idx]
        if self.target_transform:
            label = self.target_transform(label)

        return img, label


class testDataset(data.Dataset):
    def __init__(self, videolist_file, img_dir, label_dir, n, transform=None, target_transform=None):
        video_list = pd.read_csv(videolist_file, header=None)
        self.transform = transform
        self.target_transform = target_transform
        self.n = n

        video_name_list = video_list[0].values
        
        img_list = []
        labels_list = []
        
        for video_name in video_name_list:
            img_files = glob.glob(os.path.join(img_dir, video_name, '*.png'))
            img_files = natsorted(img_files)
            
            labels_csv = os.path.join(label_dir, video_name + '.csv')
            labels_pd = pd.read_csv(labels_csv, header=None)
            labels_np = labels_pd.to_numpy()
            labels = labels_np[:,0]

            img_list.extend(img_files)   
            labels_list.extend(labels)
        
        self.img_list = img_list
        self.labels_list = labels_list
        
        #print(len(img_list))
        #print(len(labels_list))
        #print([len(v) for v in img_list])
        
    def __len__(self):
        # サブセットの数
        return len(self.img_list) - self.n + 1
        
    def __getitem__(self, idx):
        
        img = []
        labels = []
        for i in range(self.n):
            img_file = self.img_list[idx + i]
            img_ = cv2.imread(img_file)
            img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
            img_ = self.transform(image=img_)['image']
            img.append(img_)
            labels.append(self.labels_list[idx + i])
        img = torch.stack(img, dim=0)
        
        if self.target_transform is not None:
            label = self.target_transform(label)
        
        #print(img.shape)
        #print(labels)
        
        return img, labels


class maskedMultiDataset(data.Dataset):
    def __init__(self, videolist_file, img_dir, label_dir, n, system, transform=None, target_transform=None):
        video_list = pd.read_csv(videolist_file, header=None)
        self.transform = transform
        self.target_transform = target_transform
        self.n = n
        
        # 各撮影systemのマスク画像を使う場合
        if system == 'olympus':
            self.mask = cv2.imread('olympus_mask.png')
        elif system == 'fujifilm':
            self.mask = cv2.imread('fujifilm_mask.png')
        
        # 同じマスク画像を使う場合
        # olympus
        #self.mask = cv2.imread('olympus_resize_mask.png')
        # fujifilm
        #self.mask = cv2.imread('fujifilm_resize_mask.png')
        #totensor = transforms.ToTensor()
        #self.mask = totensor(self.mask)
        
        video_name_list = video_list[0].values
        
        img_list = []
        labels_list = []
        
        for video_name in video_name_list:
            img_files = glob.glob(os.path.join(img_dir, video_name, '*.png'))
            img_files = natsorted(img_files)
            
            labels_csv = os.path.join(label_dir, video_name + '.csv')
            labels_pd = pd.read_csv(labels_csv, header=None)
            labels_np = labels_pd.to_numpy()
            labels = labels_np[:,0]
            
            if len(img_files) % self.n != 0:
                labels = labels[:(-1) * (len(img_files) % self.n)]
                img_files = img_files[:(-1) * (len(img_files) % self.n)]
            
            # n枚の画像で1つのデータを構成する
            img_files = [img_files[i:i+n] for i in range(0, len(img_files), n)]
            labels = [list(map(int, labels[i:i+n])) for i in range(0, len(labels), n)]
            
            #print(labels)
            #print(img_files)

            img_list.extend(img_files)   
            labels_list.extend(labels)
        
        self.img_list = img_list
        self.labels_list = labels_list
        
        #print(len(img_list))
        #print(len(labels_list))
        #print([len(v) for v in img_list])
        
    def __len__(self):
        # サブセットの数
        return len(self.img_list)
        
    def __getitem__(self, idx):
        img_files = self.img_list[idx]
        labels = self.labels_list[idx]
        #print(img_files)
        #print(labels)
        # 3枚の画像を読み込んで1つのデータを構成する
        img = []
        for img_file in img_files:
            img_ = cv2.imread(img_file)
            # 各撮影systemのマスク画像を使う場合
            img_ = cv2.bitwise_and(img_, self.mask)
            img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
            img_ = self.transform(image=img_)['image']
            # 同じマスク画像を使う場合
            #img_ = img_ * self.mask
            
            img.append(img_)
        img = torch.stack(img, dim=0)
        
        labels = self.labels_list[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)
            
        #print(img.shape)
        #print(labels)
            
        return img, labels
    

class maskedTestDataset(data.Dataset):
    def __init__(self, video_name, img_dir, label_dir, n, system, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.n = n
        
        if system == 'olympus':
            self.mask = cv2.imread('olympus_mask.png')
        elif system == 'fujifilm':
            self.mask = cv2.imread('fujifilm_mask.png')
        
        # 同じマスク画像を使う場合
        # olympus
        #self.mask = cv2.imread('olympus_resize_mask.png')
        # fujifilm
        # self.mask = cv2.imread('fujifilm_resize_mask.png')
        # totensor = transforms.ToTensor()
        # self.mask = totensor(self.mask)
        
        img_list = []
        labels_list = []
        
        img_files = glob.glob(os.path.join(img_dir, video_name, '*.png'))
        img_files = natsorted(img_files)
        
        labels_csv = os.path.join(label_dir, video_name + '.csv')
        labels_pd = pd.read_csv(labels_csv, header=None)
        labels_np = labels_pd.to_numpy()
        labels = labels_np[:,0]

        img_list.extend(img_files)   
        labels_list.extend(labels)
        
        self.img_list = img_list
        self.labels_list = labels_list
        
        #print(len(img_list))
        #print(len(labels_list))
        #print([len(v) for v in img_list])
        
    def __len__(self):
        # サブセットの数
        return len(self.img_list) - self.n + 1
        
    def __getitem__(self, idx):
        
        img = []
        labels = []
        for i in range(self.n):
            img_file = self.img_list[idx + i]
            img_ = cv2.imread(img_file)
            # 各撮影systemのマスク画像を使う場合
            img_ = cv2.bitwise_and(img_, self.mask)
            img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
            img_ = self.transform(image=img_)['image']
            # 同じマスク画像を使う場合
            # img_ = img_ * self.mask
            img.append(img_)
            labels.append(self.labels_list[idx + i])
        img = torch.stack(img, dim=0)
        
        if self.target_transform is not None:
            label = self.target_transform(label)
        
        #print(img.shape)
        #print(labels)
        
        return img, labels