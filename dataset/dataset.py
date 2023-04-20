import os

import numpy as np
import torch.utils.data as da
from skimage import io


class CHDataset(da.Dataset):
    def __init__(self, root_dir: str, train=True, transform=None):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir  # 文件目录
        self.transform = transform  # 变换
        if train:
            self.root_dir = os.path.join(self.root_dir, f"train_28_28")
        else:
            self.root_dir = os.path.join(self.root_dir, f"testset_28_28")
        self.images = []  # 目录里的所有文件
        self.label_path = os.listdir(self.root_dir)
        self.labels = {}
        step = 0
        for label in self.label_path:
            self.labels[label] = step
            step += 1
        for path in self.label_path:
            image_path = os.listdir(os.path.join(self.root_dir, path))
            self.images = self.images + image_path
        # self.images = []

    def __len__(self):  # 返回整个数据集的大小
        return len(self.images)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        image_index = self.images[index]  # 根据索引index获取该图片
        label = image_index.split('_')[0]
        img_path = os.path.join(self.root_dir, label)  # 获取索引为index的图片的路径名
        img_path = os.path.join(img_path, image_index)
        img = io.imread(img_path)  # 读取该图片
        a= []
        a.append(img)
        a = np.array(a)
        # a = []
        # a.append(img)
        # tensor = torch.Tensor(np.array(a))
        # label = torch.Tensor(label)
        sample = {"images": a, "labels": self.labels[label]}  # 根据图片和标签创建字典
        # if self.transform:
            # sample = self.transform(sample)  # 对样本进行变换
        return sample  # 返回该样本
