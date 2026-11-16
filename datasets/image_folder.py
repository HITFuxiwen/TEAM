import os
import json
from PIL import Image

import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from datasets import register

@register('image-folder')
class ImageFolder(Dataset):
    def __init__(self, path,  split_file=None, split_key=None, first_k=None, size=None,
                 repeat=1, cache='none', mask=False, return_path=False):
        paths = path.split(';')
        self.repeat = repeat
        self.cache = cache
        self.path = paths[0]
        self.Train = False
        self.split_key = split_key
        self.return_path = return_path
        self.size = size
        self.mask = mask
        if self.mask:
            self.img_transform = transforms.Compose([
                transforms.Resize((self.size, self.size), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        self.files = []
        for path in paths:
            filenames = sorted(os.listdir(path))
            # for filename in filenames[:5000]:
            for filename in filenames:
                # print(filename)
                file = os.path.join(path, filename)
                self.append_file(file)
        # if split_file is None:
        #     filenames = sorted(os.listdir(path))
        # else:
        #     with open(split_file, 'r') as f:
        #         filenames = json.load(f)[split_key]
        # if first_k is not None:
        #     filenames = filenames[:first_k]

    def append_file(self, file):
        if self.cache == 'none':
            self.files.append(file)
        elif self.cache == 'in_memory':
            # cache in memory: store processed tensor (and maybe path)
            item = self.img_process(file)
            self.files.append((item, file) if self.return_path else item)

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        i = idx % len(self.files)

        if self.cache == 'none':
            file = self.files[i]
            x = self.img_process(file)
            return (x, file) if self.return_path else x

        elif self.cache == 'in_memory':
            x = self.files[i]
            return x

    def img_process(self, file):
        if self.mask:
            img = Image.open(file).convert('L')
        else:
            img = Image.open(file).convert('RGB')
        return img

@register('paired-image-folders')
class PairedImageFolders(Dataset):


    def __init__(self, root_path_1, root_path_2, cls_json_path, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs, return_path=True)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs, mask=True, return_path=False)
        self.cls_json_path = cls_json_path
        len1 = len(self.dataset_1)
        len2 = len(self.dataset_2)
        assert len1 == len2
        self.dir2cls = {
            "photoshop": 0,
            "deepfake": 1,   # TODO: 你替换为真实目录名
            "aigc": 2,   # TODO: 你替换为真实目录名
        }
        print(cls_json_path)
        if cls_json_path != 'none':
            print(cls_json_path)
            with open(cls_json_path, "r") as f:
                self.cls_dict = json.load(f)

    def cal_cls_from_json(self, file_path: str) -> int:
        file_name = os.path.basename(file_path)
        clss = self.cls_dict[file_name]
        clss = 2 - clss
        return clss

    def cal_cls_from_path(self, file_path: str) -> int:
        # 统一分隔符，避免 Windows/Unix 差异
        parts = os.path.normpath(file_path).split(os.sep)
        if len(parts) < 4:
            raise ValueError(f"path too short to take 3rd-from-end dir: {file_path}")

        key = parts[-4]  # 倒数第三级目录名
        if key not in self.dir2cls:
            raise KeyError(f"unknown class dir '{key}' in path: {file_path}")
        return int(self.dir2cls[key])
        
    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        img, path = self.dataset_1[idx]   # img: Tensor, path: str
        # print(idx, path)
        mask = self.dataset_2[idx]        # mask: Tensor
        # print(mask)
            # print("#########################")
        if self.cls_json_path != 'none':
            clss = self.cal_cls_from_json(path)
        else:
            clss = self.cal_cls_from_path(path)
        # print(path, clss)
        return img, mask, clss
