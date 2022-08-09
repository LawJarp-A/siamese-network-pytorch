import numpy as np
import pandas as pd
from PIL import Image
import glob
import random

import torch.nn.functional as F

import timm

import torch, torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader
from torchvision.io import read_image

from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning import miners, losses

class SimpleDataset(Dataset):
    def __init__(self, df, is_valid=False, transform=None):
        self.files = list(df['file_path'])
        self.labels = list(df['class_num'])
        self.transform = transform
        self.basic_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Resize((224,224)),
            transforms.Normalize(0,1),
        ])
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, i):
        file1 = self.files[i]
        img = self.get_image(file1)
        pn = self.labels[i]
        return (img, torch.Tensor([pn]).squeeze())
    
    def get_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.basic_transforms(image)
        if self.transform:
            image = self.transform(image)
        return image