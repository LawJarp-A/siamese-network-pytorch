import numpy as np
import pandas as pd
from PIL import Image
import glob
import random

import torch.nn.functional as F

import timm

import pytorch_lightning as pl

import torch, torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader
from torchvision.io import read_image

from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning import miners, losses

import torch.nn as nn
class baseModel(pl.LightningModule):
    def __init__(self, model_name, embedding_size, steps_per_epoch, custom_model=None, pretrained=True):
        super().__init__()
        if custom_model:
            fe = custom_model
        else:
            fe = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.fe = fe
        self.model = nn.Sequential(
            self.fe,
            nn.Flatten(),
            nn.LazyLinear(1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.LazyLinear(embedding_size))
        # Using triplet mining here
        # Triplet mining greatly incresing training sppeed and efficiency
        self.loss_fn = losses.TripletMarginLoss(margin=0.3)
        self.miner = miners.TripletMarginMiner(margin=0.3, type_of_triplets="hard")
        self.pair_wise_dist = LpDistance(normalize_embeddings=True, p=2, power=1)
        self.steps_per_epoch = steps_per_epoch

    def forward(self, images):
        features1 = self.model(images)
        return (features1)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                           max_lr=0.01, 
                                                           steps_per_epoch=self.steps_per_epoch, 
                                                           epochs = 150)
        return [optimizer], [lr_scheduler]
    
    def training_step(self, batch, batch_idx):
        x,y = batch
        embs = self(x)
        hard_pairs = self.miner(embs, y)
        loss = self.loss_fn(embs, y, hard_pairs)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x,y = batch
        embs = self(x)
        val_loss = self.loss_fn(embs, y)
        self.log("val_loss", val_loss)