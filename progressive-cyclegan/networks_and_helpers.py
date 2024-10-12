#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 14:09:22 2024

@author: mahera
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import itertools

from networks_and_helpers import *
from resnet import *


'''GENERATOR'''
class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9):
        """Initialize the generator with ResNet blocks."""
        super(Generator, self).__init__()
        # Initial convolution block
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 nn.InstanceNorm2d(ngf),
                 nn.ReLU(True)]
        
        # Downsampling layers
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      nn.InstanceNorm2d(ngf * mult * 2),
                      nn.ReLU(True)]
        
        # ResNet blocks (typically 6-9 blocks for 256x256 resolution)
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult)]
        
        # Upsampling layers
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1),
                      nn.InstanceNorm2d(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        
        # Output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)




'''DISCRIMINATOR'''
class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64):
        """Initialize the PatchGAN Discriminator.
        
        Args:
            input_nc (int): Number of input image channels (e.g., 3 for RGB).
            ndf (int): Number of filters in the last convolutional layer.
        """
        super(Discriminator, self).__init__()
        
        # Layer 1: Conv layer without normalization
        model = [nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]
        
        # Layer 2: Conv layer with normalization
        model += [nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
                  nn.InstanceNorm2d(ndf * 2),
                  nn.LeakyReLU(0.2, inplace=True)]
        
        # Layer 3: Conv layer with normalization
        model += [nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
                  nn.InstanceNorm2d(ndf * 4),
                  nn.LeakyReLU(0.2, inplace=True)]
        
        # Layer 4: Conv layer with normalization
        model += [nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
                  nn.InstanceNorm2d(ndf * 8),
                  nn.LeakyReLU(0.2, inplace=True)]
        
        # Layer 5: Final output layer (1x1 conv without normalization)
        # Outputs a single channel prediction map (real or fake) for each patch
        model += [nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)]
        
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# LOSS FUNCTIONS

def cycle_consistency_loss(real_img, reconstructed_img):
    return torch.mean(torch.abs(real_img - reconstructed_img))

def identity_loss(real_img, same_img):
    return torch.mean(torch.abs(real_img - same_img))




# PROGRESSIVE GROWING
def progressive_resize(epoch, image_size, max_size, start_epoch, grow_interval):
    if epoch >= start_epoch and (epoch - start_epoch) % grow_interval == 0:
        new_size = min(image_size * 2, max_size)
        if new_size != image_size:
            image_size = new_size
            print(f'Increasing image size to {image_size}x{image_size}')
    return image_size


