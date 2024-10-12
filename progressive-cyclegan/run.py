#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 14:11:47 2024

@author: mahera
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import itertools

from networks_and_helpers import *
from train import *

# Model, optimizer, and dataloader initialization
G_A2B = Generator(input_nc=3, output_nc=3)  # Replace with actual generator initialization
G_B2A = Generator(input_nc=3, output_nc=3)
D_A = Discriminator(input_nc=3)  # Replace with actual discriminator initialization
D_B = Discriminator(input_nc=3)

# Optimizers for generators and discriminators
optimizer_G = optim.Adam(list(G_A2B.parameters()) + list(G_B2A.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(list(D_A.parameters()) + list(D_B.parameters()), lr=0.0002, betas=(0.5, 0.999))

# Create learning rate schedulers
scheduler_G, scheduler_D = create_schedulers(optimizer_G, optimizer_D, epochs=200, decay_start_epoch=100)

# Dataloaders
dataloader_A = torch.utils.data.DataLoader(...)  # Replace with actual DataLoader for domain A
dataloader_B = torch.utils.data.DataLoader(...)  # Replace with actual DataLoader for domain B

# Train the model
train_cycle_gan(epochs=200, dataloader_A=dataloader_A, dataloader_B=dataloader_B, G_A2B=G_A2B, G_B2A=G_B2A, D_A=D_A, D_B=D_B, 
                optimizer_G=optimizer_G, optimizer_D=optimizer_D, scheduler_G=scheduler_G, scheduler_D=scheduler_D,
                start_size=128, max_size=512, grow_interval=10)

