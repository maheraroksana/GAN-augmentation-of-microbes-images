#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 23:23:22 2024

@author: mahera
"""

import torch

'''TRAINING INITIALIZATION'''

data_directory = 'dataset/00resized_images'
saved_image_directory = 'dataset/00resized_images/saved_images'
saved_model_directory = 'saved_models'

n_epochs = 100
z_dim = 64
n_classes = 4
display_step = 50
batch_size = 128
lr = 0.0002
beta_1 = 0.5
beta_2 = 0.999
c_lambda = 10
crit_repeats = 5
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
num_workers = 2

import numpy as np
import subprocess  
import os

# os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

command = [
    'python', 'train.py',
    '--data_directory', data_directory,
    '--saved_image_directory', saved_image_directory,
    '--saved_model_directory', saved_model_directory,
    '--n_epochs', str(n_epochs),
    '--z_dim', str(z_dim),
    '--n_classes', str(n_classes),
    '--display_step', str(display_step),
    '--batch_size', str(batch_size),
    '--lr', str(lr),
    '--beta_1', str(beta_1),
    '--beta_2', str(beta_2),
    '--c_lambda', str(c_lambda),
    '--crit_repeats', str(crit_repeats),
    '--device', device,
    '--num_workers', str(num_workers)
]

# Run the command
subprocess.run(command)
