#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 05:20:52 2024

@author: mahera
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np 
import time
import matplotlib.pyplot as plt
import argparse
import os

import torch.nn.functional as F

# from model import *
from functions import *

from gan_architecture3 import *
from utils2 import *





class Trainer():

    def __init__(self, data_directory, z_dim, n_classes, batch_size, lr=0.0002, beta_1=0.5, beta_2=0.999, device = 'cpu', num_workers=2, data_shape=(3,128,128)):
        #load dataset
        transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)
        ])

        #check if data directory exists, if not, create it.
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
            print('Directory created.')
        else:
            print('Directory exists.')

        #get dataset from directory. If not present, download to directory
        self.dataset = torchvision.datasets.ImageFolder(root=data_directory, transform=transformation)
        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.device = device
        self.n_classes = n_classes
        self.data_shape = data_shape
        
        '''Gen + Critic Initialization'''
        generator_input_dim, critic_im_chan = get_input_dimensions(self.z_dim, self.data_shape, self.n_classes)

        self.gen = Generator(input_dim=generator_input_dim).to(device)
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=self.lr, betas=(self.beta_1, self.beta_2))
        self.crit = Discriminator(im_chan=critic_im_chan).to(device)
        self.crit_opt = torch.optim.Adam(self.crit.parameters(), lr=self.lr, betas=(self.beta_1, self.beta_2))
        

    def train(self, n_epochs, saved_image_directory, saved_model_directory, c_lambda, crit_repeats, display_step):
        start_time = time.time()
        cur_time = start_time
        cur_step = 0
        generator_losses = []
        critic_losses = []

        fake_image_and_labels = False
        real_image_and_labels = False
        crit_fake_pred = False
        crit_real_pred = False

        for epoch in range(n_epochs):
            
            # Dataloader returns the batches and the labels
            for real, labels in tqdm(self.data_loader):
                cur_batch_size = len(real)
                # Flatten the batch of real images from the dataset
                real = real.to(self.device)
        
                one_hot_labels = get_one_hot_labels(labels.to(self.device), self.n_classes)
                image_one_hot_labels = one_hot_labels[:, :, None, None]
                image_one_hot_labels = image_one_hot_labels.repeat(1, 1, self.data_shape[1], self.data_shape[2])
        
                mean_iteration_critic_loss = 0
                for _ in range(crit_repeats):
                    '''### Update critic ###'''
                    self.crit_opt.zero_grad()
                    fake_noise = get_noise(cur_batch_size, self.z_dim, device=self.device)
                    
                    
                    noise_and_labels = combine_vectors(fake_noise, one_hot_labels)
                    fake = self.gen(noise_and_labels)
        
                    fake_image_and_labels = combine_vectors(fake, image_one_hot_labels)
                    real_image_and_labels = combine_vectors(real, image_one_hot_labels)
                    crit_fake_pred = self.crit(fake_image_and_labels.detach())
                    crit_real_pred = self.crit(real_image_and_labels)
        
                    epsilon = torch.rand(len(real), 1, 1, 1, device=self.device, requires_grad=True)
                    gradient = get_gradient(self.crit, real_image_and_labels, fake_image_and_labels.detach(), epsilon)
                    gp = gradient_penalty(gradient)
                    crit_loss = get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda)
        
                    # Keep track of the average critic loss in this batch
                    mean_iteration_critic_loss += crit_loss.item() / crit_repeats
                    # Update gradients
                    crit_loss.backward(retain_graph=True)
                    # Update optimizer
                    self.crit_opt.step()
                critic_losses += [mean_iteration_critic_loss]
        
                ''' Update generator '''
                self.gen_opt.zero_grad()
        
                fake_image_and_labels = combine_vectors(fake, image_one_hot_labels)
                crit_fake_pred = self.crit(fake_image_and_labels)
                gen_loss = get_gen_loss(crit_fake_pred)
        
                gen_loss.backward()
                '''Update the weights'''
                self.gen_opt.step()
        
                # Keep track of the average generator loss
                generator_losses += [gen_loss.item()]
        
                # Training
                if cur_step % display_step == 0 and cur_step > 0:
                    gen_mean = sum(generator_losses[-display_step:]) / display_step
                    crit_mean = sum(critic_losses[-display_step:]) / display_step
                    
                    cur_time = time.time() - cur_time
                    print(f"Step {cur_step}: Generator loss: {gen_mean}, critic loss: {crit_mean}")
                    print('Time Taken: {:.4f} seconds. Estimated {:.4f} hours remaining'.format(cur_time, (n_epochs-epoch)*(cur_time)/3600))
                    
                    show_tensor_images(fake)
                    show_tensor_images(real)
                    step_bins = 20
                    num_examples = (len(generator_losses) // step_bins) * step_bins
                    plt.plot(
                        range(num_examples // step_bins),
                        torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
                        label="Generator Loss"
                    )
                    plt.plot(
                        range(num_examples // step_bins),
                        torch.Tensor(critic_losses[:num_examples]).view(-1, step_bins).mean(1),
                        label="Critic Loss"
                    )
                    plt.legend()
                    plt.show()
        
                elif cur_step == 0:
                    print("~~~Training has begun~~~")
                cur_step += 1
            #save models to model_directory
            torch.save(self.gen.state_dict(), saved_model_directory + '/generator_{}.pt'.format(epoch))
            torch.save(self.crit.state_dict(), saved_model_directory + '/critic_{}.pt'.format(epoch))
        finish_time = time.time() - start_time
        print('Training Finished. Took {:.4f} seconds or {:.4f} hours to complete.'.format(finish_time, finish_time/3600))
        return generator_losses, critic_losses



'''----------------------------------------------------------------------------------------'''
def main():
    parser = argparse.ArgumentParser(description='Hyperparameters for training GAN')
    #hyperparameter loading
    parser.add_argument('--data_directory', type=str, default='data', help='directory to dataset files')
    parser.add_argument('--saved_image_directory', type=str, default='data/saved_images', help='directory to where image samples will be saved')
    parser.add_argument('--saved_model_directory', type=str, default='saved_models', help='directory to where model weights will be saved')
    

    
    parser.add_argument('--n_epochs', type=int, default=100, help='number of iterations of dataset through network for training')
    parser.add_argument('--z_dim', type=int, default=64, help='size of gaussian noise vector')
    parser.add_argument('--n_classes', type=int, default=4, help='number of unique classes in dataset')
    parser.add_argument('--display_step', type=int, default=50, help='number of iterations after which images and losses are printed.')
    parser.add_argument('--batch_size', type=int, default=128, help='size of batches passed through networks at each step')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate of models')
    parser.add_argument('--beta_1', type=float, default=0.5, help='beta 1')
    parser.add_argument('--beta_2', type=float, default=0.999, help='beta 2')
    parser.add_argument('--c_lambda', type=int, default=10, help='regularization factor')
    parser.add_argument('--crit_repeats', type=int, default=5, help='number of times critic is updated for every generator update')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or gpu depending on availability and compatability')
    parser.add_argument('--num_workers', type=int, default=0, help='workers simultaneously putting data into RAM')

    args = parser.parse_args()

    data_dir = args.data_directory
    saved_image_dir = args.saved_image_directory
    saved_model_dir = args.saved_model_directory
    
    n_epochs = args.n_epochs
    z_dim = args.z_dim
    n_classes = args.n_classes
    display_step = args.display_step
    batch_size = args.batch_size
    lr = args.lr
    beta_1 = args.beta_1
    beta_2 = args.beta_2
    c_lambda = args.c_lambda
    crit_repeats = args.crit_repeats
    device = args.device
    num_workers = args.num_workers
    

    gan = Trainer(data_dir, z_dim, n_classes, batch_size, lr, beta_1, beta_2, device, num_workers)
    gen_loss_lost, dis_loss_list = gan.train(n_epochs, saved_image_dir, saved_model_dir, display_step, c_lambda, crit_repeats)
    # test = gan.train_print(n_epochs)

if __name__ == "__main__":
    main()
    
    # data_dir = 'dataset/00resized_images'
    # saved_image_dir = 'dataset/00resized_images/saved_images'
    # saved_model_dir = 'dataset/00resized_images/saved_models'

    # n_epochs = 100
    # z_dim = 64
    # n_classes = 4
    # display_step = 10
    # batch_size = 20
    # lr = 0.0002
    # beta_1 = 0.5
    # beta_2 = 0.999
    # c_lambda = 10
    # crit_repeats = 5
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # num_workers = 2
    
    # gan = Trainer(data_dir, z_dim, n_classes, batch_size, lr, beta_1, beta_2, device, num_workers)
    # gen_loss_lost, dis_loss_list = gan.train(n_epochs, saved_image_dir, saved_model_dir, display_step, c_lambda, crit_repeats)
