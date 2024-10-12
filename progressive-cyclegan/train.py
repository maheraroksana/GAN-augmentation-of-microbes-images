import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from networks_and_helpers import *

# Loss functions
criterion_GAN = nn.MSELoss()  # LSGAN loss for adversarial losses
criterion_cycle = nn.L1Loss()  # L1 loss for cycle consistency
criterion_identity = nn.L1Loss()  # L1 loss for identity mapping

# Define the argument parser
def get_args():
    parser = argparse.ArgumentParser(description="CycleGAN Training")

    # Training parameters
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start decaying the learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')

    # Image size for progressive resizing
    parser.add_argument('--start_size', type=int, default=128, help='starting image size for progressive resizing')
    parser.add_argument('--max_size', type=int, default=1024, help='maximum image size')
    parser.add_argument('--grow_interval', type=int, default=10, help='interval (in epochs) to grow image size')

    # Paths for data
    parser.add_argument('--dataroot_A', type=str, required=True, help='path to dataset A')
    parser.add_argument('--dataroot_B', type=str, required=True, help='path to dataset B')

    return parser.parse_args()

# Initialize learning rate schedulers
def create_schedulers(optimizer_G, optimizer_D, epochs, decay_start_epoch):
    def lambda_rule(epoch):
        lr_decay = 1.0 - max(0, epoch + 1 - decay_start_epoch) / float(epochs - decay_start_epoch)
        return lr_decay

    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule)
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda_rule)

    return scheduler_G, scheduler_D

def train_cycle_gan(epochs, dataloader_A, dataloader_B, G_A2B, G_B2A, D_A, D_B, optimizer_G, optimizer_D, scheduler_G, scheduler_D, start_size, max_size, grow_interval):
    current_size = start_size
    
    for epoch in range(epochs):
        # Update image size progressively
        current_size = progressive_resize(epoch, current_size, max_size, start_epoch=10, grow_interval=grow_interval)
        
        # Update learning rate schedulers if needed
        scheduler_G.step()
        scheduler_D.step()
        
        for i, (real_A, real_B) in enumerate(zip(dataloader_A, dataloader_B)):
            ##########################
            # Train Generators G_A2B and G_B2A
            ##########################
            optimizer_G.zero_grad()
            
            # Forward pass through the generators
            fake_B = G_A2B(real_A)  # G_A2B tries to generate B from A
            rec_A = G_B2A(fake_B)   # Cycle back to A
            fake_A = G_B2A(real_B)  # G_B2A tries to generate A from B
            rec_B = G_A2B(fake_A)   # Cycle back to B
            
            # GAN loss for generators
            loss_G_A2B = criterion_GAN(D_B(fake_B), torch.ones_like(D_B(fake_B)))
            loss_G_B2A = criterion_GAN(D_A(fake_A), torch.ones_like(D_A(fake_A)))
            
            # Cycle consistency loss
            loss_cycle_A = criterion_cycle(real_A, rec_A)
            loss_cycle_B = criterion_cycle(real_B, rec_B)
            
            # Identity loss (for preserving color/composition)
            loss_id_A = criterion_identity(real_A, G_B2A(real_A))
            loss_id_B = criterion_identity(real_B, G_A2B(real_B))
            
            # Total generator loss
            loss_G = loss_G_A2B + loss_G_B2A + loss_cycle_A + loss_cycle_B + loss_id_A + loss_id_B
            loss_G.backward()  # Backpropagate the loss
            optimizer_G.step()  # Update the weights of the generators
            
            ##########################
            # Train Discriminators D_A and D_B
            ##########################
            optimizer_D.zero_grad()

            # Discriminator A loss
            loss_D_A_real = criterion_GAN(D_A(real_A), torch.ones_like(D_A(real_A)))
            loss_D_A_fake = criterion_GAN(D_A(fake_A.detach()), torch.zeros_like(D_A(fake_A)))
            loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5
            
            # Discriminator B loss
            loss_D_B_real = criterion_GAN(D_B(real_B), torch.ones_like(D_B(real_B)))
            loss_D_B_fake = criterion_GAN(D_B(fake_B.detach()), torch.zeros_like(D_B(fake_B)))
            loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5
            
            # Backpropagate and update discriminators
            loss_D_A.backward()
            loss_D_B.backward()
            optimizer_D.step()

            # Print training progress every 100 iterations
            if i % 100 == 0:
                print(f"Epoch {epoch}, Iteration {i}, Generator Loss: {loss_G.item()}, Discriminator Loss: {(loss_D_A.item() + loss_D_B.item()) / 2}")

    print("Training Complete!")

# Main training function
if __name__ == "__main__":
    args = get_args()

    # Initialize the models
    G_A2B = Generator(input_nc=3, output_nc=3)  # Replace with your Generator class
    G_B2A = Generator(input_nc=3, output_nc=3)
    D_A = Discriminator(input_nc=3)  # Replace with your Discriminator class
    D_B = Discriminator(input_nc=3)

    # Optimizers for generators and discriminators
    optimizer_G = optim.Adam(list(G_A2B.parameters()) + list(G_B2A.parameters()), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_D = optim.Adam(list(D_A.parameters()) + list(D_B.parameters()), lr=args.lr, betas=(args.beta1, args.beta2))

    # Create learning rate schedulers
    scheduler_G, scheduler_D = create_schedulers(optimizer_G, optimizer_D, epochs=args.epochs, decay_start_epoch=args.decay_epoch)

    # Example DataLoader initialization
    dataloader_A = torch.utils.data.DataLoader(...)  # Load dataset A
    dataloader_B = torch.utils.data.DataLoader(...)  # Load dataset B

    # Train the model
    train_cycle_gan(
        epochs=args.epochs, 
        dataloader_A=dataloader_A, 
        dataloader_B=dataloader_B, 
        G_A2B=G_A2B, 
        G_B2A=G_B2A, 
        D_A=D_A, 
        D_B=D_B, 
        optimizer_G=optimizer_G, 
        optimizer_D=optimizer_D, 
        scheduler_G=scheduler_G, 
        scheduler_D=scheduler_D,
        start_size=args.start_size, 
        max_size=args.max_size, 
        grow_interval=args.grow_interval
    )