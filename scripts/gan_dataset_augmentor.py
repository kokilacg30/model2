import os
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, utils, datasets
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

# Configuration
DATA_DIR = Path(r"e:\nitin\model2\dataset\classification\train")
AUG_DIR = Path(r"e:\nitin\model2\dataset\classification\train_augmented")

IMG_SIZE = 128
LATENT_DIM = 100
NUM_CLASSES = 3
BATCH_SIZE = 16
GAN_EPOCHS = 12

# Define Conditional GAN Generator
class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=3, img_channels=3):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        
        self.init_size = IMG_SIZE // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim * 2, 128 * self.init_size ** 2))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        c_emb = self.label_emb(labels)
        gen_input = torch.cat((noise, c_emb), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# Define Conditional GAN Discriminator
class Discriminator(nn.Module):
    def __init__(self, num_classes=3, img_channels=3):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, IMG_SIZE * IMG_SIZE)

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(img_channels + 1, 32, bn=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
        )

        # Output layer
        ds_size = IMG_SIZE // 16
        self.adv_layer = nn.Sequential(nn.Linear(256 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img, labels):
        c_emb = self.label_emb(labels).view(labels.size(0), 1, IMG_SIZE, IMG_SIZE)
        d_in = torch.cat((img, c_emb), 1)
        out = self.model(d_in)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

def train_gan_and_augment(num_samples_per_class=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==================================================")
    print(f"STREET AIQ: GAN DATASET AUGMENTOR (Device: {device})")
    print(f"==================================================")
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    generator = Generator(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(device)
    discriminator = Discriminator(num_classes=NUM_CLASSES).to(device)
    
    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    print("Training Conditional GAN for Synthetic Road/Trash Augmentation...")
    
    for epoch in range(1, GAN_EPOCHS + 1):
        for i, (imgs, labels) in enumerate(dataloader):
            batch_size = imgs.shape[0]
            real_imgs = imgs.to(device)
            labels = labels.to(device)

            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, LATENT_DIM, device=device)
            gen_labels = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)
            gen_imgs = generator(z, gen_labels)
            
            g_loss = adversarial_loss(discriminator(gen_imgs, gen_labels), valid)
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs, labels), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), gen_labels), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

        print(f"GAN Epoch [{epoch:02d}/{GAN_EPOCHS}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
        
    print("\nGAN Training Complete! Generating Synthetic Augmented Images...")
    
    # Generate Synthetic Data & Save to Augmented Training Dataset
    classes = ['clean_roads', 'slightly_dirty', 'very_dirty']
    generator.eval()
    
    for class_idx, class_name in enumerate(classes):
        out_class_dir = AUG_DIR / class_name
        out_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy existing original images
        orig_dir = DATA_DIR / class_name
        for orig_img in orig_dir.glob("*.*"):
            dest = out_class_dir / orig_img.name
            if not dest.exists():
                with open(orig_img, 'rb') as sf, open(dest, 'wb') as df:
                    df.write(sf.read())
                    
        # Generate GAN synthetic samples
        with torch.no_grad():
            z = torch.randn(num_samples_per_class, LATENT_DIM, device=device)
            labels = torch.full((num_samples_per_class,), class_idx, dtype=torch.long, device=device)
            gen_imgs = generator(z, labels)
            
            # De-normalize images (-1, 1 -> 0, 1)
            gen_imgs = (gen_imgs + 1.0) / 2.0
            
            for idx in range(num_samples_per_class):
                save_path = out_class_dir / f"gan_synth_{class_name}_{idx:03d}.jpg"
                utils.save_image(gen_imgs[idx], save_path)
                
        print(f"  --> Added {num_samples_per_class} GAN synthetic images to class '{class_name}'")
        
    print(f"\nGAN Augmentation Finished! Augmented Dataset Ready at: {AUG_DIR}")

if __name__ == "__main__":
    train_gan_and_augment(num_samples_per_class=100)
