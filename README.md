# pix2pix
Teamproject pix2pix Group 2.<br />
Team members: Daniel, Jialu, Lasse<br />
Project duration: 26.Juni.2020 - 26.Juli.2020<br />

<img src="image/pix2pix_introduction_image.jpg">

## Documentation<br />
#### 02.07.2020<br />
We created our pix2pix repository.

#### 07.07.2020<br />
We had our first discussion and learned more about pix2pix.

#### 12.07.2020<br />
The structure of our code will be like :<br />
  - Dataset<br />
  - Load<br />
  - Optimize<br />
  - Discriminator and Generator<br />
  - Sample<br />
  - Training<br />

  We added discriminator and generator in models.py<br />

#### 20.07.2020<br />
The training Loop has beed added.Our code are almost done,meanwhile some trouble have been fixed.Also we tried some different variables to see how the results would change. <br />

## Tutorial on how to run the pix2pix in Colab<br />
1.Open the Notebook in Google Colab with the following link: <br />
https://colab.research.google.com/github/Victorious3/pix2pix/blob/master/pix2pix.ipynb<br />
2.In order to run the whole pix2pix-code,go under 'Run all' over the dropdown menu 'Runtime'.<br />

## Overview<br />
The Pix2Pix Generative Adversarial Network(GAN) is a framework based on adversarial neural network, which can realize image-to-image translation.The goal of image-to-image translation tasks is to output images that based on the input images(soruce images),such as converting maps to satellite photographs, black and white photographs to color, and sketches of products to product photographs.<br />
<img src="image/overview_image.png">

### Specific process <br />
Since it's based on the GAN framework,we need to frist define the input and output. The input of Generator(G) received by ordinary GAN is a random vector, and the output is an image; the input received by Discriminator(D) is an image (generated or real), and the output is real or fake. This way G and D can work together to output real images.<br />
But for the image-to-image translation tasks, the input of G should obviously be a picture x, and the output is a picture y. However, some changes should be made to the input of D, because in addition to generating a real image, it is also necessary to ensure that the generated image and the input image match.In this case,the input of D has beed changed to a loss function.<br />


## Explanation of code<br />
### Import<br />
~~~
import glob
import random
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
~~~
### Dataset<br />
~~~
_URL = 'bash ./datasets/download_pix2pix_dataset.sh facades'

class ImageDataset(Dataset):
    def __init__(self, args, root, transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.args = args
        self.files = sorted(glob.glob(os.path.join(root, mode) + '/*.*'))

    def __getitem__(self, index):

        img = Image.open(self.files[index])
        w, h = img.size

        if self.args.which_direction == 'AtoB':
            img_A = img.crop((0, 0, w/2, h))
            img_B = img.crop((w/2, 0, w, h))
        else:
            img_B = img.crop((0, 0, w/2, h))
            img_A = img.crop((w/2, 0, w, h))


        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], 'RGB')
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], 'RGB')

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {'A': img_A, 'B': img_B}

    def __len__(self):
        return len(self.files)
~~~
### Loader<br />
~~~
def Get_dataloader(args):
    transforms_ = [ transforms.Resize((args.img_height, args.img_width), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

    train_dataloader = DataLoader(ImageDataset(args, "%s/%s" % (args.data_root,args.dataset_name),
                        transforms_=transforms_,mode='train'),
                        batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu, drop_last=True)

    test_dataloader = DataLoader(ImageDataset(args, "%s/%s" % (args.data_root,args.dataset_name),
                            transforms_=transforms_, mode='test'),
                            batch_size=10, shuffle=True, num_workers=1, drop_last=True)

    val_dataloader = DataLoader(ImageDataset(args, "%s/%s" % (args.data_root,args.dataset_name),
                            transforms_=transforms_, mode='val'),
                            batch_size=10, shuffle=True, num_workers=1, drop_last=True)

    return train_dataloader, test_dataloader, val_dataloader
~~~
### Optimizer<br /> 
~~~
import torch
# Adam Optimizers
def Get_optimizer_func(args, generator, discriminator):
    ArgLr = args.lr
    ArgB1 = args.b1
    ArgB2 = args.b2

    optimizer_G = torch.optim.Adam(
                    generator.parameters(),
                    lr=ArgLr, betas=(ArgB1, ArgB2))
    optimizer_D = torch.optim.Adam(
                    discriminator.parameters(),
                    lr=ArgLr, betas=(ArgB1, ArgB2))
    
    return optimizer_G, optimizer_D
~~~

### Loss function<br />
~~~
# Loss functions
def Get_loss_func(args):
    crit_GAN = torch.nn.BCELoss()
    crit_pixelwise = torch.nn.L1Loss()
    if torch.cuda.is_available():
        crit_GAN.cuda()
        crit_pixelwise.cuda()
    return crit_GAN, crit_pixelwise
~~~

### Discriminator<br />
~~~
import torch.nn as nn
import torch

class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.out = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Sigmoid()
        )

    def forward(self, img_a, img_b):
        input_img = torch.cat((img_a, img_b), 1)
        x = self.layer1(input_img)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.out(x)
        return x
~~~
### UNet
~~~ 
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [  nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
                    nn.InstanceNorm2d(out_size),
                    nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x

~~~
### Generator <br />
~~~
class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.up8(u7)
        
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)


~~~
### Initialize G & D
~~~
def Create_nets(args):
    generator = GeneratorUNet(args.in_channels, args.out_channels)
    discriminator = Discriminator(args.out_channels)

    if torch.cuda.is_available():
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    return generator, discriminator
~~~
### Sample
~~~
import argparse
import torch
import time
import numpy as np
import datetime
import sys

from torch.autograd import Variable
from torchvision.utils import save_image

from models import Create_nets
from datasets import Get_dataloader
from optimizer import Get_loss_func, Get_optimizer_func

def sample_images(generator, test_dataloader, args, epoch, batches_done):
    """Saves a generated sample from the validation set"""

    imgs = next(iter(test_dataloader))
    real_A = Variable(imgs['A'].type(torch.FloatTensor).cuda())
    real_B = Variable(imgs['B'].type(torch.FloatTensor).cuda())
    fake_B = generator(real_A)
    img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
    save_image(img_sample, '%s/%s/%s-%s.png' % (args.dataset_name, args.img_result_dir, batches_done, epoch), nrow=5, normalize=True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Pix2Pix")

    parser.add_argument('--epoch_start', type=int, default=0, help='epoch to start training from')
    parser.add_argument('--epoch_num', type=int, default=200, help='number of epochs of training')
    parser.add_argument('--data_root', type=str, default='../../data/', help='dir of the dataset')
    parser.add_argument('--dataset_name', type=str, default="facades", help='name of the dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
    parser.add_argument('--img_height', type=int, default=256, help='size of image height')
    parser.add_argument('--img_width', type=int, default=256, help='size of image width')
    parser.add_argument('--in_channels', type=int, default=3, help='number of input image channels')
    parser.add_argument('--out_channels', type=int, default=3, help='number of output image channels')
    parser.add_argument('--sample_interval', type=int, default=200, help='interval between sampling of images from generators')
    parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
    parser.add_argument('--lambda_pixel', type=int, default=100, help='Loss weight of L1 pixel-wise loss between translated image and real image')
    parser.add_argument('--img_result_dir', type=str, default='result_images', help='where to save the result images')

    args = parser.parse_args()

    # Initialize generator and discriminator
    generator, discriminator = Create_nets(args)
    # Loss functions
    criterion_GAN, criterion_pixelwise = Get_loss_func(args)
    # Optimizers
    optimizer_G, optimizer_D = Get_optimizer_func(args, generator, discriminator)

    # Configure dataloaders
    train_dataloader, test_dataloader, _ = Get_dataloader(args)
~~~
### Training
~~~
    prev_time = time.time()
    for epoch in range(args.epoch_start, args.epoch_num):
        for i, batch in enumerate(train_dataloader):

            # Model inputs
            real_A = Variable(batch['A'].type(torch.FloatTensor).cuda())
            real_B = Variable(batch['B'].type(torch.FloatTensor).cuda())

            # Adversarial ground truths
            valid = Variable(torch.FloatTensor(np.ones((real_A.size(0), 1, 6, 6))).cuda(), requires_grad=False)
            fake = Variable(torch.FloatTensor(np.zeros((real_A.size(0), 1, 6, 6))).cuda(), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------
            optimizer_G.zero_grad()

            #loss
            fake_B = generator(real_A)
            pred_fake = discriminator(fake_B, real_A)
            loss_GAN = criterion_GAN(pred_fake, valid)
            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(fake_B, real_B)

            # Total loss
            loss_G = loss_GAN + args.lambda_pixel * loss_pixel
            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            # Real loss
            pred_real = discriminator(real_B, real_A)
            loss_real = criterion_GAN(pred_real, valid)

            # Fake loss
            pred_fake = discriminator(fake_B.detach(), real_A)
            loss_fake = criterion_GAN(pred_fake, fake)

            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------
            # Determine approximate time left
            batches_done = epoch * len(train_dataloader) + i
            batches_left = args.epoch_num * len(train_dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write("\r[Epoch%d/%d] - [Batch%d/%d] - [Dloss:%f] - [Gloss:%f, loss_pixel:%f, adv:%f] ETA:%s" %
                (epoch + 1, args.epoch_num,
                i, len(train_dataloader),
                loss_D.data.cpu(), loss_G.data.cpu(),
                loss_pixel.data.cpu(), loss_GAN.data.cpu(),
                time_left))

            # If at sample interval save image
            if batches_done % args.sample_interval == 0:
                sample_images(generator, test_dataloader, args, epoch, batches_done)
~~~


## Summary<br />


### Output<br />
