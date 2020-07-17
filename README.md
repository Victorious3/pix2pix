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
  - Install<br />
  - Dataset<br />
  - Load<br />
  - Randomize<br />
  - Training<br />
  - Testing<br />
  
  We added discriminator and generator in models.py<br />



## Tutorial on how to run the pix2pix in Colab<br />
1.Open the Notebook in Google Colab with the following link: <br />
<br />
2.In order to run the whole pix2pix-code,go under 'Run all' over the dropdown menu 'Runtime'.<br />

## Overview<br />
The Pix2Pix Generative Adversarial Network(GAN) is a framework based on adversarial neural network, which can realize image-to-image translation.The goal of image-to-image translation tasks is to output images that based on the input images(soruce images),such as converting maps to satellite photographs, black and white photographs to color, and sketches of products to product photographs.<br />
<img src="image/overview_image.png">

### Specific process <br />
Since it's based on the GAN framework,we need to frist define the input and output. The input of Generator(G) received by ordinary GAN is a random vector, and the output is an image; the input received by Discriminator(D) is an image (generated or real), and the output is real or fake. This way G and D can work together to output real images.<br />
But for the image-to-image translation tasks, the input of G should obviously be a picture x, and the output is a picture y. However, some changes should be made to the input of D, because in addition to generating a real image, it is also necessary to ensure that the generated image and the input image match.In this case,the input of D has beed changed to a loss function.<br />


## Explanation of code<br />
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

~~~
## Summary<br />


### Output<br />
