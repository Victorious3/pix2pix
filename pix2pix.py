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

    parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
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

    # ----------
    #  Training
    # ----------
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