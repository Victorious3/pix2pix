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

# Loss functions
def Get_loss_func(args):
    crit_GAN = torch.nn.BCELoss()
    crit_pixelwise = torch.nn.L1Loss()
    if torch.cuda.is_available():
        crit_GAN.cuda()
        crit_pixelwise.cuda()
    return crit_GAN, crit_pixelwise
