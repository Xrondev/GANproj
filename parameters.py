# *******Coding: UTF-8*******
# Developed by Elysium on PyCharm
# Create Time: 15/1/2022 下午9:36

class Parameters:
    dataroot = "../Data/"
    workers = 1
    batch_size = 32  # set bigger if you have enough video memory.
    # spatial size of training images. All images will be resized to this size using a transformer
    image_size = 256
    # number of channels in the training images. For RGB color, it is 3
    nc = 3
    # size of z latent vector. (i.e. size of generator input)
    nz = 100
    # size of feature maps in generator / discriminator
    ngf = 64
    ndf = 64
    # number of training epochs
    num_epochs = 5
    # learning rate for optimizers
    lr = 0.0002
    # beta 1 hyper param for Adam optimizers
    beta1 = 0.5
    # number of GPUs availiable. 0 for CPU mode
    ngpu = 1
