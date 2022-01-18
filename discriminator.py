# *******Coding: UTF-8*******
# Developed by Elysium on PyCharm
# Create Time: 18/1/2022 上午10:47
from torch import nn
from generator import Parameters

'''
Discriminator is a binary classifier.
Input: Image from the batch
Output: a scalar probability that the input image is real or fake
'''


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = Parameters.ngpu
        self.main = nn.Sequential(
            # input nc * 256 * 256
            nn.Conv2d(Parameters.nc, Parameters.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(inplace=True),
            # state size ndf  * 128 * 128
            nn.Conv2d(Parameters.ndf, Parameters.ndf * 2, 4, 2, 1),
            nn.BatchNorm2d(Parameters.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*2) * 64 * 64
            nn.Conv2d(Parameters.ndf * 2, Parameters.ndf * 4, 4, 2, 1),
            nn.BatchNorm2d(Parameters.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 32 32
            nn.Conv2d(Parameters.ndf * 4, Parameters.ndf * 8, 4, 2, 1),
            nn.BatchNorm2d(Parameters.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 16 16
            nn.Conv2d(Parameters.ndf * 8, Parameters.ndf * 16, 4, 2, 1),
            nn.BatchNorm2d(Parameters.ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # 8 8
            nn.Conv2d(Parameters.ndf * 16, Parameters.ndf * 32, 4, 2, 1),
            nn.BatchNorm2d(Parameters.ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # 4 4

            nn.Conv2d(Parameters.ndf * 32, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
