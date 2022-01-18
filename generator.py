# *******Coding: UTF-8*******
# Developed by Elysium on PyCharm
# Create Time: 15/1/2022 下午9:27
from torch import nn
from parameters import Parameters


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = Parameters.ngpu
        self.main = nn.Sequential(
            # BEFORE: input: nz
            nn.ConvTranspose2d(Parameters.nz, Parameters.ngf * 32, 4, 1, 0, bias=False),
            nn.BatchNorm2d(Parameters.ngf * 32),
            nn.ReLU(True),
            # AFTER: (ngf*32) * 4 * 4

            nn.ConvTranspose2d(Parameters.ngf * 32, Parameters.ngf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Parameters.ngf * 16),
            nn.ReLU(True),
            # AFTER: (ngf*16) * 8 * 8

            nn.ConvTranspose2d(Parameters.ngf * 16, Parameters.ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Parameters.ngf * 8),
            nn.ReLU(True),
            # AFTER: (ngf*8) * 16 * 16

            nn.ConvTranspose2d(Parameters.ngf * 8, Parameters.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Parameters.ngf * 4),
            nn.ReLU(True),
            # AFTER: (ngf*4) * 32 * 32

            nn.ConvTranspose2d(Parameters.ngf * 4, Parameters.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Parameters.ngf * 2),
            nn.ReLU(True),
            # AFTER: (ngf*2) * 64 * 64

            nn.ConvTranspose2d(Parameters.ngf * 2, Parameters.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Parameters.ngf),
            nn.ReLU(True),
            # AFTER: (ngf*16) * 128 * 128

            nn.ConvTranspose2d(Parameters.ngf, Parameters.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # AFTER: (nc) * 256 * 256
        )

    def forward(self, input):
        return self.main(input)
