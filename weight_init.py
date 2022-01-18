# *******Coding: UTF-8*******
# Developed by Elysium on PyCharm
# Create Time: 15/1/2022 下午10:08
from torch import nn

'''
From the DCGAN paper, the authors specify that all model weights shall be randomly initialized 
from a Normal distribution with mean=0, stdev=0.02. 
The weights_init function takes an initialized model as input and reinitializes 
all convolutional, convolutional-transpose, 
and batch normalization layers to meet this criteria.
'''


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
