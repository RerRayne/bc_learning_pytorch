from collections import OrderedDict

import torch
import torch.nn as nn

from models.utils import weights_init
from models.additional_layers import Flatten, Transpose

"""
 Implementation of EnvNet2 [Tokozume and Harada, 2017]
 opt.fs = 44000
 opt.inputLength = 66650
 Layer ksize stride # of filters Data shape
 Input (1, 1, 66,650)
 conv1 (1, 64) (1, 2) 32
 conv2 (1, 16) (1, 2) 64
 pool2 (1, 64) (1, 64) (64, 1, 260)
 swapaxes (1, 64, 260)
 conv3, 4 (8, 8) (1, 1) 32
 pool4 (5, 3) (5, 3) (32, 10, 82)
 conv5, 6 (1, 4) (1, 1) 64
 pool6 (1, 2) (1, 2) (64, 10, 38)
 conv7, 8 (1, 2) (1, 1) 128
 pool8 (1, 2) (1, 2) (128, 10, 18)
 conv9, 10 (1, 2) (1, 1) 256
 pool10 (1, 2) (1, 2) (256, 10, 8)
 fc11 - - 4096 (4,096,)
 fc12 - - 4096 (4,096,)
 fc13 - - # of classes (# of classes,)

# """


class EnvReLu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(EnvReLu, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, input):
        return self.layer(input)


class EnvNet2(nn.Module):
    def __init__(self, n_classes):
        super(EnvNet2, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('conv1', EnvReLu(in_channels=1,
                              out_channels=32,
                              kernel_size=(1, 64),
                              stride=(1, 2),
                              padding=0)),
            ('conv2', EnvReLu(in_channels=32,
                              out_channels=64,
                              kernel_size=(1, 16),
                              stride=(1, 2),
                              padding=0)),
            ('max_pool2', nn.MaxPool2d(kernel_size=(1, 64),
                                       stride=(1, 64),
                                       ceil_mode=True)),
            ('transpose', Transpose()),
            ('conv3', EnvReLu(in_channels=1,
                              out_channels=32,
                              kernel_size=(8, 8),
                              stride=(1, 1),
                              padding=0)),
            ('conv4', EnvReLu(in_channels=32,
                              out_channels=32,
                              kernel_size=(8, 8),
                              stride=(1, 1),
                              padding=0)),
            ('max_pool4', nn.MaxPool2d(kernel_size=(5, 3),
                                       stride=(5, 3),
                                       ceil_mode=True)),
            ('conv5', EnvReLu(in_channels=32,
                              out_channels=64,
                              kernel_size=(1, 4),
                              stride=(1, 1),
                              padding=0)),
            ('conv6', EnvReLu(in_channels=64,
                              out_channels=64,
                              kernel_size=(1, 4),
                              stride=(1, 1),
                              padding=0)),
            ('max_pool6', nn.MaxPool2d(kernel_size=(1, 2),
                                       stride=(1, 2),
                                       ceil_mode=True)),
            ('conv7', EnvReLu(in_channels=64,
                              out_channels=128,
                              kernel_size=(1, 2),
                              stride=(1, 1),
                              padding=0)),
            ('conv8', EnvReLu(in_channels=128,
                              out_channels=128,
                              kernel_size=(1, 2),
                              stride=(1, 1),
                              padding=0)),
            ('max_pool8', nn.MaxPool2d(kernel_size=(1, 2),
                                       stride=(1, 2),
                                       ceil_mode=True)),
            ('conv9', EnvReLu(in_channels=128,
                              out_channels=256,
                              kernel_size=(1, 2),
                              stride=(1, 1),
                              padding=0)),
            ('conv10', EnvReLu(in_channels=256,
                               out_channels=256,
                               kernel_size=(1, 2),
                               stride=(1, 1),
                               padding=0)),
            ('max_pool10', nn.MaxPool2d(kernel_size=(1, 2),
                                        stride=(1, 2),
                                        ceil_mode=True)),
            ('flatten', Flatten()),
            ('fc11', nn.Linear(in_features=256 * 10 * 8, out_features=4096, bias=True)),
            ('relu11', nn.ReLU()),
            ('dropout11', nn.Dropout()),
            ('fc12', nn.Linear(in_features=4096, out_features=4096, bias=True)),
            ('relu12', nn.ReLU()),
            ('dropout12', nn.Dropout()),
            ('fc13', nn.Linear(in_features=4096, out_features=n_classes, bias=True)),
            ('softmax', nn.Softmax(dim=-1))
        ]))

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    inp = torch.empty(2, 1, 1, 66650).uniform_(0, 1)
    model = EnvNet2(5)
    model.apply(weights_init)
    out = model(inp)
    print(out.size())
    print(out)
