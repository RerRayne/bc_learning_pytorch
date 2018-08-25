from collections import OrderedDict

import torch
import torch.nn as nn

from models.utils import weights_init
from models.additional_layers import Transpose, Flatten
from models.convbrelu import ConvBNReLu

"""
 Implementation of EnvNet [Tokozume and Harada, 2017]
 opt.fs = 16000
 opt.inputLength = 24014

# """


class EnvNet(nn.Module):
    def __init__(self, n_classes):
        super(EnvNet, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('conv1', ConvBNReLu(1, 40, (1, 8))),
            ('conv2', ConvBNReLu(40, 40, (1, 8))),
            ('max_pool2', nn.MaxPool2d((1, 160), ceil_mode=True)),
            ('transpose', Transpose()),
            ('conv3', ConvBNReLu(1, 50, (8, 13))),
            ('max_pool3', nn.MaxPool2d((3, 3), ceil_mode=True)),
            ('conv4', ConvBNReLu(50, 50, (1, 5))),
            ('max_pool4', nn.MaxPool2d((1, 3), ceil_mode=True)),
            ('flatten', Flatten()),
            ('fc5', nn.Linear(in_features=50 * 11 * 14, out_features=4096, bias=True)),
            ('relu5', nn.ReLU()),
            ('dropout5', nn.Dropout()),
            ('fc6', nn.Linear(4096, 4096)),
            ('relu6', nn.ReLU()),
            ('dropout6', nn.Dropout()),
            ('fc7', nn.Linear(4096, n_classes)),
            ('softmax', nn.Softmax(dim=-1))
        ]))

    def forward(self, inp):
        return self.model(inp)


if __name__ == "__main__":
    inp = torch.empty(2, 1, 1, 24014).uniform_(0, 1)
    model = EnvNet(5)
    model.apply(weights_init)
    out = model(inp)
    print(out.size())
    print(out)
