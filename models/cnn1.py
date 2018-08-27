from collections import OrderedDict

import torch
import torch.nn as nn

from models.utils import weights_init
from models.additional_layers import Transpose, Flatten
from models.convbrelu import ConvBNReLu
# from utils import weights_init
# from additional_layers import Transpose, Flatten
# from convbrelu import ConvBNReLu


"""
 Implementation of EnvNet [Tokozume and Harada, 2017]
 opt.fs = 16000
 opt.inputLength = 2**16
 channels_num = 220
 width = 128

# """


class CNN1(nn.Module):
    def __init__(self, n_classes):
        super(CNN1, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('conv1', ConvBNReLu(220, 128, (1, 1))),
            ('conv2', ConvBNReLu(128, 128, (1, 3))),
            ('maxpool1', nn.MaxPool2d((1, 2), ceil_mode = True)),
            ('flatten', Flatten()),
            ('fc1', nn.Linear(in_features = 128*1*63, out_features = 4096, bias = True)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout()),
            ('fc2', nn.Linear(4096, 4096)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout()),
            ('fc3', nn.Linear(4096, n_classes)),
            ('softmax', nn.Softmax(dim=-1))
            ]))

    def forward(self, inp):
        return self.model(inp)


if __name__ == "__main__":
    inp = torch.empty(2, 220, 1, 128).uniform_(0, 1)
    model = CNN1(10)
    model.apply(weights_init)
    out = model(inp)
    print(out.size())
#     print(out)
