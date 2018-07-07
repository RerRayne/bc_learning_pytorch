import torch.nn as nn

from models.convbrelu import ConvBNReLu

"""
 Implementation of EnvNet [Tokozume and Harada, 2017]
 opt.fs = 16000
 opt.inputLength = 24014

# """


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class EnvNet(nn.Module):
    def __init__(self, n_classes):
        super(EnvNet, self).__init__()
        self.conv1 = ConvBNReLu(1, 40, (1, 8))
        self.conv2 = ConvBNReLu(40, 40, (1, 8))
        self.max_pool2 = nn.MaxPool2d((1, 160), ceil_mode=True)

        self.conv3 = ConvBNReLu(1, 50, (8, 13))
        self.max_pool3 = nn.MaxPool2d((3, 3), ceil_mode=True)

        self.conv4 = ConvBNReLu(50, 50, (1, 5))
        self.max_pool4 = nn.MaxPool2d((1, 3), ceil_mode=True)

        self.flatten = Flatten()

        self.fc5 = nn.Linear(in_features=50 * 11 * 14, out_features=4096, bias=True)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout()

        self.fc6 = nn.Linear(4096, 4096)
        self.relu6 = nn.ReLU()
        self.dropout6 = nn.Dropout()

        self.fc7 = nn.Linear(4096, n_classes)
        self.out_l = nn.LogSoftmax()

    def forward(self, inp):
        out = self.conv1(inp)
        out = self.conv2(out)
        out = self.max_pool2(out)

        out = out.transpose(1, 2)

        out = self.conv3(out)
        out = self.max_pool3(out)

        out = self.conv4(out)
        out = self.max_pool4(out)

        out = self.flatten(out)

        out = self.fc5(out)
        out = self.relu5(out)
        out = self.dropout5(out)

        out = self.fc6(out)
        out = self.relu6(out)
        out = self.dropout6(out)

        out = self.fc7(out)
        out = self.out_l(out)
        return out
