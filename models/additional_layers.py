import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class Transpose(nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)
