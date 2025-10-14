import torch
import torch.nn as nn
import torch.nn.functional as F
from args import *

'''VGG11/13/16/19 in Pytorch.'''
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):

        super(VGG, self).__init__()

        self.width = 32
        self.height = 32
        self.channel = 3

        self.conv = self._make_layers(cfg[vgg_name])

        self.linear = nn.Sequential(
            nn.Linear(512, 10),
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.channel
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def initialize_weights(self):
        print('init model VGG')
        if parser_args.init == 'xavier_normal':
            from torch.nn.init import xavier_normal_ as init_f
        elif parser_args.init == 'xavier_uniform':
            from torch.nn.init import xavier_uniform_ as init_f
        elif parser_args.init == 'kaiming_normal':
            from torch.nn.init import kaiming_normal_ as init_f
        elif parser_args.init == 'kaiming_uniform':
            from torch.nn.init import kaiming_uniform_ as init_f
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_f(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init_f(m.weight.data)
                m.bias.data.zero_()

def VGG16():
    return VGG('VGG16')