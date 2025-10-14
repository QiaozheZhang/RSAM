"""resnet in pytorch
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn

from args.args_utils import parser_args

class tinyBasicBlock(nn.Module):
    M = 2
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, base_width=64):
        super(tinyBasicBlock, self).__init__()
        if base_width / 64 > 1:
            raise ValueError("Base width >64 does not work for BasicBlock")
        
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels))
        
        self.shortcut = nn.Sequential()

        if downsample is not None:
            self.shortcut = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.residual_function(x)

        if self.shortcut is not None:
            residual = self.shortcut(x)

        out = nn.ReLU(inplace=True)(out+residual)

        return out

# Bottleneck {{{
class tinyBottleneck(nn.Module):
    M = 3
    expansion = 4

    def __init__(self, in_channels, planes, stride=1, downsample=None, base_width=64):
        super(tinyBottleneck, self).__init__()
        width = int(planes * base_width / 64)

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=1, bias=False),
            nn.BatchNorm2d(width),
            nn.Conv2d(width, width, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * self.expansion),
            nn.ReLU(inplace=True))
        
        self.shortcut = nn.Sequential()

        if downsample is not None:
            self.shortcut = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.residual_function(x)

        if self.shortcut is not None:
            residual = self.shortcut(x)

        out = nn.ReLU(inplace=True)(out+residual)
        
        return out


# Bottleneck }}}

# ResNet {{{
class tinyResNet(nn.Module):
    def __init__(self, block, layers, num_classes=200, base_width=64):
        self.inplanes = 64
        super(tinyResNet, self).__init__()

        self.base_width = base_width
        if self.base_width // 64 > 1:
            print(f"==> Using {self.base_width // 64}x wide model")

        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True))
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_x = self._make_layer(block, 64, layers[0], stride=1)
        self.conv3_x = self._make_layer(block, 128, layers[1], stride=2)
        self.conv4_x = self._make_layer(block, 256, layers[2], stride=2)
        self.conv5_x = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            dconv = nn.Conv2d(self.inplanes,  planes * block.expansion, kernel_size=1, stride=stride, bias=False)
            dbn = nn.BatchNorm2d(planes * block.expansion)
            if dbn is not None:
                downsample = nn.Sequential(dconv, dbn)
            else:
                downsample = dconv

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, base_width=self.base_width))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, base_width=self.base_width))

        return nn.Sequential(*layers)

    

    def forward(self, x):
        x = self.conv1(x)
        #x = self.maxpool(x)

        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
    def initialize_weights(self):
        print(f'Initializing model with {parser_args.init}')
        init_methods = {
            'xavier_normal': nn.init.xavier_normal_,
            'xavier_uniform': nn.init.xavier_uniform_,
            'kaiming_normal': lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu'),
            'kaiming_uniform': lambda w: nn.init.kaiming_uniform_(w, nonlinearity='relu'),
        }

        init_f = init_methods.get(parser_args.init, None)
        if init_f is None:
            print(f'[Warning] Unknown init method: {parser_args.init}. Skip initialization.')
            return

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if hasattr(m, 'weight') and m.weight is not None:
                    init_f(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# ResNet }}}
def TinyResNet18():
    return tinyResNet(tinyBasicBlock, [2, 2, 2, 2], 200)


def TinyResNet50():
    return tinyResNet(tinyBottleneck, [3, 4, 6, 3], 200)


def TinyResNet101():
    return tinyResNet(tinyBottleneck, [3, 4, 23, 3], 200)


def TinyWideResNet50_2():
    return tinyResNet(tinyBottleneck, [3, 4, 6, 3], num_classes=200, base_width=64 * 2
    )


def TinyWideResNet101_2():
    return tinyResNet(tinyBottleneck, [3, 4, 23, 3], num_classes=200, base_width=64 * 2
    )
