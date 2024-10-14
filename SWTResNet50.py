import datetime

import torch
import torch.nn as nn

# Other
DEVICE = "cuda:0"
GRAYSCALE = True


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SWTResNet50(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(SWTResNet50, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** .5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def wave1(self, x):
        """wavelet transform for the first level"""


    def wave2(self, x):
        """wavelet transform for the second level"""


    def wave3(self, x):
        """wavelet transform for the third level"""


    def wave4(self, x):
        """wavelet transform for the fourth level"""


    def forward(self, x):
        x_forward = self.conv1(x[0])
        x_forward = self.bn1(x_forward)
        x_forward = self.relu(x_forward)
        x_forward = self.maxpool(x_forward)
        x_forward = self.layer1(x_forward)

        H1 = self.wave1(x[4])
        x_forward = torch.add(x_forward, H1)
        x_forward = self.layer2(x_forward)

        H2 = self.wave2(x[3])
        x_forward = torch.add(x_forward, H2)
        x_forward = self.layer3(x_forward)

        H3 = self.wave3(x[2])
        x_forward = torch.add(x_forward, H3)
        x_forward = self.layer4(x_forward)

        H4 = self.wave4(x[1])
        x_forward = torch.add(x_forward, H4)
        x_forward = self.avgpool(x_forward)
        x_forward = x_forward.view(x_forward.size(0), -1)
        self.dropout(x_forward)
        logits = self.fc(x_forward)
        return logits

    def get_embedding(self, x):
        return self.forward(x)

def SWTResNet(pretrained=False, **kwargs):
    """Constructs a SWTResNet model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SWTResNet50(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

