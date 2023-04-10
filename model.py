import numpy as np
from glob import glob
import torch.nn as nn
from torchvision import models
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torchvision.models.resnet import BasicBlock, Bottleneck, model_urls
import copy
from torch import Tensor
import torch
import math
import torch.nn.functional as F
import random


class SCNorm(nn.Module):
    def __init__(self, channels, channel_first=True):
        super().__init__()
        self.channels = channels
        self.channel_first = channel_first
        if channel_first:
            self.weight = nn.Parameter(torch.ones(channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(channels, 1, 1))
            self.h = nn.Parameter(torch.zeros(2, 1, 1))
        else:
            self.weight = nn.Parameter(torch.ones(channels))
            self.bias = nn.Parameter(torch.zeros(channels))
            self.h = nn.Parameter(torch.zeros(2))
        self.T = 0.1

    def forward(self, x):
        weight = F.softmax(self.h/self.T, dim=0)
        
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2)
        norm = F.instance_norm(x)
        phase, amp_ori = decompose(x, 'all')
        norm_phase, norm_amp = decompose(norm, 'all')
        amp = norm_amp * weight[0] + amp_ori * weight[1]
        x = compose(phase, amp)
        if not self.channel_first:
            x = x.permute(0, 2, 3, 1)
            
        return x * self.weight + self.bias


def ccnorm(residual, bn, weight):
    norm = F.batch_norm(
        residual, bn.running_mean, bn.running_var,
        None, None, bn.training, bn.momentum, bn.eps)
    mean = torch.mean(residual, dim=(0, 2, 3), keepdim=True) if bn.training else bn.running_mean.reshape(-1, 1, 1)

    weight = F.softmax(weight/1e-6, dim=0)
    biased_residual = residual - mean * weight[0]
    phase, _ = decompose(biased_residual, 'phase')
    _, norm_amp = decompose(norm, 'amp')

    residual = compose(
        phase,
        norm_amp
    )
    
    residual = residual * bn.weight.view(1, -1, 1, 1) + bn.bias.view(1, -1, 1, 1)
    return residual


def pcnorm(residual, bn):
    norm = F.batch_norm(
        residual, bn.running_mean, bn.running_var,
        None, None, bn.training, bn.momentum, bn.eps)

    phase, _ = decompose(residual, 'phase')
    _, norm_amp = decompose(norm, 'amp')

    residual = compose(
        phase,
        norm_amp
    )
    
    residual = residual * bn.weight.view(1, -1, 1, 1) + bn.bias.view(1, -1, 1, 1)
    return residual


def replace_denormals(x, threshold=1e-5):
    y = x.clone()
    y[(x < threshold) & (x > -1.0 * threshold)] = threshold
    return y


def decompose(x, mode='all'):
    fft_im = torch.view_as_real(torch.fft.fft2(x, norm='backward'))
    if mode == 'all' or mode == 'amp':
        fft_amp = fft_im.pow(2).sum(dim=-1, keepdim=False)
        fft_amp = torch.sqrt(replace_denormals(fft_amp))
    else:
        fft_amp = None

    if mode == 'all' or mode == 'phase':
        fft_pha = torch.atan2(fft_im[..., 1], replace_denormals(fft_im[..., 0]))
    else:
        fft_pha = None
    return fft_pha, fft_amp


def compose(phase, amp):
    x = torch.stack([torch.cos(phase) * amp, torch.sin(phase) * amp], dim=-1) 
    x = x / math.sqrt(x.shape[2] * x.shape[3])
    x = torch.view_as_complex(x)
    return torch.fft.irfft2(x, s=x.shape[2:], norm='ortho')


def conv3x3(in_planes, out_planes, stride = 1, groups = 1, dilation = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride = 1,
        downsample = None,
        groups = 1,
        base_width = 64,
        dilation = 1,
        norm_layer = None,
    ) -> None:
        super().__init__()
        # norm_layer = GNorm
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.resnorm = True

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample[0](x)
            identity = compose(
                decompose(identity, 'phase')[0],
                decompose(self.downsample[1](x), 'amp')[1]
            )

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, 
        inplanes,
        planes,
        stride = 1,
        downsample = None,
        groups = 1,
        base_width = 64,
        dilation = 1,
        norm_layer = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        if self.downsample is not None:
            self.weight = nn.Parameter(torch.zeros(2))

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
            residual = self.downsample[0](x)
            residual = ccnorm(residual, self.downsample[1], self.weight)

        out += residual
        out = self.relu(out)
        return out


class BottleneckCC3(nn.Module):
    expansion = 4

    def __init__(self, 
        inplanes,
        planes,
        stride = 1,
        downsample = None,
        groups = 1,
        base_width = 64,
        dilation = 1,
        norm_layer = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.weight = nn.Parameter(torch.zeros(2))

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        if self.downsample is not None:
            out = ccnorm(out, self.bn3, self.weight)
        else:
            out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class BottleneckCC2(nn.Module):
    expansion = 4

    def __init__(self, 
        inplanes,
        planes,
        stride = 1,
        downsample = None,
        groups = 1,
        base_width = 64,
        dilation = 1,
        norm_layer = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.weight = nn.Parameter(torch.zeros(2))

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            out = ccnorm(out, self.bn2, self.weight)
        else:
            out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class BottleneckCC1(nn.Module):
    expansion = 4

    def __init__(self, 
        inplanes,
        planes,
        stride = 1,
        downsample = None,
        groups = 1,
        base_width = 64,
        dilation = 1,
        norm_layer = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.weight = nn.Parameter(torch.zeros(2))

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        if self.downsample is not None:
            out = ccnorm(out, self.bn1, self.weight)
        else:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        # out = ccnorm(out, self.bn2, self.weight)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        # out = ccnorm(out, self.bn3, self.weights3)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class BottleneckPC3(nn.Module):
    expansion = 4

    def __init__(self, 
        inplanes,
        planes,
        stride = 1,
        downsample = None,
        groups = 1,
        base_width = 64,
        dilation = 1,
        norm_layer = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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
        if self.downsample is not None:
            out = pcnorm(out, self.bn3)
        else:
            out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class BottleneckPC2(nn.Module):
    expansion = 4

    def __init__(self, 
        inplanes,
        planes,
        stride = 1,
        downsample = None,
        groups = 1,
        base_width = 64,
        dilation = 1,
        norm_layer = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            out = pcnorm(out, self.bn2)
        else:
            out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class BottleneckPC1(nn.Module):
    expansion = 4

    def __init__(self, 
        inplanes,
        planes,
        stride = 1,
        downsample = None,
        groups = 1,
        base_width = 64,
        dilation = 1,
        norm_layer = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.downsample is not None: 
            out = pcnorm(out, self.bn1)
        else:
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


class DAC(models.ResNet):
    """ResNets without fully connected layer"""

    def __init__(self, *args, **kwargs):
        super(DAC, self).__init__(*args, **kwargs)
        self._out_features = self.fc.in_features
        self.resnorm = False
        self.norms = nn.ModuleList([
            SCNorm(c) for c in [256, 512, 1024]])
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.norms[0](x)
        x = self.layer2(x)
        x = self.norms[1](x)
        x = self.layer3(x)
        x = self.norms[2](x)
        x = self.layer4(x)
        return x

    @property
    def out_features(self) -> int:
        """The dimension of output features"""
        return self._out_features

    def copy_head(self) -> nn.Module:
        """Copy the origin fully connected layer"""
        return copy.deepcopy(self.fc)


def _dac(arch, block, layers, pretrained, progress, **kwargs):
    model = DAC(block, layers, **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        # remove keys from pretrained dict that doesn't appear in model dict
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        model.load_state_dict(pretrained_dict, strict=False)
    return model


def DAC50(pretrained=True, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _dac('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def DAC50_CC1(pretrained=True, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _dac('resnet50', BottleneckCC1, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def DAC50_PC1(pretrained=True, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _dac('resnet50', BottleneckPC1, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def DAC50_CC3(pretrained=True, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _dac('resnet50', BottleneckCC3, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def DAC50_PC3(pretrained=True, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _dac('resnet50', BottleneckPC3, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def DAC50_CC2(pretrained=True, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _dac('resnet50', BottleneckCC2, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def DAC50_PC2(pretrained=True, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _dac('resnet50', BottleneckPC2, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


class Classifier(nn.Module):

    def __init__(self, backbone: nn.Module, num_classes: int):
        super(Classifier, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.pool_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten()
        )
        self.head = nn.Linear(2048, num_classes)

    def forward(self, x):
        f = self.backbone(x)
        f = self.pool_layer(f)
        predictions = self.head(f)

        return predictions
