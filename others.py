from torchvision.models import swin_t, Swin_B_Weights, mobilenet_v2, MobileNet_V2_Weights, Swin_T_Weights
from torch import nn
import torch
import torch.nn.functional as F
import math
from model import SCNorm, replace_denormals


def decompose(x, mode='all'):
    fft_im = torch.view_as_real(torch.fft.fft(x, norm='backward'))
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
    x = x / math.sqrt(x.shape[3])
    x = torch.view_as_complex(x)
    return torch.fft.irfft(x, n=x.shape[-1], norm='ortho')



class SWIN_SCNorm(nn.Module):
    def __init__(self, channels, channel_first=True):
        super().__init__()
        self.channels = channels
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.h = nn.Parameter(torch.zeros(2))
        self.T = 0.1

    def forward(self, x):
        weight = F.softmax(self.h/self.T, dim=0)
        shape = x.shape
        norm = F.instance_norm(x.reshape(shape[0], -1, shape[-1]))
        norm = norm.reshape(shape)
        # norm = F.layer_norm(x, x.shape[-1:], None, None, 1e-5)
        # norm = norm.permute(0, 3, 1, 2)
        # x = x.permute(0, 3, 1, 2)
        phase, amp_ori = decompose(x, 'all')
        norm_phase, norm_amp = decompose(norm, 'all')
        amp = norm_amp * weight[0] + amp_ori * weight[1]
        x = compose(phase, amp)
        
        # x = x.permute(0, 2, 3, 1)
            
        return x * self.weight + self.bias

class SWIN_CC(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.weight = norm.weight
        self.bias = norm.bias
        self.shape = norm.weight.shape[0]
        self.eps = norm.eps
        self.weight_ = nn.Parameter(torch.zeros(2))

    def forward(self, x):
        norm = F.layer_norm(x, x.shape[-1:], None, None, self.eps)
        # norm = norm.permute(0, 3, 1, 2)
        # x = x.permute(0, 3, 1, 2)
        mean = torch.mean(x, dim=(-1), keepdim=True)

        weight = F.softmax(self.weight_/1e-6, dim=0)
        biased_residual = x - mean * weight[0]
        phase, _ = decompose(biased_residual, 'phase')
        _, norm_amp = decompose(norm, 'amp')

        residual = compose(
            phase,
            norm_amp
        )
        # residual = residual.permute(0, 2, 3, 1)
        
        residual = residual * self.weight + self.bias
        return residual


class SWIN_PC(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.weight = norm.weight
        self.bias = norm.bias
        self.shape = norm.weight.shape[0]
        self.eps = norm.eps
        self.weight_ = nn.Parameter(torch.zeros(2))

    def forward(self, x):
        norm = F.layer_norm(x, x.shape[-1:], None, None, self.eps)
        norm = norm.permute(0, 3, 1, 2)
        x = x.permute(0, 3, 1, 2)
        # mean = torch.mean(x, dim=(1), keepdim=True)

        # weight = F.softmax(self.weight_/1e-6, dim=0)
        # biased_residual = x - mean * weight[0]
        phase, _ = decompose(x, 'phase')
        _, norm_amp = decompose(norm, 'amp')

        residual = compose(
            phase,
            norm_amp
        )
        residual = residual.permute(0, 2, 3, 1)
        
        residual = residual * self.weight + self.bias
        return residual


class SWIN_PC(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.weight = norm.weight
        self.bias = norm.bias
        self.shape = norm.weight.shape[0]
        self.eps = norm.eps
        self.weight_ = nn.Parameter(torch.zeros(2))

    def forward(self, x):
        norm = F.layer_norm(x, x.shape[-1:], None, None, self.eps)
        norm = norm.permute(0, 3, 1, 2)
        x = x.permute(0, 3, 1, 2)
        # mean = torch.mean(x, dim=(1), keepdim=True)

        # weight = F.softmax(self.weight_/1e-6, dim=0)
        # biased_residual = x - mean * weight[0]
        phase, _ = decompose(x, 'phase')
        _, norm_amp = decompose(norm, 'amp')

        residual = compose(
            phase,
            norm_amp
        )
        residual = residual.permute(0, 2, 3, 1)
        
        residual = residual * self.weight + self.bias
        return residual



class SWIN_SC(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)
        features = []
        for c in range(len(self.model.features)):
            if c == 0:
                features.append(self.model.features[c])
            elif isinstance(self.model.features[c], nn.Sequential):
                features.append(self.model.features[c])
                features.append(SWIN_SCNorm(self.model.features[c][-1].norm1.weight.shape[0], False))
            else:
                self.model.features[c].norm = SWIN_CC(self.model.features[c].norm)
                features.append(self.model.features[c])
        self.model.features = nn.Sequential(*features)

    def forward(self, x):
        return self.model(x)


class SWIN_K(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)
        features = []
        for c in range(len(self.model.features)):
            if c == 0:
                features.append(self.model.features[c])
            elif isinstance(self.model.features[c], nn.Sequential):
                self.model.features[c][-1].norm2 = SWIN_CC(self.model.features[c][-1].norm2)
                # for m in self.model.features[c]:
                #     m.norm2 = SWIN_CC(m.norm2)
                features.append(self.model.features[c])
                features.append(SWIN_SCNorm(self.model.features[c][-1].norm1.weight.shape[0], False))
            else:
                # self.model.features[c].norm = SWIN_CC(self.model.features[c].norm)
                features.append(self.model.features[c])
        self.model.features = nn.Sequential(*features)

    def forward(self, x):
        return self.model(x)


class SWIN_P(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)
        features = []
        for c in range(len(self.model.features)):
            if c == 0:
                features.append(self.model.features[c])
            elif isinstance(self.model.features[c], nn.Sequential):
                features.append(self.model.features[c])
                features.append(SWIN_SCNorm(self.model.features[c][-1].norm1.weight.shape[0], False))
            else:
                self.model.features[c].norm = SWIN_PC(self.model.features[c].norm)
                features.append(self.model.features[c])
        self.model.features = nn.Sequential(*features)

    def forward(self, x):
        return self.model(x)


def swin(num_classes):
    model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model


def swin_sc(num_classes):
    # model = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
    # model.head = nn.Linear(model.head.in_features, num_classes)
    return SWIN_SC(num_classes)


def swin_p(num_classes):
    # model = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
    # model.head = nn.Linear(model.head.in_features, num_classes)
    return SWIN_P(num_classes)

def swin_k(num_classes):
    # model = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
    # model.head = nn.Linear(model.head.in_features, num_classes)
    return SWIN_K(num_classes)