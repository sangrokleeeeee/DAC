from torchvision.models import swin_b, Swin_B_Weights, mobilenet_v2, MobileNet_V2_Weights
from torchvision.ops.misc import Conv2dNormActivation
from torch import nn
import torch
import torch.nn.functional as F
from typing import Any, Callable, List, Optional
from torch import nn, Tensor

from model import decompose, compose, SCNorm, ccnorm, pcnorm


class InvertedResidualN(nn.Module):
    def __init__(
        self, inp: int, oup: int, stride: int, expand_ratio: int, norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        if not self.use_res_connect:
            self.down = nn.Sequential(
                nn.Conv2d(inp, oup, 1, stride=stride, padding=0, bias=False),
                norm_layer(oup),
            )
            self.weight = nn.Parameter(torch.zeros(2))

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                Conv2dNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6)
            )
        layers.extend(
            [
                # dw
                Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                    activation_layer=nn.ReLU6,
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup) if self.use_res_connect else norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            x = self.conv[:-1](x)
            return ccnorm(x, self.conv[-1], self.weight)
            # down = self.down[0](x)
            # return ccnorm(down, self.down[1], self.weight) + self.conv(x)


class InvertedResidualCC(nn.Module):
    def __init__(
        self, inp: int, oup: int, stride: int, expand_ratio: int, norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        if not self.use_res_connect:
            self.down = nn.Sequential(
                nn.Conv2d(inp, oup, 1, stride=stride, padding=0, bias=False),
                norm_layer(oup),
            )
            self.weight = nn.Parameter(torch.zeros(2))

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                Conv2dNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6)
            )
        layers.extend(
            [
                # dw
                Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                    activation_layer=nn.ReLU6,
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            down = self.down[0](x)
            return ccnorm(down, self.down[1], self.weight) + self.conv(x)


class InvertedResidualP(nn.Module):
    def __init__(
        self, inp: int, oup: int, stride: int, expand_ratio: int, norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        if not self.use_res_connect:
            self.down = nn.Sequential(
                nn.Conv2d(inp, oup, 1, stride=stride, padding=0, bias=False),
                norm_layer(oup),
            )

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                Conv2dNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6)
            )
        layers.extend(
            [
                # dw
                Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                    activation_layer=nn.ReLU6,
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            down = self.down[0](x)
            return pcnorm(down, self.down[1]) + self.conv(x)


class MobileNet_SC(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        dict_ = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        self.model = mobilenet_v2(block=InvertedResidualCC)
        self.model.load_state_dict(dict_.state_dict(), strict=False)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        self.norms = []
        self.in_ = self.model.features[0]
        # 1, 2, 4, 6
        self.stage1 = self.model.features[1:4]
        self.stage2 = self.model.features[4:7]
        self.stage3 = self.model.features[7:14]
        self.stage4 = self.model.features[14:]
        self.norms = nn.ModuleList([
            SCNorm(self.stage1[-1].out_channels),
            SCNorm(self.stage2[-1].out_channels),
            SCNorm(self.stage3[-1].out_channels),
        ])

    def forward(self, x):
        x = self.in_(x)
        x = self.stage1(x)
        x = self.norms[0](x)
        x = self.stage2(x)
        x = self.norms[1](x)
        x = self.stage3(x)
        x = self.norms[2](x)
        x = self.stage4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.model.classifier(x)
        return x


class MobileNet_P(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        dict_ = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        self.model = mobilenet_v2(block=InvertedResidualP)
        self.model.load_state_dict(dict_.state_dict(), strict=False)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        self.norms = []
        self.in_ = self.model.features[0]
        # 1, 2, 4, 6
        self.stage1 = self.model.features[1:4]
        self.stage2 = self.model.features[4:7]
        self.stage3 = self.model.features[7:14]
        self.stage4 = self.model.features[14:]
        self.norms = nn.ModuleList([
            SCNorm(self.stage1[-1].out_channels),
            SCNorm(self.stage2[-1].out_channels),
            SCNorm(self.stage3[-1].out_channels),
        ])

    def forward(self, x):
        x = self.in_(x)
        x = self.stage1(x)
        x = self.norms[0](x)
        x = self.stage2(x)
        x = self.norms[1](x)
        x = self.stage3(x)
        x = self.norms[2](x)
        x = self.stage4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.model.classifier(x)
        return x


class MobileNet_N(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        dict_ = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        self.model = mobilenet_v2(block=InvertedResidualN)
        self.model.load_state_dict(dict_.state_dict(), strict=False)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        self.norms = []
        self.in_ = self.model.features[0]
        # 1, 2, 4, 6
        self.stage1 = self.model.features[1:4]
        self.stage2 = self.model.features[4:7]
        self.stage3 = self.model.features[7:14]
        self.stage4 = self.model.features[14:]
        self.norms = nn.ModuleList([
            SCNorm(self.stage1[-1].out_channels),
            SCNorm(self.stage2[-1].out_channels),
            SCNorm(self.stage3[-1].out_channels),
        ])

    def forward(self, x):
        x = self.in_(x)
        x = self.stage1(x)
        x = self.norms[0](x)
        x = self.stage2(x)
        x = self.norms[1](x)
        x = self.stage3(x)
        x = self.norms[2](x)
        x = self.stage4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.model.classifier(x)
        return x


def mobilenet(num_classes):
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def mobilenet_sc(num_classes):
    model = MobileNet_SC(num_classes)
    return model


def mobilenet_p(num_classes):
    model = MobileNet_P(num_classes)
    return model


def mobilenet_n(num_classes):
    model = MobileNet_N(num_classes)
    return model