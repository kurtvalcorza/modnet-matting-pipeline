"""MODNet architecture (Ke et al., 2022), vendored from `ZHKKKe/MODNet` at commit
`28165a451e4610c9d77cfdf925a94610bb2810fb` (Apache-2.0; `src/models/modnet.py` SHA-256 `2f26f5f0…`,
`src/models/backbones/mobilenetv2.py` `e3cc8ad6…`, `src/models/backbones/wrapper.py` `41197be7…`).

Parameter and buffer names are the upstream ones, so the pinned checkpoint loads strictly once its `module.`
(DataParallel) prefix is stripped. Differences from upstream, all deliberate: no `torch.load` of a backbone
checkpoint (`backbone_pretrained` is gone — the pinned matting checkpoint carries the backbone), no
`exit()` calls, no `nn.DataParallel`, `forward(img, inference)` unchanged in signature and semantics. Nothing
here imports outside `torch`, and the module is imported lazily by `pipeline.py` (fleet RTM-001).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# --------------------------------------------------------------------------------------------------
# MobileNetV2 backbone (upstream: adapted from thuyngch/Human-Segmentation-PyTorch)
# --------------------------------------------------------------------------------------------------


def _make_divisible(v: float, divisor: int, min_value: int | None = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:  # make sure that rounding down does not go down by more than 10 %
        new_v += divisor
    return new_v


def _conv_bn(inp: int, oup: int, stride: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))


def _conv_1x1_bn(inp: int, oup: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))


class InvertedResidual(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int, expansion: int, dilation: int = 1) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("stride must be 1 or 2")
        self.stride = stride
        hidden_dim = round(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup
        if expansion == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, dilation=dilation, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, dilation=dilation, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x) if self.use_res_connect else self.conv(x)


class MobileNetV2(nn.Module):
    """The feature extractor only (`num_classes=None` upstream): 19 stages under `features`."""

    def __init__(self, in_channels: int, alpha: float = 1.0, expansion: int = 6) -> None:
        super().__init__()
        self.in_channels = in_channels
        input_channel = _make_divisible(32 * alpha, 8)
        self.last_channel = _make_divisible(1280 * alpha, 8) if alpha > 1.0 else 1280
        settings = [  # t, c, n, s
            [1, 16, 1, 1],
            [expansion, 24, 2, 2],
            [expansion, 32, 3, 2],
            [expansion, 64, 4, 2],
            [expansion, 96, 3, 1],
            [expansion, 160, 3, 2],
            [expansion, 320, 1, 1],
        ]
        features: list[nn.Module] = [_conv_bn(self.in_channels, input_channel, 2)]
        for t, c, n, s in settings:
            output_channel = _make_divisible(int(c * alpha), 8)
            for i in range(n):
                features.append(InvertedResidual(input_channel, output_channel, s if i == 0 else 1, expansion=t))
                input_channel = output_channel
        features.append(_conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class MobileNetV2Backbone(nn.Module):
    """Upstream `MobileNetV2Backbone`: returns the five encoder scales (2×, 4×, 8×, 16×, 32×)."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.model = MobileNetV2(self.in_channels, alpha=1.0, expansion=6)
        self.enc_channels = [16, 24, 32, 96, 1280]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = self.model.features
        out = []
        for start, stop in ((0, 2), (2, 4), (4, 7), (7, 14), (14, 19)):
            for index in range(start, stop):
                x = features[index](x)
            out.append(x)
        return out


# --------------------------------------------------------------------------------------------------
# MODNet basic modules
# --------------------------------------------------------------------------------------------------


class IBNorm(nn.Module):
    """Instance norm and batch norm side by side over two halves of the channels."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.bnorm_channels = int(in_channels / 2)
        self.inorm_channels = in_channels - self.bnorm_channels
        self.bnorm = nn.BatchNorm2d(self.bnorm_channels, affine=True)
        self.inorm = nn.InstanceNorm2d(self.inorm_channels, affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bn_x = self.bnorm(x[:, : self.bnorm_channels, ...].contiguous())
        in_x = self.inorm(x[:, self.bnorm_channels :, ...].contiguous())
        return torch.cat((bn_x, in_x), 1)


class Conv2dIBNormRelu(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        with_ibn: bool = True,
        with_relu: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]
        if with_ibn:
            layers.append(IBNorm(out_channels))
        if with_relu:
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SEBlock(nn.Module):
    """Squeeze-and-excitation (Hu et al., 2018)."""

    def __init__(self, in_channels: int, out_channels: int, reduction: int = 1) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, int(in_channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels // reduction), out_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w.expand_as(x)


# --------------------------------------------------------------------------------------------------
# MODNet branches
# --------------------------------------------------------------------------------------------------


class LRBranch(nn.Module):
    """Low-resolution (semantic) branch."""

    def __init__(self, backbone: MobileNetV2Backbone) -> None:
        super().__init__()
        enc_channels = backbone.enc_channels
        self.backbone = backbone
        self.se_block = SEBlock(enc_channels[4], enc_channels[4], reduction=4)
        self.conv_lr16x = Conv2dIBNormRelu(enc_channels[4], enc_channels[3], 5, stride=1, padding=2)
        self.conv_lr8x = Conv2dIBNormRelu(enc_channels[3], enc_channels[2], 5, stride=1, padding=2)
        self.conv_lr = Conv2dIBNormRelu(enc_channels[2], 1, kernel_size=3, stride=2, padding=1, with_ibn=False, with_relu=False)

    def forward(self, img: torch.Tensor, inference: bool):
        enc_features = self.backbone(img)
        enc2x, enc4x, enc32x = enc_features[0], enc_features[1], enc_features[4]
        enc32x = self.se_block(enc32x)
        lr16x = F.interpolate(enc32x, scale_factor=2, mode="bilinear", align_corners=False)
        lr16x = self.conv_lr16x(lr16x)
        lr8x = F.interpolate(lr16x, scale_factor=2, mode="bilinear", align_corners=False)
        lr8x = self.conv_lr8x(lr8x)
        pred_semantic = None
        if not inference:
            pred_semantic = torch.sigmoid(self.conv_lr(lr8x))
        return pred_semantic, lr8x, [enc2x, enc4x]


class HRBranch(nn.Module):
    """High-resolution (detail) branch."""

    def __init__(self, hr_channels: int, enc_channels: list[int]) -> None:
        super().__init__()
        self.tohr_enc2x = Conv2dIBNormRelu(enc_channels[0], hr_channels, 1, stride=1, padding=0)
        self.conv_enc2x = Conv2dIBNormRelu(hr_channels + 3, hr_channels, 3, stride=2, padding=1)
        self.tohr_enc4x = Conv2dIBNormRelu(enc_channels[1], hr_channels, 1, stride=1, padding=0)
        self.conv_enc4x = Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1)
        self.conv_hr4x = nn.Sequential(
            Conv2dIBNormRelu(3 * hr_channels + 3, 2 * hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1),
        )
        self.conv_hr2x = nn.Sequential(
            Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, hr_channels, 3, stride=1, padding=1),
        )
        self.conv_hr = nn.Sequential(
            Conv2dIBNormRelu(hr_channels + 3, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, 1, kernel_size=1, stride=1, padding=0, with_ibn=False, with_relu=False),
        )

    def forward(self, img: torch.Tensor, enc2x: torch.Tensor, enc4x: torch.Tensor, lr8x: torch.Tensor, inference: bool):
        img2x = F.interpolate(img, scale_factor=1 / 2, mode="bilinear", align_corners=False)
        img4x = F.interpolate(img, scale_factor=1 / 4, mode="bilinear", align_corners=False)
        enc2x = self.tohr_enc2x(enc2x)
        hr4x = self.conv_enc2x(torch.cat((img2x, enc2x), dim=1))
        enc4x = self.tohr_enc4x(enc4x)
        hr4x = self.conv_enc4x(torch.cat((hr4x, enc4x), dim=1))
        lr4x = F.interpolate(lr8x, scale_factor=2, mode="bilinear", align_corners=False)
        hr4x = self.conv_hr4x(torch.cat((hr4x, lr4x, img4x), dim=1))
        hr2x = F.interpolate(hr4x, scale_factor=2, mode="bilinear", align_corners=False)
        hr2x = self.conv_hr2x(torch.cat((hr2x, enc2x), dim=1))
        pred_detail = None
        if not inference:
            hr = F.interpolate(hr2x, scale_factor=2, mode="bilinear", align_corners=False)
            hr = self.conv_hr(torch.cat((hr, img), dim=1))
            pred_detail = torch.sigmoid(hr)
        return pred_detail, hr2x


class FusionBranch(nn.Module):
    """Semantic-detail fusion branch: the alpha matte."""

    def __init__(self, hr_channels: int, enc_channels: list[int]) -> None:
        super().__init__()
        self.conv_lr4x = Conv2dIBNormRelu(enc_channels[2], hr_channels, 5, stride=1, padding=2)
        self.conv_f2x = Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1)
        self.conv_f = nn.Sequential(
            Conv2dIBNormRelu(hr_channels + 3, int(hr_channels / 2), 3, stride=1, padding=1),
            Conv2dIBNormRelu(int(hr_channels / 2), 1, 1, stride=1, padding=0, with_ibn=False, with_relu=False),
        )

    def forward(self, img: torch.Tensor, lr8x: torch.Tensor, hr2x: torch.Tensor) -> torch.Tensor:
        lr4x = F.interpolate(lr8x, scale_factor=2, mode="bilinear", align_corners=False)
        lr4x = self.conv_lr4x(lr4x)
        lr2x = F.interpolate(lr4x, scale_factor=2, mode="bilinear", align_corners=False)
        f2x = self.conv_f2x(torch.cat((lr2x, hr2x), dim=1))
        f = F.interpolate(f2x, scale_factor=2, mode="bilinear", align_corners=False)
        f = self.conv_f(torch.cat((f, img), dim=1))
        return torch.sigmoid(f)


class MODNet(nn.Module):
    """MODNet: a MobileNetV2 low-resolution branch, a high-resolution detail branch and a fusion branch.

    `forward(img, inference)` returns `(pred_semantic, pred_detail, pred_matte)`; with `inference=True` the first
    two are `None` (the heads that produce them are skipped, as upstream). `img` is (B, 3, H, W) normalised to
    [-1, 1] with H and W multiples of 32.
    """

    def __init__(self, in_channels: int = 3, hr_channels: int = 32) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hr_channels = hr_channels
        self.backbone = MobileNetV2Backbone(self.in_channels)
        self.lr_branch = LRBranch(self.backbone)
        self.hr_branch = HRBranch(self.hr_channels, self.backbone.enc_channels)
        self.f_branch = FusionBranch(self.hr_channels, self.backbone.enc_channels)

    def forward(self, img: torch.Tensor, inference: bool = True):
        pred_semantic, lr8x, (enc2x, enc4x) = self.lr_branch(img, inference)
        pred_detail, hr2x = self.hr_branch(img, enc2x, enc4x, lr8x, inference)
        pred_matte = self.f_branch(img, lr8x, hr2x)
        return pred_semantic, pred_detail, pred_matte

    def freeze_norm(self) -> None:
        """Put every BatchNorm / InstanceNorm layer in eval mode (upstream `freeze_norm`)."""
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d | nn.InstanceNorm2d):
                module.eval()
