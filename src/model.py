"""FlexScaleSR - a compact residual super-resolution network with a flexible output size.

Why "FlexScale"
---------------
The HR targets are not all the same size (307..336 px) and the scale factor
between LR and HR is not a whole number (it is about 6.4x to 7.0x). A normal
PixelShuffle network can only multiply the size by a fixed integer, so it cannot
hit those sizes exactly. FlexScaleSR solves this in two steps:

    1. learned upsampling by a fixed integer factor (default 8x) with PixelShuffle
    2. a plain resize to the EXACT target height and width

Because 8 is larger than the biggest real scale factor (7.0), step 2 always
shrinks the image slightly. Shrinking is safer than stretching: it never has to
invent pixels that the learned upsampler did not produce.

Data flow
---------
    LR (4, 48, 48)
        |
        v  head: Conv 3x3
    features (64, 48, 48)
        |
        v  N residual blocks (Conv - ReLU - Conv + skip)
        v  fusion Conv 3x3 + long skip connection
    features (64, 48, 48)
        |
        v  3 x [Conv -> PixelShuffle(2) -> ReLU]      48 -> 96 -> 192 -> 384
    features (64, 384, 384)
        |
        v  bilinear resize to the exact (H, W) asked for
    features (64, H, W)
        |
        v  reconstruction: Conv 3x3 -> ReLU -> Conv 3x3
    residual (4, H, W)
        |
        +--- bicubic upsample of the input LR  (global skip)
        v
    prediction (4, H, W)

The global skip means the network only has to predict the DIFFERENCE between a
bicubic upsample and the true HR image. That is an easier job than predicting
the whole image, it trains faster, and it starts out already as good as bicubic.

Input and output are always 4 channels in canonical order [R, G, B, NIR].
NIR is a normal output channel; it is never dropped or converted to RGB.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Conv - ReLU - Conv, then add the input back (a standard residual block)."""

    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv2(self.relu(self.conv1(x)))
        return x + out


class FlexScaleSR(nn.Module):
    """Compact residual SR network whose output size is chosen at call time.

    Parameters
    ----------
    in_channels, out_channels : 4 and 4 (R, G, B, NIR)
    features                  : width of the network (64 is a good default)
    num_res_blocks            : depth of the network
    upsample_factor           : integer PixelShuffle factor; must be a power of 2
                                and should be >= the largest real scale factor
    """

    def __init__(self, in_channels=4, out_channels=4, features=64,
                 num_res_blocks=12, upsample_factor=8, max_reflectance=1.5):
        super().__init__()
        assert upsample_factor in (2, 4, 8, 16), "upsample_factor must be 2, 4, 8 or 16"
        self.upsample_factor = upsample_factor
        self.max_reflectance = max_reflectance

        # 1. head
        self.head = nn.Conv2d(in_channels, features, 3, padding=1)

        # 2. residual body
        self.body = nn.Sequential(*[ResidualBlock(features) for _ in range(num_res_blocks)])

        # 3. feature fusion (used with a long skip around the whole body)
        self.fusion = nn.Conv2d(features, features, 3, padding=1)

        # 4. learned upsampling: one PixelShuffle(2) stage per doubling
        stages = []
        n_stages = int(torch.log2(torch.tensor(float(upsample_factor))).item())
        for _ in range(n_stages):
            stages += [nn.Conv2d(features, features * 4, 3, padding=1),
                       nn.PixelShuffle(2),
                       nn.ReLU(inplace=True)]
        self.upsample = nn.Sequential(*stages)

        # 5. reconstruction
        self.reconstruct = nn.Sequential(
            nn.Conv2d(features, features // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features // 2, out_channels, 3, padding=1),
        )

    def forward(self, lr, target_hw):
        """Super-resolve `lr` to exactly `target_hw`.

        lr        : tensor (B, 4, h, w)
        target_hw : (H, W) taken from the matching HR image
        returns   : tensor (B, 4, H, W)
        """
        H, W = int(target_hw[0]), int(target_hw[1])

        x = self.head(lr)
        x = x + self.fusion(self.body(x))        # long skip around the residual body
        x = self.upsample(x)                     # integer learned upsampling
        # flexible step: land on the EXACT requested size
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        residual = self.reconstruct(x)

        # global skip: bicubic upsample of the input, so we only learn the detail
        base = F.interpolate(lr, size=(H, W), mode="bicubic", align_corners=False)
        out = base + residual

        # Physical range guard, applied in EVAL MODE ONLY.
        #
        # Reflectance can never be negative, and on this dataset only 302 of
        # 57.5 million real HR pixels exceed 1.5 (0.0005 percent), so [0, 1.5]
        # is a safe physical bound that still catches a runaway prediction.
        #
        # It is deliberately NOT applied during training. Clamping would set the
        # gradient to zero for exactly the pixels the model got most wrong, so
        # it could never learn its way back into range. Clamping only at
        # evaluation time is a projection onto the feasible set: against
        # in-range targets it can only reduce the error, never increase it.
        if not self.training:
            out = out.clamp(0.0, self.max_reflectance)
        return out


def build_model(cfg):
    """Create a FlexScaleSR from the `model` section of the config dict."""
    m = cfg["model"]
    return FlexScaleSR(
        in_channels=m.get("in_channels", 4),
        out_channels=m.get("out_channels", 4),
        features=m.get("features", 64),
        num_res_blocks=m.get("num_res_blocks", 12),
        upsample_factor=m.get("upsample_factor", 8),
        max_reflectance=m.get("max_reflectance", 1.5),
    )


def count_parameters(model):
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
