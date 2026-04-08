# === model.py ===
#
# The segmentation model that converts an ECG row crop into
# a probability distribution over the vertical axis.
#
# Architecture: ResNet34 encoder + UNet decoder with CoordConv.
#
# How it works:
#   1. Encoder extracts features at 4 scales
#   2. Decoder upsamples back to full resolution with skip connections
#   3. CoordConv injects y/x coordinates so the model knows where it is
#   4. Soft-argmax head converts the feature map into a y-position
#      per column, then maps that to millivolts
#
# The soft-argmax head is the key difference from a basic UNet.
# Instead of predicting a binary mask and doing argmax post-processing,
# it predicts a probability distribution over y and computes the
# expected value. This is differentiable and gives sub-pixel accuracy.
#
# To change the backbone: edit CFG.backbone in config.py
# To change decoder size: edit CFG.decoder_dims in config.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from config import CFG


class CoordConvDecoderBlock(nn.Module):
    """
    One block of the UNet decoder.
    Upsamples 2x, concatenates with the skip connection from the encoder,
    then injects y/x coordinate channels so the model knows its
    spatial position. Two conv layers refine the features.
    """
    def __init__(self, in_ch, skip_ch, out_ch, scale=2):
        super().__init__()
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch + 2, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=self.scale, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)

        B, C, H, W = x.shape
        cy = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
        cx = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
        x = torch.cat([x, cy, cx], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SoftArgmaxHead(nn.Module):
    """
    Converts a feature map into millivolt predictions.

    For each pixel column, it predicts a probability distribution
    over the vertical axis (via softmax with temperature), then
    computes the expected y-position as a weighted sum.
    That y-position is converted to millivolts using the known
    baseline and scale factor.

    Temperature controls the sharpness of the distribution:
      - Low temperature (0.1) = sharp, confident predictions
      - High temperature (1.0) = spread out, uncertain predictions
      - We use 0.5 as a balanced default
    """
    def __init__(self, in_ch, n_leads=1, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.lead_logits = nn.Conv2d(in_ch, n_leads, kernel_size=1)

    def forward(self, feat, half_h, mv_to_pixel):
        """
        Args:
            feat: (B, C, H, W) decoder output
            half_h: half the crop height (baseline position in crop)
            mv_to_pixel: pixels per millivolt
        Returns:
            pred_mv: (B, 1, W) predicted millivolts per column
            prob: (B, 1, H, W) probability distribution for visualization
        """
        logits = self.lead_logits(feat)
        prob = torch.softmax(logits / self.temperature, dim=2)

        B, _, H, W = prob.shape
        y_coord = torch.arange(H, device=feat.device, dtype=feat.dtype).view(1, 1, H, 1)
        y_pixel = (prob * y_coord).sum(dim=2)  # (B, 1, W)

        pred_mv = (half_h - y_pixel) / mv_to_pixel
        return pred_mv, prob


def encode_with_resnet(encoder, x):
    """Extract multi-scale features from a ResNet encoder."""
    encode = []
    x = encoder.conv1(x)
    x = encoder.bn1(x)
    x = encoder.act1(x)
    x = encoder.layer1(x); encode.append(x)
    x = encoder.layer2(x); encode.append(x)
    x = encoder.layer3(x); encode.append(x)
    x = encoder.layer4(x); encode.append(x)
    return encode


class ECGRowNet(nn.Module):
    """
    Full model: takes a row crop image and predicts millivolts.

    Input:  (B, 3, 480, 5000) — a row crop of the rectified ECG
    Output: (B, 1, 5000) — millivolt values for each pixel column
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Encoder
        self.encoder = timm.create_model(
            cfg.backbone, pretrained=True,
            in_chans=3, num_classes=0, global_pool=''
        )
        enc_dims = [64, 128, 256, 512]  # resnet34
        dec_dims = cfg.decoder_dims

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        in_channels = [enc_dims[-1]] + dec_dims[:-1]
        skip_channels = enc_dims[:-1][::-1] + [0]
        for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, dec_dims):
            self.decoder_blocks.append(CoordConvDecoderBlock(in_ch, skip_ch, out_ch))

        # Soft-argmax regression head
        self.head = SoftArgmaxHead(dec_dims[-1], cfg.n_leads, cfg.temperature)

    def forward(self, batch):
        image = batch['image'].to(self.mean.device)

        # Normalize to ImageNet range
        x = image.float() / 255.0 if image.max() > 1 else image.float()
        x = (x - self.mean) / self.std

        # Encode
        encode = encode_with_resnet(self.encoder, x)

        # Decode
        d = encode[-1]
        skips = encode[:-1][::-1] + [None]
        for block, skip in zip(self.decoder_blocks, skips):
            d = block(d, skip)

        # Predict millivolts
        half_h = self.cfg.crop_h // 2
        pred_mv, pred_prob = self.head(d, half_h, self.cfg.mv_to_pixel)
        return pred_mv, pred_prob