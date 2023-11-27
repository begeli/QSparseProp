import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.prototypes.proto_functions import quantize, quantize_grad
from configurations import conf


class Conv2dQ(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=False, inputBits=8., weightBits=8., gradBits=8.,
                 fwdScheme='SAWB', bwdScheme='dithered', quantizeFwd=True, quantizeBwd=True, repeatBwd=1, scale=conf.DITHERED_8_BIT_QUANTIZATION_SCALE):
        super(Conv2dQ, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias
        )
        self.layerIdx = 0
        self.quantizeFwd = quantizeFwd
        self.quantizeBwd = quantizeBwd
        self.repeatBwd = repeatBwd  # Hardcoded - In the paper "2" gives much better results

        self.inputBits = inputBits
        self.weightBits = weightBits
        self.gradBits = gradBits
        self.fwdScheme = fwdScheme
        self.bwdScheme = bwdScheme
        self.scale = scale

    def forward(self, input):
        if self.quantizeFwd:
            qWeights = quantize(self.weight, self.weightBits, signed=True, scheme=self.fwdScheme)
            qInput = quantize(input, self.inputBits, signed=torch.min(input) < 0, scheme=self.fwdScheme)

            output = F.conv2d(
                qInput,
                qWeights,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups
            )
        else:
            output = F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups
            )

        output = quantize_grad(
            output,
            self.quantizeBwd,
            self.layerIdx,
            self.repeatBwd,
            scheme=self.bwdScheme,
            scale=self.scale,
            bits=self.gradBits
        )
        return output