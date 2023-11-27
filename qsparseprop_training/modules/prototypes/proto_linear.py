import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.prototypes.proto_functions import quantize, quantize_grad
from configurations import conf


class LinearQ(nn.Linear):
    # TODO: Implement quantized linear layer
    def __init__(self, in_features, out_features, bias=False, inputBits=8., weightBits=8., gradBits=8.,
                 fwdScheme='SAWB', bwdScheme='dithered', quantizeFwd=True, quantizeBwd=True, repeatBwd=1, scale=conf.DITHERED_8_BIT_QUANTIZATION_SCALE):
        super(LinearQ, self).__init__(in_features, out_features, bias)
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

            output = F.linear(qInput, qWeights, self.bias)
        else:
            output = F.linear(input, self.weight, self.bias)

        output = quantize_grad(output, self.quantizeBwd, self.layerIdx, self.repeatBwd, scheme=self.bwdScheme, scale=self.scale, bits=self.gradBits)
        return output