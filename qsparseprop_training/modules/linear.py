import torch
from utils.sparse_utils import to_csr_2d, from_csr_2d
from utils.quantization_utils import quantize
from configurations import conf
from modules.functions import QuantizedSparseLinearFunction


class QuantizedSparseLinear(torch.nn.Module):
    def __init__(self, dense_weight, bias=None):
        super(QuantizedSparseLinear, self).__init__()
        self.N, self.M = dense_weight.shape

        W_val, W_idx = to_csr_2d(dense_weight)

        self.weight = torch.nn.Parameter(W_val)
        self.quantize_weights()
        self.W_idx = W_idx

        assert bias is None or isinstance(bias, torch.nn.Parameter), f"bias is not a parameter but it's {type(bias)}"
        self.bias = bias

    @staticmethod
    def from_dense(module):
        return QuantizedSparseLinear(
            dense_weight=module.weight.data,
            bias=None if module.bias is None else torch.nn.Parameter(module.bias.data.clone())
        )

    def to_dense(self):
        dense_weight = from_csr_2d(
            self.weight,#self.W_val_i,
            self.W_idx,
            shape=(self.N, self.M)
        )

        linear = torch.nn.Linear(
            self.M,
            self.N,
            bias=self.bias is not None
        )

        with torch.no_grad():
            linear.weight.mul_(0)
            linear.weight.add_(dense_weight)

            if self.bias is not None:
                linear.bias.mul_(0)
                linear.bias.add_(self.bias)

        return linear

    #@property
    #def weight(self):
    #    return self.weight  #return self.W_val_i

    def quantize_weights(self):
        self.W_val_i, self.W_val_scales = quantize(self.weight.data, 'sawb', conf.PARALLEL, conf.GROUPED)

    def forward(self, input):
        return QuantizedSparseLinearFunction.apply(
            input, self.weight, self.W_idx, self.bias, self.N, self.W_val_i, self.W_val_scales
        )
