import torch
from utils.sparse_utils import to_sparse_format_conv2d, from_sparse_format_conv2d
from modules.sparseprop.functions import SparseConvFunction
from copy import deepcopy


class SparseConv2d(torch.nn.Module):
    def __init__(self, dense_weight, bias=None, padding=0, stride=1, vectorizing_over_on=False):
        super(SparseConv2d, self).__init__()

        self.OC, self.IC, self.K, _ = dense_weight.shape
        self.padding = padding
        self.stride = stride
        self.set_vectorizing_over_on(vectorizing_over_on)

        W_val, W_idx = to_sparse_format_conv2d(dense_weight)

        self.weight = torch.nn.Parameter(W_val)
        self.W_idx = W_idx

        assert bias is None or isinstance(bias, torch.nn.Parameter), f"bias is not a parameter but it's {type(bias)}"
        self.bias = bias

    @staticmethod
    def from_dense(conv, vectorizing_over_on=False):
        def bias_to_param():
            if conv.bias is None:
                return None
            return torch.nn.Parameter(conv.bias.data.clone())

        dense_weight = conv.weight.data
        stride = conv.stride[0]
        padding = conv.padding[0]

        return SparseConv2d(
            dense_weight,
            bias=bias_to_param(),
            padding=padding,
            stride=stride,
            vectorizing_over_on=vectorizing_over_on
        )

    def to_dense(self):
        dense_weight = from_sparse_format_conv2d(
            self.weight,
            self.W_idx,
            shape=(self.OC, self.IC, self.K, self.K)
        )

        conv = torch.nn.Conv2d(
            self.IC,
            self.OC,
            self.K,
            stride=self.stride,
            padding=self.padding,
            bias=self.bias is not None
        )

        with torch.no_grad():
            conv.weight.mul_(0)
            conv.weight.add_(dense_weight)

            if self.bias is not None:
                conv.bias.mul_(0)
                conv.bias.add_(self.bias)

        return conv

    def set_vectorizing_over_on(self, vectorizing_over_on):
        self.vectorizing_over_on = vectorizing_over_on
        self.vectorizing_bwd_over_on = vectorizing_over_on
        self.vectorizing_fwd_over_on = vectorizing_over_on and self.stride == 1  # stride 2 is not supported over on

    #@property
    #def weight(self):
    #    return self.W_val

    def forward(self, input):
        return SparseConvFunction.apply(input, self.weight, self.W_idx, self.bias, self.OC, self.K, self.padding,
                                        self.stride, self.vectorizing_fwd_over_on, self.vectorizing_bwd_over_on)

    def __repr__(self):
        nnz = len(self.weight)
        numel = self.OC * self.IC * self.K * self.K
        return f'SparseConv2d([{self.OC}, {self.IC}, {self.K}, {self.K}], sp={1. - nnz / numel:.2f}, nnz={nnz}, s={self.stride}, p={self.padding}, voo={self.vectorizing_over_on})'