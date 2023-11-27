import torch
from scipy.sparse import csr_matrix

from modules.sparseprop.functions import SparseLinearFunction
from utils.sparse_utils import to_csr_2d, from_csr_2d


class SparseLinear(torch.nn.Module):
    def __init__(self, dense_weight, bias=None):
        super(SparseLinear, self).__init__()
        self.N, self.M = dense_weight.shape

        W_val, W_idx = to_csr_2d(dense_weight)
        self.weight = torch.nn.Parameter(W_val)
        self.W_idx = W_idx

        assert bias is None or isinstance(bias, torch.nn.Parameter), f"bias is not a parameter but it's {type(bias)}"
        self.bias = bias

    @staticmethod
    def from_dense(module):
        return SparseLinear(
            dense_weight=module.weight.data,
            bias=None if module.bias is None else torch.nn.Parameter(module.bias.data.clone())
        )

    def to_dense(self):
        dense_weight = from_csr_2d(
            self.weight,
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
    #    return self.W_val

    def forward(self, input):
        return SparseLinearFunction.apply(input, self.weight, self.W_idx, self.bias, self.N)

    def __repr__(self):
        nnz = len(self.weight)
        numel = self.N * self.M
        return f"SparseLinear([{self.N}, {self.M}], sp={1. - nnz / numel:.2f}, nnz={nnz})"