import os

from modules.conv2d import QuantizedSparseConv2d
from modules.linear import QuantizedSparseLinear
from modules.sparseprop.conv2d import SparseConv2d
from modules.sparseprop.linear import SparseLinear
from modules.prototypes.proto_conv2d import Conv2dQ
from modules.prototypes.proto_linear import LinearQ
from configurations import conf

import torch
import time
from copy import deepcopy


@torch.enable_grad()
def run_and_choose(modules, input_shape, verbose=False):
    if len(modules) == 1:
        if verbose:
            print('only one option...')
        return modules[0]

    X_orig = torch.randn(*input_shape)
    Y_orig = None

    min_time = float('inf')
    best_module = None
    for module in modules:
        module_copy = deepcopy(module)
        X = X_orig.clone()
        X.requires_grad_()
        X.retain_grad()

        temp = time.time()
        O = module_copy(X)
        fwd_time = time.time() - temp

        if Y_orig is None:
            Y_orig = torch.randn_like(O)
        Y = Y_orig.clone()

        L = torch.mean((O - Y) ** 2)
        temp = time.time()
        L.backward()
        bwd_time = time.time() - temp

        if verbose:
            print(f'module {module} took {fwd_time} fwd and {bwd_time} bwd')

        full_time = fwd_time + bwd_time
        if full_time < min_time:
            min_time = full_time
            best_module = module

    if verbose:
        print(f'going with {best_module} with full time of {min_time}')
    return best_module


def _quantize_and_sparsify_if_faster_linear(module, input_shape, include_dense, verbose, prototype, quantize):
    if quantize:
        if prototype:
            sp = LinearQ(
                module.in_features, module.out_features, module.bias is not None,
                8, 8, 8, 'SAWB', 'dithered', True, True, 1, conf.DITHERED_8_BIT_QUANTIZATION_SCALE
            )
            with torch.no_grad():
                sp.weight.mul_(0.)
                sp.weight.add_(module.weight.data)
                sp.bias = None if module.bias is None else torch.nn.Parameter(module.bias.data.clone())
        else:
            sp = QuantizedSparseLinear(
                dense_weight=module.weight.data,
                bias=None if module.bias is None else torch.nn.Parameter(module.bias.data.clone())
            )
    else:
        sp = SparseLinear(
            dense_weight=module.weight.data,
            bias=None if module.bias is None else torch.nn.Parameter(module.bias.data.clone())
        )

    if not include_dense:
        return sp

    assert input_shape is not None
    return run_and_choose([module, sp], input_shape, verbose=verbose)


def _quantize_and_sparsify_if_faster_conv2d(conv, input_shape, include_dense, verbose, prototype, quantize):
    def bias_to_param():
        if conv.bias is None:
            return None
        return torch.nn.Parameter(conv.bias.data.clone())

    dense_weight = conv.weight.data
    stride = conv.stride[0]
    padding = conv.padding[0]

    if quantize:
        if prototype:
            qsp1 = Conv2dQ(
                conv.in_channels, conv.out_channels, conv.kernel_size, stride,
                padding, conv.dilation, conv.groups, conv.bias is not None,
                8, 8, 8, 'SAWB', 'dithered', True, True, 1, conf.DITHERED_8_BIT_QUANTIZATION_SCALE
            )
            with torch.no_grad():
                qsp1.weight.mul_(0.)
                qsp1.weight.add_(conv.weight.data)
                qsp1.bias = None if conv.bias is None else torch.nn.Parameter(conv.bias.data.clone())
        else:
            qsp1 = QuantizedSparseConv2d(
                dense_weight,
                bias=bias_to_param(),
                padding=padding,
                stride=stride,
                vectorizing_over_on=False
            )
    else:
        sp1 = SparseConv2d(
            dense_weight,
            bias=bias_to_param(),
            padding=padding,
            stride=stride,
            vectorizing_over_on=False
        )

        sp2 = SparseConv2d(
            dense_weight,
            bias=bias_to_param(),
            padding=padding,
            stride=stride,
            vectorizing_over_on=True
        )

    modules = []
    if include_dense:
        modules.append(conv)
    if quantize:
        modules += [qsp1]
    else:
        modules += [sp1, sp2]

    return run_and_choose(modules, input_shape, verbose=verbose)


def sparsify_if_faster(module, input_shape, include_dense=True, verbose=False, prototype=False, quantize=False):
    if isinstance(module, torch.nn.Linear):
        return _quantize_and_sparsify_if_faster_linear(module, input_shape, include_dense, verbose, prototype, quantize)
    else:
        assert isinstance(module, torch.nn.Conv2d)
        return _quantize_and_sparsify_if_faster_conv2d(module, input_shape, include_dense, verbose, prototype, quantize)


def sparsify_conv2d_auto(conv, input_shape, verbose=False):
    return _quantize_and_sparsify_if_faster_conv2d(conv, input_shape, include_dense=False, verbose=verbose)


def set_num_threads(num_threads):
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
