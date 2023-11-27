from modules import QuantizedSparseConv2d
from modules.prototypes.proto_conv2d import Conv2dQ
import torch
from utils.network_utils import error
import math
from copy import deepcopy
from configurations import conf

if __name__ == '__main__':
    torch.manual_seed(11)

    B = 256  # batch size
    IC = 512  # input channels
    OC = 512  # output channels
    M = 7  # input height
    N = 7  # input width
    K = 3  # kernel size
    stride = 1  # stride
    padding = 0  # padding
    vectorizing_over_on = False  # as described in the paper
    sparsity = .9  # sparsity of the weights

    OM = math.ceil((M + 2 * padding - K + 1) / stride)
    ON = math.ceil((N + 2 * padding - K + 1) / stride)

    W = torch.randn(OC, IC, K, K)
    bias = torch.randn(OC)
    mask = torch.rand_like(W) > sparsity
    W *= mask

    Y_orig = torch.randn(B, OC, OM, ON)

    X_orig = torch.randn(B, IC, M, N)
    X_orig.requires_grad_()
    X_orig.retain_grad()

    torch_X = X_orig.clone()
    torch_X.retain_grad()
    torch_Y = Y_orig.clone()
    conv = torch.nn.Conv2d(IC, OC, K, stride=stride, padding=padding, bias=True)
    with torch.no_grad():
        conv.weight.mul_(0.)
        conv.weight.add_(W)
        conv.bias.mul_(0.)
        conv.bias.add_(bias)
    torch_O = conv(torch_X)
    torch.mean((torch_O - torch_Y) ** 2).backward()
    torch_X_grad = torch_X.grad
    torch_W_grad = conv.weight.grad[conv.weight != 0]

    proto_X = X_orig.clone()
    proto_X.retain_grad()
    proto_Y = Y_orig.clone()
    conv2dq = Conv2dQ(
        conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
        conv.padding, conv.dilation, conv.groups, True, 8, 8, 8, 'SAWB', 'dithered',
        True, True, 1, conf.DITHERED_8_BIT_QUANTIZATION_SCALE,
    )
    with torch.no_grad():
        conv2dq.weight.mul_(0.)
        conv2dq.weight.add_(W)
        conv2dq.bias.mul_(0.)
        conv2dq.bias.add_(bias)
    proto_O = conv2dq(proto_X)
    torch.mean((proto_O - proto_Y) ** 2).backward()
    proto_X_grad = proto_X.grad
    proto_W_grad = conv2dq.weight.grad[conv2dq.weight != 0]

    our_X = X_orig.clone()
    our_X.retain_grad()
    our_Y = Y_orig.clone()
    spconv = QuantizedSparseConv2d(W, bias=torch.nn.Parameter(deepcopy(bias)), padding=padding, stride=stride)
    our_O = spconv(our_X)
    torch.mean((our_O - our_Y) ** 2).backward()
    our_X_grad = our_X.grad
    our_W_grad = spconv.weight.grad

    print('[QForward]\n O error:', error(our_O, proto_O))
    print('[QBackward]\n X grad error:', error(our_X_grad, proto_X_grad), '\n W grad error:', error(our_W_grad, proto_W_grad))
    print('[QBackward]\n bias grad error:', error(spconv.bias.grad, conv2dq.bias.grad))

    print('[Proto Forward]\n O error:', error(proto_O, torch_O))
    print('[Proto Backward]\n X grad error:', error(proto_X_grad, torch_X_grad), '\n W grad error:',
          error(our_W_grad, torch_W_grad))
    print('[Proto Backward]\n bias grad error:', error(conv2dq.bias.grad, conv.bias.grad))

    """
    count = 0
    for tup in zip(torch_X_grad.reshape(-1), proto_X_grad.reshape(-1)):
        print(tup)
        count += 1
        if count == 1000:
            break
    
    print(torch.unique(our_X_grad, return_counts=True))
    print(torch.unique(proto_X_grad, return_counts=True))
    print(torch.unique(torch_X_grad, return_counts=True))
    print(torch.unique(torch.abs(torch_X_grad - proto_X_grad), return_counts=True))
    """
    """
    count = 0
    for tup in zip(our_X_grad.reshape(-1), proto_X_grad.reshape(-1)):
        print(tup)
        count += 1
        if count == 1000:
            break
    """
