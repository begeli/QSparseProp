from modules import QuantizedSparseLinear
from modules.prototypes.proto_linear import LinearQ
import torch
from utils.network_utils import error
import math
from copy import deepcopy
from configurations import conf

if __name__ == '__main__':
    torch.manual_seed(11)

    B = 128 # batch size
    M = 512 # input height
    N = 256 # input width
    sparsity = .9

    W = torch.randn(N, M)
    bias = torch.randn(N)
    mask = torch.rand_like(W) > sparsity
    W *= mask

    Y_orig = torch.randn(B, N)

    X_orig = torch.randn(B, M)
    X_orig.requires_grad_()
    X_orig.retain_grad()

    torch_X = X_orig.clone()
    torch_X.retain_grad()
    torch_Y = Y_orig.clone()
    linear = torch.nn.Linear(M, N, bias=True)
    with torch.no_grad():
        linear.weight.mul_(0.)
        linear.weight.add_(W)
        linear.bias.mul_(0.)
        linear.bias.add_(bias)
    torch_O = linear(torch_X)
    torch.mean((torch_O - torch_Y) ** 2).backward()
    torch_X_grad = torch_X.grad
    torch_W_grad = linear.weight.grad[linear.weight != 0]

    proto_X = X_orig.clone()
    proto_X.retain_grad()
    proto_Y = Y_orig.clone()
    linearq = LinearQ(
        linear.in_features, linear.out_features, True, 8, 8, 8,
        'SAWB', 'dithered', True, True, 1, conf.DITHERED_8_BIT_QUANTIZATION_SCALE
    )
    with torch.no_grad():
        linearq.weight.mul_(0.)
        linearq.weight.add_(W)
        linearq.bias.mul_(0.)
        linearq.bias.add_(bias)
    proto_O = linearq(proto_X)
    torch.mean((proto_O - proto_Y) ** 2).backward()
    proto_X_grad = proto_X.grad
    proto_W_grad = linearq.weight.grad[linearq.weight != 0]

    our_X = X_orig.clone()
    our_X.retain_grad()
    our_Y = Y_orig.clone()
    splinear = QuantizedSparseLinear(W, bias=torch.nn.Parameter(deepcopy(bias)))
    our_O = splinear(our_X)
    torch.mean((our_O - our_Y) ** 2).backward()
    our_X_grad = our_X.grad
    our_W_grad = splinear.weight.grad

    print('[QForward]\n O error:', error(our_O, proto_O))
    print('[QBackward]\n X grad error:', error(our_X_grad, proto_X_grad), '\n W grad error:',
          error(our_W_grad, proto_W_grad))
    print('[QBackward]\n bias grad error:', error(splinear.bias.grad, linearq.bias.grad))

    print('[Proto Forward]\n O error:', error(proto_O, torch_O))
    print('[Proto Backward]\n X grad error:', error(proto_X_grad, torch_X_grad), '\n W grad error:',
          error(proto_W_grad, torch_W_grad))
    print('[Proto Backward]\n bias grad error:', error(linearq.bias.grad, linear.bias.grad))

    #print(torch.unique(our_X_grad, return_counts=True))
    #print(torch.unique(proto_X_grad, return_counts=True))

    """
    count = 0
    for tup in zip(torch_W_grad.reshape(-1), proto_W_grad.reshape(-1)):
        print(tup)
        count += 1
        if count == 1000:
            break
    """