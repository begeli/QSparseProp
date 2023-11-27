from modules import QuantizedSparseConv2d
import torch
from utils.network_utils import error
import math
from copy import deepcopy

if __name__ == '__main__':
    torch.manual_seed(11)

    B = 64  # batch size
    IC = 128  # input channels
    OC = 128  # output channels
    M = 8  # input height
    N = 8  # input width
    K = 3  # kernel size
    stride = 2  # stride
    padding = 0  # padding
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

    # QSparse over ON
    over_on_X = X_orig.clone()
    over_on_X.retain_grad()
    over_on_Y = Y_orig.clone()
    over_on_spconv = QuantizedSparseConv2d(W, bias=torch.nn.Parameter(deepcopy(bias)), padding=padding, stride=stride, vectorizing_over_on=True)
    over_on_O = over_on_spconv(over_on_X)
    torch.mean((over_on_O - over_on_Y) ** 2).backward()
    over_on_X_grad = over_on_X.grad
    over_on_W_grad = over_on_spconv.weight.grad

    # QSparse not over ON
    our_X = X_orig.clone()
    our_X.retain_grad()
    our_Y = Y_orig.clone()
    spconv = QuantizedSparseConv2d(W, bias=torch.nn.Parameter(deepcopy(bias)), padding=padding, stride=stride, vectorizing_over_on=False)
    our_O = spconv(our_X)
    torch.mean((our_O - our_Y) ** 2).backward()
    our_X_grad = our_X.grad
    our_W_grad = spconv.weight.grad

    print('[Forward]\n O error:', error(our_O, over_on_O))
    print('[Backward]\n X grad error:', error(our_X_grad, over_on_X_grad), '\n W grad error:',
          error(our_W_grad, over_on_W_grad))
    print('[Backward]\n bias grad error:', error(spconv.bias.grad, over_on_spconv.bias.grad))

    """
    from sparseml.pytorch.models import resnet50, resnet18, mobilenet_v2
    from utils.network_utils import generate_intermediate_shapes
    model = mobilenet_v2(num_classes=100)
    input_shape = (3, 32, 32)
    intermediate_shapes, _ = generate_intermediate_shapes(model, input_shape)
    print(intermediate_shapes)
    
    count = 0
    for tup in zip(our_W_grad.reshape(-1), over_on_W_grad.reshape(-1)):
        print(tup)
        count += 1
        if count == 1000:
            break
    """