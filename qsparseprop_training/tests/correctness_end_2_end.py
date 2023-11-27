from utils.network_utils import swap_modules_with_sparse, error

from sparseml.pytorch.models import ModelRegistry
import torch
from copy import deepcopy


if __name__ == '__main__':
    torch.manual_seed(11)

    B = 256  # batch size
    IC = 3  # input channels
    M = 32  # input height
    N = 32  # input width
    input_shape = (B, IC, M, N)
    num_classes = 10

    Y_orig = torch.randn(num_classes)

    X_orig = torch.randn(input_shape)
    X_orig.requires_grad_()
    X_orig.retain_grad()

    model_orig = ModelRegistry.create(
        key="resnet50",
        pretrained_path="zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned90-none",
        pretrained_dataset="cifar10",
        num_classes=num_classes,
        ignore_error_tensors=["classifier.fc.weight", "classifier.fc.bias"],
    )
    torch_X = X_orig.clone()
    torch_X.retain_grad()
    torch_Y = Y_orig.clone()
    torch_O, _ = model_orig(torch_X)
    torch.mean((torch_Y - torch_O) ** 2).backward()

    # TODO: Implement swap module helpers for replacing layers with prototype modules
    model_proto = deepcopy(model_orig)
    model_proto = swap_modules_with_sparse(model_proto, input_shape, inplace=True, verbose=True, prototype=True)
    proto_X = X_orig.clone()
    proto_X.retain_grad()
    proto_Y = Y_orig.clone()
    proto_O, _ = model_proto(proto_X)
    torch.mean((proto_Y - proto_O) ** 2).backward()

    model_actual_q = deepcopy(model_orig)
    model_actual_q = swap_modules_with_sparse(model_actual_q, input_shape, inplace=True, verbose=True, prototype=False)
    our_X = X_orig.clone()
    our_X.retain_grad()
    our_Y = Y_orig.clone()
    our_O, _ = model_actual_q(our_X)
    torch.mean((our_Y - our_O) ** 2).backward()

    print('[Proto vs. Our Forward]\n O error:', error(our_O, proto_O))
    print('[Proto vs. Torch Forward]\n O error:', error(proto_O, torch_O))
    print('[Our vs. Torch Forward]\n O error:', error(our_O, torch_O))

    """
    print(torch_O.size(), proto_O.size(), our_O.size())

    count = 0
    for tup in zip(torch_O.reshape(-1), our_O.reshape(-1)):
        print(tup)
        count += 1
        if count == 1000:
            break
    """
