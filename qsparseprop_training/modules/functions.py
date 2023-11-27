import qsparseprop_backend as qsppb
import torch
import math
from utils.function_utils import linear_forward, linear_backward, conv2d_forward, conv2d_backward, conv2d_forward_over_ON, conv2d_backward_over_ON
from utils.quantization_utils import quantize
from configurations import conf

import torch.nn.functional as F
from utils.sparse_utils import from_sparse_format_conv2d

TRANSPOSE_BLOCK_SIZE = 16


class QuantizedSparseLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputT, W_val, W_idx, bias, N, W_val_i, W_val_scales):
        input_flat_t = inputT.reshape(-1, inputT.shape[-1]).contiguous()
        B, M = input_flat_t.shape
        if B % TRANSPOSE_BLOCK_SIZE == 0 and M % TRANSPOSE_BLOCK_SIZE == 0:
            input_flat = torch.zeros(M, B)
            qsppb.transpose(input_flat_t, input_flat, TRANSPOSE_BLOCK_SIZE)
        else:
            input_flat = input_flat_t.t().contiguous()
        ctx.inputT_shape = inputT.shape

        M, B = input_flat.shape
        W_idx_N, W_idx_M = W_idx

        q_input_flat, q_dq_info = quantize(input_flat, 'sawb', conf.PARALLEL, conf.GROUPED)
        output = torch.zeros(N, B).float()
        linear_forward(
            q_input_flat, q_dq_info, W_val_i, W_idx_N, W_idx_M, W_val_scales,
            output, conf.PARALLEL, conf.GROUPED
        )

        ctx.save_for_backward(W_val_i, bias)
        ctx.svd = (q_input_flat, q_dq_info, W_idx_N, W_idx_M, W_val_scales)

        if bias is not None:
            output += bias.view(-1, 1)

        if B % TRANSPOSE_BLOCK_SIZE == 0 and N % TRANSPOSE_BLOCK_SIZE == 0:
            output_t = torch.zeros(B, N)
            qsppb.transpose(output, output_t, TRANSPOSE_BLOCK_SIZE)
        else:
            output_t = output.t()  # (B, N)
        output_t = output_t.reshape(*ctx.inputT_shape[:-1], N)
        return output_t

    @staticmethod
    def backward(ctx, grad_output_t):
        W_val_i, bias = ctx.saved_tensors
        q_input_flat, q_input_dq_info, W_idx_N, W_idx_M, W_val_scales = ctx.svd

        grad_output_t = grad_output_t.reshape(-1, grad_output_t.shape[-1]).contiguous()
        B, N = grad_output_t.shape
        if B % TRANSPOSE_BLOCK_SIZE == 0 and N % TRANSPOSE_BLOCK_SIZE == 0:
            grad_output = torch.zeros(N, B)
            qsppb.transpose(grad_output_t, grad_output, TRANSPOSE_BLOCK_SIZE)
        else:
            grad_output = grad_output_t.t().contiguous()

        q_grad_output, q_grad_output_dq_info = quantize(
            grad_output, 'dithered', conf.PARALLEL, conf.GROUPED
        )
        grad_input = torch.zeros_like(q_input_flat).float().contiguous()  # (M, B)
        grad_W_val = torch.zeros_like(W_val_i).float().contiguous()

        linear_backward(
            q_input_flat, q_input_dq_info, W_val_i, W_idx_N, W_idx_M,
            W_val_scales, q_grad_output, q_grad_output_dq_info,
            grad_input, grad_W_val, conf.PARALLEL, conf.GROUPED
        )

        M = q_input_flat.shape[0]
        if B % TRANSPOSE_BLOCK_SIZE == 0 and M % TRANSPOSE_BLOCK_SIZE == 0:
            grad_input_t = torch.zeros(B, M)
            qsppb.transpose(grad_input, grad_input_t, TRANSPOSE_BLOCK_SIZE)
        else:
            grad_input_t = grad_input.t()  # (B, M)
        grad_input_t = grad_input_t.reshape(ctx.inputT_shape)

        grad_bias = None
        if bias is not None:
            grad_bias = grad_output_t.sum([i for i in range(len(grad_output_t.shape) - 1)])

        return grad_input_t, grad_W_val, None, grad_bias, None, None, None  # Need to return one more value for W_scale


class QuantizedSparseConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, W_val, W_idx, bias, OC, K, padding, stride, W_val_i, W_val_scales, vectorizing_fwd_over_on, vectorizing_bwd_over_on):
        orig_input = input

        assert stride in [1, 2], 'only strides 1 and 2 are supported'

        B, IC, M, N = orig_input.shape
        OM = math.ceil((M + 2 * padding - K + 1) / stride)
        ON = math.ceil((N + 2 * padding - K + 1) / stride)

        q_input, q_input_dq_info = quantize(input, 'sawb', conf.PARALLEL, conf.GROUPED)
        if vectorizing_fwd_over_on:
            assert stride == 1  # only stride 1 is supported in this case, for now
            #q_input, q_input_dq_info = quantize(input, 'sawb', conf.PARALLEL, conf.GROUPED)
            output = torch.zeros(B, OC, OM, ON).float()
            conv2d_forward_over_ON(
                q_input, q_input_dq_info, *W_idx, W_val_i, W_val_scales,
                output, K, padding, conf.PARALLEL
            )
        else:
            q_input_perm = q_input.permute(1, 2, 3, 0).contiguous()
            #q_input, q_input_dq_info = quantize(input, 'sawb', conf.PARALLEL, conf.GROUPED)
            output = torch.zeros(OC, OM, ON, B).float()
            conv2d_forward(
                q_input_perm, q_input_dq_info, *W_idx, W_val_i, W_val_scales,
                output, K, padding, stride, conf.PARALLEL, conf.GROUPED
            )

            output = output.permute(3, 0, 1, 2)
        #if stride == 1:
        #    qsppb.uq_sparse_conv2d_vectorized_forward_stride_1(q_input * q_input_dq_info, *W_idx, W_val_i * W_val_scales, output, K, padding)
        #elif stride == 2:
        #    qsppb.uq_sparse_conv2d_vectorized_forward_stride_2(q_input * q_input_dq_info, *W_idx, W_val_i * W_val_scales, output, K, padding)

        ctx.save_for_backward(W_val_i, bias)
        if vectorizing_bwd_over_on:
            ctx.svd = (q_input, q_input_dq_info, *W_idx, W_val_scales)
        else:
            ctx.svd = (q_input_perm, q_input_dq_info, *W_idx, W_val_scales)
        ctx.K, ctx.padding, ctx.stride = K, padding, stride
        ctx.vectorizing_bwd_over_on = vectorizing_bwd_over_on

        if bias is not None:
            output += bias.view(1, -1, 1, 1)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        W_val_i, bias = ctx.saved_tensors
        q_input, q_input_dq_info, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val_scales = ctx.svd
        K, padding, stride = ctx.K, ctx.padding, ctx.stride

        vectorizing_bwd_over_on = ctx.vectorizing_bwd_over_on

        grad_input = torch.zeros_like(q_input).float().contiguous()
        grad_W_val = torch.zeros_like(W_val_i).float().contiguous()

        assert stride in [1, 2], 'only strides 1 and 2 are supported'

        if vectorizing_bwd_over_on:
            q_grad_output, q_grad_output_dq_info = quantize(grad_output, 'dithered', conf.PARALLEL, conf.GROUPED)
            conv2d_backward_over_ON(
                q_input, q_input_dq_info, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val_i,
                W_val_scales, q_grad_output, q_grad_output_dq_info, grad_input, grad_W_val, K, padding, stride,
                conf.PARALLEL
            )
        else:
            go = grad_output.permute(1, 2, 3, 0).contiguous()
            q_grad_output, q_grad_output_dq_info = quantize(go, 'dithered', conf.PARALLEL, conf.GROUPED)
            conv2d_backward(
                q_input, q_input_dq_info, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val_i,
                W_val_scales, q_grad_output, q_grad_output_dq_info, grad_input, grad_W_val, K, padding, stride,
                conf.PARALLEL, conf.GROUPED
            )

        #if stride == 1:
        #    qsppb.uq_sparse_conv2d_vectorized_backward_stride_1(q_input * q_input_dq_info, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val_i * W_val_scales, q_grad_output * q_grad_output_dq_info, grad_input, grad_W_val, K, padding)
        #elif stride == 2:
        #    qsppb.uq_sparse_conv2d_vectorized_backward_stride_2(q_input * q_input_dq_info, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val_i * W_val_scales, q_grad_output * q_grad_output_dq_info, grad_input, grad_W_val, K, padding)

        grad_bias = None
        if bias is not None:
            grad_bias = grad_output.sum(dim=(0, 2, 3))

        if not vectorizing_bwd_over_on:
            grad_input = grad_input.permute(3, 0, 1, 2)

        return grad_input, grad_W_val, None, grad_bias, None, None, None, None, None, None, None, None  # Need to return one more value for W_scale
