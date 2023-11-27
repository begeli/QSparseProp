import qsparseprop_backend as qsppb
from configurations import conf
import torch
import numpy as np


def quantize(vals, quantization_type, parallel=False, grouped=False):
    if quantization_type == 'std':
        return _standard_8_bit_quantization(vals, parallel, grouped)
    elif quantization_type == 'sawb':
        return _sawb_8_bit_quantization(vals, parallel, grouped)
    elif quantization_type == 'dithered':
        return _dithered_8_bit_quantization(vals, parallel, grouped)
    else:
        raise NotImplementedError()


def _standard_8_bit_quantization(vals, parallel=False, grouped=False):
    quantized_vals = torch.zeros_like(vals, dtype=torch.int8)  #  .contiguous()
    if grouped:
        size = np.prod(vals.size())
        group_count = (size + conf.FORWARD_QUANTIZATION_GROUP_SIZE - 1) // conf.FORWARD_QUANTIZATION_GROUP_SIZE
        scales = torch.zeros(group_count)  #  .contiguous()
        dq_consts = torch.zeros(group_count)  #  .contiguous()

        if parallel:
            qsppb.standard_8_bit_quantization_grouped_parallel(
                vals, quantized_vals, scales, dq_consts,
                conf.FORWARD_QUANTIZATION_GROUP_SIZE, conf.FORWARD_QUANTIZATION_GROUP_SHIFT_AMOUNT,
                conf.STD_LOWER_BOUND, conf.STD_UPPER_BOUND
            )
        else:
            qsppb.standard_8_bit_quantization_grouped(
                vals, quantized_vals, scales, dq_consts,
                conf.FORWARD_QUANTIZATION_GROUP_SIZE, conf.FORWARD_QUANTIZATION_GROUP_SHIFT_AMOUNT,
                conf.STD_LOWER_BOUND, conf.STD_UPPER_BOUND
            )
    else:
        if parallel:
            scales, dq_consts = qsppb.standard_8_bit_quantization_parallel(
                vals, quantized_vals, conf.STD_LOWER_BOUND, conf.STD_UPPER_BOUND
            )
        else:
            scales, dq_consts = qsppb.standard_8_bit_quantization(
                vals, quantized_vals, conf.STD_LOWER_BOUND, conf.STD_UPPER_BOUND
            )

    return quantized_vals, scales, dq_consts


def _sawb_8_bit_quantization(vals, parallel=False, grouped=False):
    quantized_vals = torch.zeros_like(vals, dtype=torch.int8)  #.contiguous()
    if grouped:
        size = np.prod(vals.size())
        group_count = (size + conf.FORWARD_QUANTIZATION_GROUP_SIZE - 1) // conf.FORWARD_QUANTIZATION_GROUP_SIZE
        scales = torch.zeros(group_count)  #  .contiguous()
        dq_consts = torch.zeros(group_count)  #  .contiguous()

        if parallel:
            qsppb.sawb_8_bit_quantization_grouped_parallel(
                vals, quantized_vals, scales, dq_consts,
                conf.FORWARD_QUANTIZATION_GROUP_SIZE, conf.FORWARD_QUANTIZATION_GROUP_SHIFT_AMOUNT,
                conf.SAWB_LOWER_BOUND, conf.SAWB_UPPER_BOUND, conf.SAWB_COEFFICIENT_1, conf.SAWB_COEFFICIENT_2
            )
        else:
            qsppb.sawb_8_bit_quantization_grouped(
                vals, quantized_vals, scales, dq_consts,
                conf.FORWARD_QUANTIZATION_GROUP_SIZE, conf.FORWARD_QUANTIZATION_GROUP_SHIFT_AMOUNT,
                conf.SAWB_LOWER_BOUND, conf.SAWB_UPPER_BOUND, conf.SAWB_COEFFICIENT_1, conf.SAWB_COEFFICIENT_2
            )
    else:
        if parallel:
            scales = qsppb.sawb_8_bit_quantization_parallel(
                vals, quantized_vals, conf.SAWB_LOWER_BOUND, conf.SAWB_UPPER_BOUND,
                conf.SAWB_COEFFICIENT_1, conf.SAWB_COEFFICIENT_2
            )
        else:
            scales = qsppb.sawb_8_bit_quantization(
                vals, quantized_vals, conf.SAWB_LOWER_BOUND, conf.SAWB_UPPER_BOUND,
                conf.SAWB_COEFFICIENT_1, conf.SAWB_COEFFICIENT_2
            )

    return quantized_vals, scales


def _dithered_8_bit_quantization(vals, parallel=False, grouped=False):
    quantized_vals = torch.zeros_like(vals, dtype=torch.int8)  #  .contiguous()
    if grouped:
        size = np.prod(vals.size())
        group_count = (size + conf.BACKWARD_QUANTIZATION_GROUP_SIZE - 1) // conf.BACKWARD_QUANTIZATION_GROUP_SIZE
        scales = torch.zeros(group_count)  #  .contiguous()
        dq_consts = torch.zeros(group_count)  #  .contiguous()

        if parallel:
            qsppb.dithered_8_bit_quantization_grouped_parallel(
                vals, quantized_vals, scales, dq_consts,
                conf.BACKWARD_QUANTIZATION_GROUP_SIZE, conf.BACKWARD_QUANTIZATION_GROUP_SHIFT_AMOUNT,
                conf.DITHERED_8_BIT_QUANTIZATION_SCALE
            )
        else:
            qsppb.dithered_8_bit_quantization_grouped(
                vals, quantized_vals, scales, dq_consts,
                conf.BACKWARD_QUANTIZATION_GROUP_SIZE, conf.BACKWARD_QUANTIZATION_GROUP_SHIFT_AMOUNT,
                conf.DITHERED_8_BIT_QUANTIZATION_SCALE
            )
    else:
        #signal = torch.empty_like(vals, dtype=torch.float, device=vals.device).uniform_(-0.5, 0.5)
        #scales = qsppb.dithered_8_bit_scalar_quantization(
        #    vals, quantized_vals, conf.DITHERED_8_BIT_QUANTIZATION_SCALE, signal
        #)
        # scales = _proto_dithered_quantization(vals, quantized_vals, conf.DITHERED_8_BIT_QUANTIZATION_SCALE)
        if parallel:
            scales = qsppb.dithered_8_bit_quantization_parallel(
                vals, quantized_vals, conf.DITHERED_8_BIT_QUANTIZATION_SCALE #, signal
            )
        else:
            scales = qsppb.dithered_8_bit_quantization(
                vals, quantized_vals, conf.DITHERED_8_BIT_QUANTIZATION_SCALE #, signal
            )

    return quantized_vals, scales


def _proto_sawb_quantization(input, output):
    tmp_output = input.clone()

    with torch.no_grad():
        clip = (conf.SAWB_COEFFICIENT_1 * torch.sqrt(torch.mean(input ** 2))) - (conf.SAWB_COEFFICIENT_2 * torch.mean(input.abs()))
        scale = 2 * clip / (conf.SAWB_UPPER_BOUND - conf.SAWB_LOWER_BOUND)
        tmp_output.div_(scale)
        tmp_output.clamp_(conf.SAWB_LOWER_BOUND, conf.SAWB_UPPER_BOUND).round_()
        tmp_output = tmp_output.to(torch.int8)
        output.copy_(tmp_output)

    return scale

def _proto_dithered_quantization(vals, quantized_vals, scale):
    grad_input = vals.clone()

    # Step 1: Compute standard deviation : Straightforward
    std = torch.std(grad_input, unbiased=False)

    # Step 2: Compute range : scale * standard dev
    step_size = scale * std  # delta

    # Step 3: Compute quantization : Look at the formula from the paper
    signal = torch.empty_like(grad_input, dtype=torch.float, device=grad_input.device).uniform_(-0.5, 0.5) * step_size
        #.uniform_(-step_size * 0.5, step_size * 0.5)  # v
    grad_input = torch.floor(
        grad_input.add_(signal)
        .div_(step_size)  # TODO: Add a small epsilon to avoid div by 0 error
        .add_(torch.tensor([0.5], dtype=torch.float, device=grad_input.device))
    ).to(torch.int8)
    quantized_vals.copy_(grad_input)

    return step_size