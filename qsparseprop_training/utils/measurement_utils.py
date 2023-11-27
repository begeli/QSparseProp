import numpy as np
import math

"""
Utility functions to count number of flops and the byte transfers 
in order to compute an approximation of the operational intensity
"""
# Utils for flop counts
def count_sawb_quantization_flops(input):
    """
    4 Flops for computing scale, 1 FMADD, 1 Logical AND, 1 ADD
    3 Flops for computing unpacked integer values, 1 MUL, 1 MAX, 1 MIN
    (11 / 4) Flops for packing, we assume size is divisible by 4
    :param input:
    :return: Approximate number of floating point operations executed during quantization of input
    """
    size = np.prod(input.size())
    return 4 * size + 3 * size + (11 / 4) * size


def count_dithered_quantization_flops(input):
    """
    4 Flops for computing scale, 1 FMADD, 1 Logical AND, 1 ADD
    7 Flops for computing unpacked integer values, 5 for RNG - 1 AND/Shift, 1 FMSUB, 1 FMADD, 2 for rest, 1 DIV, 1 ADD
    (11 / 4) Flops for packing, we assume size is divisible by 4
    :param input:
    :return: Approximate number of floating point operations executed during quantization of input
    """
    size = np.prod(input.size())
    return 3 * size + 7 * size + (11 / 4) * size


def count_conv2d_forward_flops(OC, IC, B, OM, ON, M, N, W_idx, padding, stride):
    op_count = 0
    W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y = W_idx
    for oc in range(OC):
        for ic in range(IC):
            oc_start = W_idx_OC[oc]
            ic_start = oc_start + W_idx_IC[(IC + 1) * oc + ic]
            ic_end = oc_start + W_idx_IC[(IC + 1) * oc + ic + 1]
            for si in range(ic_start, ic_end, 1):
                i = W_idx_X[si]
                j = W_idx_Y[si]

                op_count += 2

                pdmi = padding - i
                pdmj = padding - j
                if stride == 1:
                    p_start = max(pdmi, 0)
                    p_end = min(pdmi + M, OM)
                    q_start = max(pdmj, 0)
                    q_end = min(pdmj + N, ON)
                elif stride == 2:
                    p_start = max(math.ceil(pdmi / 2.0), 0)
                    p_end = min(math.floor((pdmi + M - 1) / 2.0) + 1, OM)
                    q_start = max(math.ceil(pdmj / 2.0), 0)
                    q_end = min(math.floor((pdmj + N - 1)) + 1, ON)

                for po in range(p_start, p_end, 1):
                    for qo in range(q_start, q_end, 1):
                        op_count += B * 2 + B * 3  # 1 FMADD, 1 Logical OR, 1 Left Shift, 1 Right Shift

    return op_count


def count_conv2d_backward_flops(OC, IC, B, OM, ON, M, N, W_idx, padding, stride):
    op_count = 0
    W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y = W_idx
    for ic in range(IC):
        for oc in range(OC):
            oc_start = W_idx_OC[oc]
            ic_start = oc_start + W_idx_IC[(IC + 1) * oc + ic]
            ic_end = oc_start + W_idx_IC[(IC + 1) * oc + ic + 1]
            for si in range(ic_start, ic_end, 1):
                i = W_idx_X[si]
                j = W_idx_Y[si]

                op_count += 3

                pdmi = padding - i
                pdmj = padding - j
                if stride == 1:
                    p_start = max(pdmi, 0)
                    p_end = min(pdmi + M, OM)
                    q_start = max(pdmj, 0)
                    q_end = min(pdmj + N, ON)
                elif stride == 2:
                    p_start = max(math.ceil(pdmi / 2.0), 0)
                    p_end = min(math.floor((pdmi + M - 1) / 2.0) + 1, OM)
                    q_start = max(math.ceil(pdmj / 2.0), 0)
                    q_end = min(math.floor((pdmj + N - 1)) + 1, ON)

                for po in range(p_start, p_end, 1):
                    for qo in range(q_start, q_end, 1):
                        """
                        (B / 64) iterations
                        32 * 4 flops for 4 FMADD
                        16 flops for 1 AND
                        64 * 2 flops for 2 ABS
                        16 flops for 1 AND
                        64 flops for 1 8-bit CMP
                        64 flops for 1 8-bit SUB
                        128 flops for dpbusd
                        (B/64) * 548 flops = 8.5 B flops
                        
                        3 * B flops for unpacking
                        """
                        op_count += B * 8.5 + B * 3

    return op_count


# Utils for Byte transfers from memory to cache
def count_sawb_byte_transfers(input):
    """
    LOAD 4 bytes per element to compute scale
    LOAD 4 bytes per element to compute unpacked integers
    STORE 1 byte per element to store packed quantized integers
    :param input:
    :return: Approximate number of bytes transferred from memory to caches during quantization,
    assuming the data is too large to fit into cache
    """
    size = np.prod(input.size())
    return 4 * size + 4 * size + size


def count_dithered_byte_transfers(input):
    """
    LOAD 4 bytes per element to compute scale
    LOAD 4 bytes per element to compute unpacked integers
    STORE 1 byte per element to store packed quantized integers
    :param input:
    :return: Approximate number of bytes transferred from memory to caches during quantization,
    assuming the data is too large to fit into cache
    """
    size = np.prod(input.size())
    return 4 * size + 4 * size + size


def count_conv2d_forward_byte_transfers(OC, IC, B, OM, ON, M, N, W_idx, padding, stride, data_size):
    byte_transfers = 0
    W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y = W_idx
    for oc in range(OC):
        byte_transfers += 4  # 4 bytes for each element of W_idx_OC
        for ic in range(IC):
            oc_start = W_idx_OC[oc]
            ic_start = oc_start + W_idx_IC[(IC + 1) * oc + ic]
            ic_end = oc_start + W_idx_IC[(IC + 1) * oc + ic + 1]
            byte_transfers += 2 + 2  # 2 bytes per element of W_idx_IC
            for si in range(ic_start, ic_end, 1):
                i = W_idx_X[si]
                j = W_idx_Y[si]
                byte_transfers += 1 + 1 + data_size  # 1 byte for each element of W_idx_X, 1 byte for each element of W_idx_Y, 1 Byte to load the weight

                pdmi = padding - i
                pdmj = padding - j
                if stride == 1:
                    p_start = max(pdmi, 0)
                    p_end = min(pdmi + M, OM)
                    q_start = max(pdmj, 0)
                    q_end = min(pdmj + N, ON)
                elif stride == 2:
                    p_start = max(math.ceil(pdmi / 2.0), 0)
                    p_end = min(math.floor((pdmi + M - 1) / 2.0) + 1, OM)
                    q_start = max(math.ceil(pdmj / 2.0), 0)
                    q_end = min(math.floor((pdmj + N - 1)) + 1, ON)

                for po in range(p_start, p_end, 1):
                    for qo in range(q_start, q_end, 1):
                        byte_transfers += B * 4 * 2 + B * data_size  # Load & store 4 bytes per each element of O, Load 1 byte per each element of X

    return byte_transfers


def count_conv2d_backward_byte_transfers(OC, IC, B, OM, ON, M, N, W_idx, padding, stride, data_size):
    byte_transfers = 0
    W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y = W_idx
    for ic in range(IC):
        for oc in range(OC):
            oc_start = W_idx_OC[oc]
            ic_start = oc_start + W_idx_IC[(IC + 1) * oc + ic]
            ic_end = oc_start + W_idx_IC[(IC + 1) * oc + ic + 1]
            byte_transfers += 4 + 2 + 2  # 4 bytes to read W_idx_OC[oc], 2 bytes to read each element of W_idx_IC
            for si in range(ic_start, ic_end, 1):
                i = W_idx_X[si]
                j = W_idx_Y[si]
                byte_transfers += 1 + 1 + data_size

                pdmi = padding - i
                pdmj = padding - j
                if stride == 1:
                    p_start = max(pdmi, 0)
                    p_end = min(pdmi + M, OM)
                    q_start = max(pdmj, 0)
                    q_end = min(pdmj + N, ON)
                elif stride == 2:
                    p_start = max(math.ceil(pdmi / 2.0), 0)
                    p_end = min(math.floor((pdmi + M - 1) / 2.0) + 1, OM)
                    q_start = max(math.ceil(pdmj / 2.0), 0)
                    q_end = min(math.floor((pdmj + N - 1)) + 1, ON)

                for po in range(p_start, p_end, 1):
                    for qo in range(q_start, q_end, 1):
                        """
                        4 bytes to read and store each element of dLdX, 
                        1 byte to read each element of X, 1 byte to read each element of dLdO
                        """
                        byte_transfers += 4 * B + 4 * B + data_size * B + data_size * B

                byte_transfers += 4 + 4  # 4 bytes to read dLdW[si], 4 bytes to write dLdW[si]

    return byte_transfers
