import qsparseprop_backend as qsppb
import torch
from torch.autograd.function import InplaceFunction
from torch.autograd.function import Function
from configurations import conf

from utils import quantization_utils as qu

def quantize(input, bits, signed, scheme='SAWB'):
    if scheme == 'SAWB':
        # Some constants (I don't know what they are but I assume they are from the SAWB paper/implementation.)
        # - constant for 4 bits
        return UniformQuantizeSawb.apply(input, conf.SAWB_COEFFICIENT_1, conf.SAWB_COEFFICIENT_2, conf.SAWB_UPPER_BOUND, conf.SAWB_LOWER_BOUND)
    elif scheme == 'std':
        return StandardQuantization.apply(input, conf.STD_UPPER_BOUND, conf.STD_LOWER_BOUND)
    else:
        raise ValueError


def quantize_grad(output, quantizeBwd=None, layerIdx=None, repeatBwd=None, scheme='dithered', scale=conf.DITHERED_8_BIT_QUANTIZATION_SCALE, bits=4.):
    if scheme == 'LUQ':
        return GradStochasticClippingQ.apply(output, quantizeBwd, layerIdx, repeatBwd, bits)
    elif scheme == 'dithered':
        return DitheredQuantization.apply(output, quantizeBwd, layerIdx, scale)
    else:
        raise NotImplementedError

class UniformQuantizeSawb(InplaceFunction):
    @staticmethod
    def forward(ctx, input, c1, c2, Qu, Ql):

        q_input, q_input_dq_info = qu.quantize(input, 'sawb', conf.PARALLEL, conf.GROUPED)
        return q_input * q_input_dq_info
        """
        output = input.clone()

        with torch.no_grad():
            clip = (c1 * torch.sqrt(torch.mean(input ** 2))) - (c2 * torch.mean(input.abs()))
            scale = 2 * clip / (Qu - Ql)
            output.div_(scale)
            output.clamp_(Ql, Qu).round_()
            output.mul_(scale)
        return output
        """


    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        return grad_input, None, None, None, None


class StandardQuantization(InplaceFunction):
    # TODO: Implement forward and backward implementations for standard quantization
    @staticmethod
    def forward(ctx, input, Qu, Ql, stochastic=False):
        output = input.clone()

        zero_point = torch.min(input)
        range_values = torch.max(input) - zero_point
        scale = range_values / (Qu - Ql)
        with torch.no_grad():
            output.add_(Ql * scale - zero_point).div_(scale)
            if stochastic:
                noise = output.new(output.shape).uniform_(-0.5, 0.5)
                output.add_(noise)
            # quantize
            output.clamp_(Ql, Qu).round_()
            output.mul_(scale).add_(zero_point - Ql * scale)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        return grad_input, None, None


class GradStochasticClippingQ(Function):
    @staticmethod
    def forward(ctx, input, quantizeBwd, layerIdx, repeatBwd, bits):
        ctx.save_for_backward(
            torch.tensor(quantizeBwd),
            torch.tensor(layerIdx),
            torch.tensor(repeatBwd),
            torch.tensor(bits)
        )
        return input

    @staticmethod
    def backward(ctx, grad_output):
        quantizeBwd, layerIdx, repeatBwd, bits = ctx.saved_tensors
        if quantizeBwd:
            out = []
            # SMP to reduce variance - Repeatedly sampling and
            # averaging stochastically quantized tensors should reduce variance
            for i in range(repeatBwd):
                maxOutput = torch.max(grad_output)
                bits = bits - 1. #3  # TODO: Change for different quantization levels
                alpha = maxOutput / pow(2., (2.**bits - 1.))# 2 ** (2 ** bits - 1) # Underflow threshold
                alphaEps = alpha * torch.rand(grad_output.shape, device=grad_output.device)

                grad_abs = grad_output.abs()
                # noinspection PyTypeChecker
                grad_input = torch.where(grad_abs < alpha, alpha * torch.sign(grad_output), grad_output)
                # noinspection PyTypeChecker
                grad_input = torch.where(
                    grad_abs < alphaEps,
                    torch.tensor([0], dtype=torch.float32, device=grad_output.device),
                    grad_input
                )

                grad_inputQ = grad_input.clone()
                # Add random noise to quantized to simulate a cheaper stochastic quantization process
                # Random noise allows us to use standard quantization
                noise = (2. ** torch.floor(torch.log2((grad_inputQ.abs() / alpha)))) * grad_inputQ.new(
                    grad_inputQ.shape).uniform_(-0.5, 0.5)
                grad_inputQ = 2. ** torch.floor(torch.log2(((grad_inputQ.abs() / alpha) + noise) * 4. / 3.)) * alpha
                # Clip at the thresholds
                threshold = (alpha * (2. ** torch.floor(torch.log2(((grad_input.abs() / alpha))))))
                # noinspection PyTypeChecker
                grad_inputQ = torch.sign(grad_input) * torch.where(
                    grad_inputQ < threshold,
                    threshold,
                    grad_inputQ
                )
                grad_inputQ = torch.where(
                    grad_input == 0,
                    torch.tensor([0], dtype=torch.float, device=grad_output.device),
                    grad_inputQ
                )
                grad_inputQ = torch.nan_to_num(grad_inputQ, nan=0.0, neginf=-maxOutput.item(), posinf=maxOutput.item())

                out.append(grad_inputQ)
            # Average proto to reduce variance
            grad_input = sum(out) / repeatBwd
        else:
            grad_input = grad_output

        return grad_input, None, None, None, None


# TODO: Implement Dithered quantization for sparsification
class DitheredQuantization(Function):
    @staticmethod
    def forward(ctx, input, quantizeBwd, layerIdx, scale):
        ctx.save_for_backward(torch.tensor(quantizeBwd), torch.tensor(layerIdx), torch.tensor(scale))
        return input

    @staticmethod
    def backward(ctx, grad_output):
        quantizeBwd, layerIdx, scale = ctx.saved_tensors
        if quantizeBwd:
            # TODO: Try changing rcp_step_size etch with division operations
            """
            q_grad_output, q_grad_output_dq_info = qu.quantize(grad_output, 'dithered', True, conf.GROUPED)
            grad_input = q_grad_output * q_grad_output_dq_info
            """
            grad_input = grad_output.clone()

            # Step 1: Compute standard deviation : Straightforward
            std = torch.std(grad_input, unbiased=False)
            scale = scale.item() #6  # According to paper, scale is a natural number and scale = 6 gives 80% probability of quantiaztion to 0

            # Step 2: Compute range : scale * standard dev
            step_size = scale * std  # delta

            # Step 3: Compute quantization : Look at the formula from the paper
            signal = torch.empty_like(grad_input, dtype=torch.float, device=grad_input.device)\
                .uniform_(-step_size * 0.5, step_size * 0.5)  # v
            grad_input = torch.floor(
                grad_input.add_(signal)
                .div_(step_size) # TODO: Add a small epsilon to avoid div by 0 error
                .add_(torch.tensor([0.5], dtype=torch.float, device=grad_input.device))
            ) #.to(torch.int8)
            grad_input = grad_input * step_size  #    .mul_(step_size)
        else:
            grad_input = grad_output

        return grad_input, None, None, None