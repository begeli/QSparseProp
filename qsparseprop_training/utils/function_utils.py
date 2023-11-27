import qsparseprop_backend as qsppb
from configurations import conf


def linear_forward(X, X_scale, W_val, W_idx_N, W_idx_M, W_val_scale, O, parallel=False, grouped=False):
    if parallel:
        if grouped:
            qsppb.sparse_linear_vectorized_grouped_parallel_forward(
                X, X_scale, conf.FORWARD_QUANTIZATION_GROUP_SIZE,
                W_idx_N, W_idx_M, W_val, W_val_scale,
                conf.FORWARD_QUANTIZATION_GROUP_SIZE, O
            )
        else:
            qsppb.sparse_linear_vectorized_parallel_forward(X, X_scale, W_idx_N, W_idx_M, W_val, W_val_scale, O)
    else:
        if grouped:
            qsppb.sparse_linear_vectorized_grouped_forward(
                X, X_scale, conf.FORWARD_QUANTIZATION_GROUP_SIZE,
                W_idx_N, W_idx_M, W_val, W_val_scale,
                conf.FORWARD_QUANTIZATION_GROUP_SIZE, O
            )
        else:
            qsppb.sparse_linear_vectorized_forward(X, X_scale, W_idx_N, W_idx_M, W_val, W_val_scale, O)


def linear_backward(
        X, X_scale, W_val, W_idx_N, W_idx_M, W_val_scale,
        dLdO, dLdO_scale, dLdX, dLdW, parallel=False, grouped=False
):
    if parallel:
        if grouped:
            qsppb.sparse_linear_vectorized_grouped_parallel_backward(
                X, X_scale, conf.FORWARD_QUANTIZATION_GROUP_SIZE,
                W_idx_N, W_idx_M, W_val, W_val_scale,
                conf.FORWARD_QUANTIZATION_GROUP_SIZE, dLdO, dLdO_scale,
                conf.BACKWARD_QUANTIZATION_GROUP_SIZE, dLdX, dLdW
            )
        else:
            qsppb.sparse_linear_vectorized_parallel_backward(
                X, X_scale, W_idx_N, W_idx_M, W_val,
                W_val_scale, dLdO, dLdO_scale, dLdX, dLdW
            )
    else:
        if grouped:
            qsppb.sparse_linear_vectorized_grouped_backward(
                X, X_scale, conf.FORWARD_QUANTIZATION_GROUP_SIZE,
                W_idx_N, W_idx_M, W_val, W_val_scale,
                conf.FORWARD_QUANTIZATION_GROUP_SIZE, dLdO, dLdO_scale,
                conf.BACKWARD_QUANTIZATION_GROUP_SIZE, dLdX, dLdW
            )
        else:
            qsppb.sparse_linear_vectorized_backward(
                X, X_scale, W_idx_N, W_idx_M, W_val,
                W_val_scale, dLdO, dLdO_scale, dLdX, dLdW
            )


def conv2d_forward_over_ON(X, X_scale, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val, W_scale, O, kernel_size, padding, parallel=False):
    if parallel:
        qsppb.sparse_conv2d_vectorized_parallel_forward_over_on_stride_1(
            X, X_scale, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val, W_scale, O, kernel_size, padding
        )
    else:
        qsppb.sparse_conv2d_vectorized_forward_over_on_stride_1(
            X, X_scale, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val, W_scale, O, kernel_size, padding
        )


def conv2d_forward(
    X, X_scale, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val, W_scale, O, K, padding, stride, parallel=False, grouped=False
):
    if stride == 1:
        _conv2d_forward_stride_1(
            X, X_scale, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val, W_scale, O, K, padding, parallel, grouped
        )
    else:
        _conv2d_forward_stride_2(
            X, X_scale, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val, W_scale, O, K, padding, parallel, grouped
        )



def _conv2d_forward_stride_1(X, X_scale, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val, W_scale, O, K, padding, parallel=False, grouped=False):
    if parallel:
        if grouped:
            qsppb.sparse_conv2d_vectorized_grouped_parallel_forward_stride_1(
                X, X_scale, conf.FORWARD_QUANTIZATION_GROUP_SIZE, W_idx_OC, W_idx_IC,
                W_idx_X, W_idx_Y, W_val, W_scale, conf.FORWARD_QUANTIZATION_GROUP_SIZE, O, K, padding
            )
        else:
            qsppb.sparse_conv2d_vectorized_parallel_forward_stride_1(
                X, X_scale, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val, W_scale, O, K, padding
            )
    else:
        if grouped:
            qsppb.sparse_conv2d_vectorized_grouped_forward_stride_1(
                X, X_scale, conf.FORWARD_QUANTIZATION_GROUP_SIZE, W_idx_OC, W_idx_IC,
                W_idx_X, W_idx_Y, W_val, W_scale, conf.FORWARD_QUANTIZATION_GROUP_SIZE, O, K, padding
            )
        else:
            qsppb.sparse_conv2d_vectorized_forward_stride_1(
                X, X_scale, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val, W_scale, O, K, padding
            )


def _conv2d_forward_stride_2(X, X_scale, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val, W_scale, O, K, padding, parallel=False, grouped=False):
    if parallel:
        if grouped:
            qsppb.sparse_conv2d_vectorized_grouped_parallel_forward_stride_2(
                X, X_scale, conf.FORWARD_QUANTIZATION_GROUP_SIZE, W_idx_OC, W_idx_IC,
                W_idx_X, W_idx_Y, W_val, W_scale, conf.FORWARD_QUANTIZATION_GROUP_SIZE, O, K, padding
            )
        else:
            qsppb.sparse_conv2d_vectorized_parallel_forward_stride_2(
                X, X_scale, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val, W_scale, O, K, padding
            )
    else:
        if grouped:
            qsppb.sparse_conv2d_vectorized_grouped_forward_stride_2(
                X, X_scale, conf.FORWARD_QUANTIZATION_GROUP_SIZE, W_idx_OC, W_idx_IC,
                W_idx_X, W_idx_Y, W_val, W_scale, conf.FORWARD_QUANTIZATION_GROUP_SIZE, O, K, padding
            )
        else:
            qsppb.sparse_conv2d_vectorized_forward_stride_2(
                X, X_scale, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val, W_scale, O, K, padding
            )


def conv2d_backward_over_ON(
    X, X_scale, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val, W_scale,
    dLdO, dLdO_scale, dLdX, dLdW_val, kernel_size, padding, stride, parallel=False
):
    if parallel:
        if stride == 1:
            qsppb.sparse_conv2d_vectorized_parallel_backward_over_on_stride_1(
                X, X_scale, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val, W_scale,
                dLdO, dLdO_scale, dLdX, dLdW_val, kernel_size, padding
            )
        elif stride == 2:
            qsppb.sparse_conv2d_vectorized_parallel_backward_over_on_stride_2(
                X, X_scale, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val, W_scale,
                dLdO, dLdO_scale, dLdX, dLdW_val, kernel_size, padding
            )
    else:
        if stride == 1:
            qsppb.sparse_conv2d_vectorized_backward_over_on_stride_1(
                X, X_scale, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val, W_scale,
                dLdO, dLdO_scale, dLdX, dLdW_val, kernel_size, padding
            )
        elif stride == 2:
            qsppb.sparse_conv2d_vectorized_backward_over_on_stride_2(
                X, X_scale, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val, W_scale,
                dLdO, dLdO_scale, dLdX, dLdW_val, kernel_size, padding
            )


def conv2d_backward(
        X, X_scale, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val, W_scale,
        dLdO, dLdO_scale, dLdX, dLdW, K, padding, stride, parallel=False, grouped=False
):
    if stride == 1:
        _conv2d_backward_stride_1(
            X, X_scale, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val, W_scale,
            dLdO, dLdO_scale, dLdX, dLdW, K, padding, parallel, grouped
        )
    elif stride == 2:
        _conv2d_backward_stride_2(
            X, X_scale, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val, W_scale,
            dLdO, dLdO_scale, dLdX, dLdW, K, padding, parallel, grouped
        )


def _conv2d_backward_stride_1(
    X, X_scale, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val, W_scale,
    dLdO, dLdO_scale, dLdX, dLdW, K, padding, parallel=False, grouped=False
):
    if parallel:
        if grouped:
            qsppb.sparse_conv2d_vectorized_grouped_parallel_backward_stride_1(
                X, X_scale, conf.FORWARD_QUANTIZATION_GROUP_SIZE, W_idx_OC, W_idx_IC,
                W_idx_X, W_idx_Y, W_val, W_scale, conf.FORWARD_QUANTIZATION_GROUP_SIZE,
                dLdO, dLdO_scale, conf.BACKWARD_QUANTIZATION_GROUP_SIZE, dLdX, dLdW, K, padding
            )
        else:
            qsppb.sparse_conv2d_vectorized_parallel_backward_stride_1(
                X, X_scale, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val, W_scale,
                dLdO, dLdO_scale, dLdX, dLdW, K, padding
            )
    else:
        if grouped:
            qsppb.sparse_conv2d_vectorized_grouped_backward_stride_1(
                X, X_scale, conf.FORWARD_QUANTIZATION_GROUP_SIZE, W_idx_OC, W_idx_IC,
                W_idx_X, W_idx_Y, W_val, W_scale, conf.FORWARD_QUANTIZATION_GROUP_SIZE,
                dLdO, dLdO_scale, conf.BACKWARD_QUANTIZATION_GROUP_SIZE, dLdX, dLdW, K, padding
            )
        else:
            qsppb.sparse_conv2d_vectorized_backward_stride_1(
                X, X_scale, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val, W_scale,
                dLdO, dLdO_scale, dLdX, dLdW, K, padding
            )


def _conv2d_backward_stride_2(
    X, X_scale, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val, W_scale,
    dLdO, dLdO_scale, dLdX, dLdW, K, padding, parallel=False, grouped=False
):
    if parallel:
        if grouped:
            qsppb.sparse_conv2d_vectorized_grouped_parallel_backward_stride_2(
                X, X_scale, conf.FORWARD_QUANTIZATION_GROUP_SIZE, W_idx_OC, W_idx_IC,
                W_idx_X, W_idx_Y, W_val, W_scale, conf.FORWARD_QUANTIZATION_GROUP_SIZE,
                dLdO, dLdO_scale, conf.BACKWARD_QUANTIZATION_GROUP_SIZE, dLdX, dLdW, K, padding
            )
        else:
            qsppb.sparse_conv2d_vectorized_parallel_backward_stride_2(
                X, X_scale, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val, W_scale,
                dLdO, dLdO_scale, dLdX, dLdW, K, padding
            )
    else:
        if grouped:
            qsppb.sparse_conv2d_vectorized_grouped_backward_stride_2(
                X, X_scale, conf.FORWARD_QUANTIZATION_GROUP_SIZE, W_idx_OC, W_idx_IC,
                W_idx_X, W_idx_Y, W_val, W_scale, conf.FORWARD_QUANTIZATION_GROUP_SIZE,
                dLdO, dLdO_scale, conf.BACKWARD_QUANTIZATION_GROUP_SIZE, dLdX, dLdW, K, padding
            )
        else:
            qsppb.sparse_conv2d_vectorized_backward_stride_2(
                X, X_scale, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val, W_scale,
                dLdO, dLdO_scale, dLdX, dLdW, K, padding
            )