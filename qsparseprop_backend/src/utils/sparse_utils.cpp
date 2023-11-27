#include "sparse_utils.h"

void sparsify_conv2d(
    int IC, int OC, int K, float* W, int* W_idx_OC,
    int16_t* W_idx_IC, uint8_t* W_idx_X,
    uint8_t* W_idx_Y, float* W_val
) {
    int new_si = 0;
    W_idx_OC[0] = 0;
    for (int oc = 0; oc < OC; oc++) {
        W_idx_IC[oc * (IC + 1)] = 0;

        for (int ic = 0; ic < IC; ic++) {
            int counter = 0;

            for(int x = 0; x < K; x++) {
                for(int y = 0; y < K; y++) {
                    int idx = oc * IC * K * K + ic * K * K + x * K + y;
                    if(W[idx] != 0){
                        W_val[new_si] = W[idx];
                        W_idx_X[new_si] = x;
                        W_idx_Y[new_si] = y;
                        new_si++;
                        counter++;
                    }
                }
            }
            W_idx_IC[(IC + 1) * oc + ic + 1] = W_idx_IC[(IC + 1) * oc + ic] + counter;
        }
        W_idx_OC[oc+1] = new_si;
    }
}

void densify_conv2d(
        int IC, int OC, int K, float* W, int* W_idx_OC,
        int16_t* W_idx_IC, uint8_t* W_idx_X,
        uint8_t* W_idx_Y, float* W_val
) {
    for (int oc = 0; oc < OC; oc++){
        for (int ic = 0; ic < IC; ic++){
            int oc_s = W_idx_OC[oc];
            int ic_s = oc_s + W_idx_IC[(IC + 1) * oc + ic];
            int ic_e = oc_s + W_idx_IC[(IC + 1) * oc + ic + 1];

            for (int si = ic_s; si < ic_e; si++) {
                uint8_t i = W_idx_X[si];
                uint8_t j = W_idx_Y[si];

                float v = W_val[si];
                W[IC * K * K * oc + K * K * ic + K * i + j] = v;
            }
        }
    }
}