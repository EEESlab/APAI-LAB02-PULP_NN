#include "pmsis.h"


// generic matrix multiplication
void gemm(int * MatA, int * MatB, int* MatC, int NN, int MM, int KK){
    uint32_t i, core_id, i_chunk, i_start, i_end;

    core_id = pi_core_id();
    i_chunk = NN / NUM_CORES;
    i_start = core_id * i_chunk;
    i_end   = i_start + i_chunk;

    // task to profile
    for (i = i_start; i < i_end; i += i_chunk) {
      for (int j = 0; j < MM; j++) {
        int acc = 0;
        for (int k = 0; k < KK; k++) {
          acc += MatA[i*KK+k] * MatB[k*MM+j];
        } //k
        MatC[i*MM+j] = acc;
      }//j
    }//i
}
