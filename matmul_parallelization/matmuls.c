#include "pmsis.h"


// generic matrix multiplication
void gemm(int * MatA, int * MatB, int* MatC, int NN, int MM, int KK){
    uint32_t core_id, i_chunk, i_start, i_end;
    uint32_t i = 0;

    core_id = pi_core_id();
    i_chunk = (NN + NUM_CORES-1) / NUM_CORES;
    i_start = core_id * i_chunk;
    i_end   = i_start + i_chunk < NN ? i_start + i_chunk : NN;

    // task to profile
    for (i = i_start; i < i_end; i ++) {
      for (int j = 0; j < MM; j++) {
        int acc = 0;
        for (int k = 0; k < KK; k++) {
          acc += MatA[i*KK+k] * MatB[k*MM+j];
        } //k
        MatC[i*MM+j] = acc;
      }//j
    }//i
    pi_cl_team_barrier();

}
/*
// matrix multiplication with loop unrolling 1x4
void gemm_unroll_1x4(int * MatA, int * MatB, int* MatC, int NN, int MM, int KK){

  core_id = pi_core_id();
  i_chunk = (NN + NUM_CORES-1) / NUM_CORES;
  i_start = core_id * i_chunk;
  i_end   = i_start + i_chunk < NN ? i_start + i_chunk : NN;

  for (i = i_start; i < i_end; i ++) {

  // INSERT YOUR CODE HERE  

  }
  pi_cl_team_barrier();

}
*/