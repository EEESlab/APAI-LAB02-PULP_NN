#include "pmsis.h"

#define N 8
#define M 12
#define K 4

int A[N*K];
int B[K*M];
int C[N*M];


void fill_matrix(int * Mat, int height, int width, int val){
    for (int i=0; i<height*width; i++)
    {
      Mat[i] = val;
    }
}

/************************************/
/* MATRIX MULTIPLIACTIONS FUNCTIONS */
/************************************/

// generic matrix multiplication
void gemm(int * MatA, int * MatB, int* MatC, int NN, int MM, int KK){
    // task to profile
    for (int i = 0; i < NN; i++) {
      for (int j = 0; j < MM; j++) {
        int acc = 0;
        for (int k = 0; k < KK; k++) {
          acc += MatA[i*KK+k] * MatB[k*MM+j];
        } //k
        MatC[i*MM+j] = acc;
      }//j
    }//i
}

// matrix multiplication with loop unrolling 4x1
void gemm_unroll_4x1(int * MatA, int * MatB, int* MatC, int NN, int MM, int KK){
    // task to profile
    for (int i = 0; i < NN; i++) {
      for (int j = 0; j < MM; j=j+4) {
        int acc0 = 0;
        int acc1 = 0;
        int acc2 = 0;
        int acc3 = 0;
        for (int k = 0; k < KK; k++) {
           int shared_op = MatA[i*KK+k];
           int idx = k*MM+j;
           acc0   += shared_op * MatB[idx];
           acc1   += shared_op * MatB[idx+1];
           acc2   += shared_op * MatB[idx+2];
           acc3   += shared_op * MatB[idx+3];
        } //k
        MatC[i*MM+j] = acc0;
        MatC[i*MM+j+1] = acc1;
        MatC[i*MM+j+2] = acc2;
        MatC[i*MM+j+3] = acc3;
      }//j
    }//i
}



int main()
{
    // INIT MATRICES, e.g. with the same value each cell
    int mat_a_val = 2;
    int mat_b_val = 1;

    fill_matrix(A, N, K, mat_a_val);
    fill_matrix(B, K, M, mat_b_val);
    fill_matrix(C, N, M, 0);

    // define other variables
    uint32_t instr_cnt,cycles_cnt;

    pi_perf_conf(
        1 << PI_PERF_CYCLES |
        1 << PI_PERF_INSTR
    );

    pi_perf_stop(); // stop the performance counters
    pi_perf_reset();
    pi_perf_start();

    // task to profile
    gemm(A, B, C, N, M, K);

    pi_perf_stop(); // stop the performance counters

    instr_cnt = pi_perf_read(PI_PERF_INSTR);
    cycles_cnt = pi_perf_read(PI_PERF_CYCLES);

    printf("Number of Instructions: %d\nClock Cycles: %d\n",
        instr_cnt, cycles_cnt);

    /* RESULTS CHECKSUM */

    for (int i=0; i<N*M; i++){
      if (i%M==0) printf("\n");
      printf(" %d ", C[i]);
    }
    printf("\n");

/*******************************************************************************/

    fill_matrix(A, N, K, mat_a_val);
    fill_matrix(B, K, M, mat_b_val);
    fill_matrix(C, N, M, 0);

    pi_perf_conf(
      1 << PI_PERF_CYCLES |
      1 << PI_PERF_INSTR
    );

    pi_perf_stop(); // stop the performance counters
    pi_perf_reset();
    pi_perf_start();

    // task to profile
    gemm_unroll_4x1(A, B, C, N, M, K);

    pi_perf_stop(); // stop the performance counters

    instr_cnt = pi_perf_read(PI_PERF_INSTR);
    cycles_cnt = pi_perf_read(PI_PERF_CYCLES);

    printf("Number of Instructions: %d\nClock Cycles: %d\n",
      instr_cnt, cycles_cnt);

    /* RESULTS CHECKSUM */

    for (int i=0; i<N*M; i++){
      if (i%M==0) printf("\n");
      if (C[i]!=8)
      {
        printf("CHECKSUM WRONG!");
        break;
      }
      printf(" %d ", C[i]);
    }
    printf("\n");

}
