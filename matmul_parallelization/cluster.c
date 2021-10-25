#include "pmsis.h"
#include "perf.h"

#define N 80
#define M 12
#define K 4

PI_L1 int A[N*K];
PI_L1 int B[K*M];
PI_L1 int C[N*M];

void gemm(int * MatA, int * MatB, int* MatC, int NN, int MM, int KK);

void fill_matrix(int * Mat, int height, int width, int val){
    for (int i=0; i<height*width; i++)
    {
      Mat[i] = val;
    }
}

void cluster_fn() {

  // INIT MATRICES, e.g. with the same value each cell
  int mat_a_val = 2;
  int mat_b_val = 1;

  fill_matrix(A, N, K, mat_a_val);
  fill_matrix(B, K, M, mat_b_val);
  fill_matrix(C, N, M, 0);

  // // init performance counters
  // INIT_STATS();
  //
  // // executing the code multiple times to perform average statistics
  // ENTER_STATS_LOOP();
  //
  // // start measuring
  // START_STATS();

  uint32_t instr_cnt, cycles_cnt;

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


  // // stop measuring
  // STOP_STATS();
  //
  // // end of the performance statistics loop
  // EXIT_STATS_LOOP();
  //
  // /* RESULTS CHECKSUM */


}
