#include "pmsis.h"
#include "perf.h"

#define N 80
#define M 16
#define K 8

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

void checksum(int * MatA, int a_val, int b_val, int NN, int MM, int KK){
  for (int i=0; i<NN*MM; i++){
    if (MatA[i]!=a_val*b_val*KK){
        printf("ERROR - CHECKSUM WRONG! \n");
        break;
    }
    if (i==NN*MM-1){
      printf("CHECKSUM CORRECT!\n");
    }
  }
}

void cluster_fn() {

  // INIT MATRICES, e.g. with the same value each cell
  int mat_a_val = 2;
  int mat_b_val = 1;

  fill_matrix(A, N, K, mat_a_val);
  fill_matrix(B, K, M, mat_b_val);
  fill_matrix(C, N, M, 0);

  // init performance counters
  INIT_STATS();

  // executing the code multiple times to perform average statistics
  ENTER_STATS_LOOP();

  // start measuring
  START_STATS();

  // task to profile
  gemm(A, B, C, N, M, K);

  // stop measuring
  STOP_STATS();

  // end of the performance statistics loop
  EXIT_STATS_LOOP();

  /* RESULTS CHECKSUM */
  checksum(C, mat_a_val, mat_b_val, N, M, K);

}
