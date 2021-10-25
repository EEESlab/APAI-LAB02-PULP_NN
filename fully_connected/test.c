/*
 * test.c
 * Nazareno Bruschi <nazareno.bruschi@unibo.it>
 * Manuele Rusci <manuele.rusci@unibo.it>
 *
 * Copyright (C) 2019-2021 University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "pmsis.h"
#include "pulp_nn_kernels.h"
#include "pulp_nn_utils.h"



// data allocation and golden models
#include "GoldenModelLinearNoQuant/golden_8_32_8.h"
#include "DataAllocationLinearNoQuant/data_allocation_8_32_8.h"


// function prototype
void test();
void pulp_parallel();

void pulp_parallel()
{
  pi_cl_team_fork(NUM_CORES, (void *)test, NULL);
}

void test()
{
  // initialize checksum 
  uint32_t errors = 0;


  // copy inputs and weights from L2 to L1
  if(pi_core_id()==0)
  {
    for(int i=0; i<(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN); i++)
    {
      IN_INT8_L1[i] = IN_INT8_L2[i];
    }

    for(int i=0; i<(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN * CH_IM_OUT); i++)
    {
      WEIGHT_INT8_L1[i] = WEIGHT_INT8_L2[i];
    }
    
    printf("\n\nGoing to run the fully-connected layer!\n");
  }
  pi_cl_team_barrier(0);


  // setup and start performance counters
  pi_perf_conf(1<<PI_PERF_CYCLES);          
  pi_perf_reset();                      
  pi_perf_stop();                       
  pi_perf_start(); 

  // call the fully connected
  pulp_nn_linear_u8_i32_i8(
    IN_INT8_L1,
    BIAS_L1,
    OUT_L1,
    WEIGHT_INT8_L1,
    DIM_IM_IN_X*DIM_IM_IN_Y*CH_IM_IN,
    CH_IM_OUT
  );

  // compute print performance
  pi_perf_stop();          
  int cid = pi_core_id();   
  int perf_cyc =  pi_perf_read(PI_PERF_CYCLES);
  int MACs = CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_OUT;
  float perf_MAC =  (float)MACs/perf_cyc;
  if (cid == 0)
  {
    printf("Fully-connected layer completed!\nRuntime statistics on %d cores:\n", NUM_CORES);
    printf("[%d] : num_cycles: %d\n",cid,perf_cyc); 
    printf("[%d] : MACs: %d\n",cid,MACs ); 
    printf("[%d] : MAC/cycle: %f\n",cid,perf_MAC ); 
  }
  pi_cl_team_barrier(0);

  //check results
  if(pi_core_id()==0)
  {
    for (int i=0; i<CH_IM_OUT; i++)
    {
      if(OUT_L1[i] != OUT_L2[i])
      {
        errors++;
      }
    }
    printf("errors: %d\n", errors);
  }
  pi_cl_team_barrier(0);
}

///////////////////////////////////////////////////////////////////
////------------------------MAIN------------------------------/////
///////////////////////////////////////////////////////////////////

int main()
{
  struct pi_device cluster_dev = {0};
  struct pi_cluster_conf conf;
  struct pi_cluster_task cluster_task = {0};

  // task parameters allocation
  pi_cluster_task(&cluster_task, pulp_parallel, NULL);
  cluster_task.stack_size = 1024;
  cluster_task.slave_stack_size = 1024;

  // First open the cluster
  pi_cluster_conf_init(&conf);
  pi_open_from_conf(&cluster_dev, &conf);
  if (pi_cluster_open(&cluster_dev))
    return -1;

  // Then offload an entry point, this will get executed on the cluster controller
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

  // closing of the cluster
  pi_cluster_close(&cluster_dev);

  return 0;
}
