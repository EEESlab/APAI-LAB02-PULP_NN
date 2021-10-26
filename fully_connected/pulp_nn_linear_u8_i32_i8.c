/*
 * pulp_nn_linear_u8_i32_i8.c
 * Nazareno Bruschi <nazareno.bruschi@unibo.it>
 *
 * Copyright (C) 2019-2020 University of Bologna
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
#include "pulp_nn_utils.h"


void pulp_nn_linear_u8_i32_i8(
                  uint8_t *pIn,
                  int8_t *pBias,
                  int32_t *pOut,
                  int8_t *pWeight,
                  uint16_t dim_vec,
                  uint16_t num_o_neurons)
{

  int core_id = pi_core_id();
  int Log2Core = log2(NUM_CORES);

  // parallelize over output feature dimension
  int chunk = (num_o_neurons >> Log2Core) + ((num_o_neurons & (NUM_CORES-1))!=0);
  int start = min(chunk * core_id, num_o_neurons);
  int stop = min(start + chunk, num_o_neurons);

  // every core make a copy and increase the output address
  pOut = (int32_t *) pOut + start;

  // iterate over the output feature space
  for(int i=start; i<stop; i++)
  {
    int sum = 0;

    if (pBias != NULL)
    {
      sum = ((int)(pBias[i]));
    }

    uint8_t *pA = pIn; // pA points to the input vector
    int8_t *pB = pWeight + (i * dim_vec); // pB points to weight tensor

    // compute the vectorized dot products
    for (int j=0; j<dim_vec; j++) 
    {
      uint8_t inA = *pA;
      pA++;
      int8_t inB = *pB;
      pB++;
      sum += inA * inB;
    }

    // activation could be applied here
    // else write the accumulator to the output tensor as below
    *pOut = sum;
    pOut++;
  }
  pi_cl_team_barrier(0);
}
