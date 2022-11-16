#include "timer.hpp"
#include "cuda_errchk.hpp"
#include <algorithm>
#include <iostream>
#include <stdio.h>


// result = (x, y)
__global__ void cuda_1c(int N, double *x, double *sum, double *abssum, double *squares, double *zeros)
{

  double sum_thr = 0;
  double abssum_thr = 0;
  double squares_thr = 0;
  double zeros_thr = 0;

  //calculation
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    sum_thr  += x[i];
    abssum_thr += abs(x[i]);
    squares_thr += pow(x[i],2);
    zeros_thr += (double)x[i]==0;
  }
  //reduction
  for (int i=warpSize/2; i>0; i=i/2){
     sum_thr += __shfl_down_sync(-1, sum_thr, i);
     abssum_thr += __shfl_down_sync(-1, abssum_thr, i);
     squares_thr += __shfl_down_sync(-1, squares_thr, i);
     zeros_thr += __shfl_down_sync(-1, zeros_thr, i);
  }

  //atomicAdd

  if ((threadIdx.x &(warpSize-1))== 0) {
    atomicAdd(sum, sum_thr);
    atomicAdd(abssum, abssum_thr);
    atomicAdd(squares, squares_thr);
    atomicAdd(zeros, zeros_thr);
    
  }


}



int main() {

  int N = 1000000;

  // Allocate arrays and doubles on CPU
  double *x = (double *)malloc(sizeof(double) * N);
  double *x_sum = (double *)malloc(sizeof(double));
  double *x_abssum = (double *)malloc(sizeof(double));
  double *x_squares = (double *)malloc(sizeof(double));
  double *x_zeros = (double *)malloc(sizeof(double));
  // Initialize arrays and doubles on CPU
  std::fill(x, x + N, 1);
  *x_sum = 0;
  *x_abssum = 0;
  *x_squares = 0;
  *x_zeros = 0;

  // Allocate arrays and doubles on GPU
  double *cuda_x, *cuda_sum, *cuda_abssum, *cuda_squares, *cuda_zeros;
  
  CUDA_ERRCHK(cudaMalloc(&cuda_x, sizeof(double) * N));
  CUDA_ERRCHK(cudaMalloc(&cuda_sum, sizeof(double)));
  CUDA_ERRCHK(cudaMalloc(&cuda_abssum, sizeof(double)));
  CUDA_ERRCHK(cudaMalloc(&cuda_squares, sizeof(double)));
  CUDA_ERRCHK(cudaMalloc(&cuda_zeros, sizeof(double)));

  // Initialize arrays and doubles on GPU
  CUDA_ERRCHK(cudaMemcpy(cuda_x, x, sizeof(double) * N, cudaMemcpyHostToDevice));
  CUDA_ERRCHK(cudaMemcpy(cuda_sum, x_sum, sizeof(double), cudaMemcpyHostToDevice));
  CUDA_ERRCHK(cudaMemcpy(cuda_abssum, x_abssum, sizeof(double), cudaMemcpyHostToDevice));
  CUDA_ERRCHK(cudaMemcpy(cuda_squares, x_squares, sizeof(double), cudaMemcpyHostToDevice));
  CUDA_ERRCHK(cudaMemcpy(cuda_zeros, x_zeros, sizeof(double), cudaMemcpyHostToDevice));

  cuda_1c<<<((N+255)/256), 256>>>(N, cuda_x, cuda_sum, cuda_abssum, cuda_squares, cuda_zeros);
  
  CUDA_ERRCHK(cudaMemcpy(x_sum, cuda_sum, sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_ERRCHK(cudaMemcpy(x_abssum, cuda_abssum, sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_ERRCHK(cudaMemcpy(x_squares, cuda_squares, sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_ERRCHK(cudaMemcpy(x_zeros, cuda_zeros, sizeof(double), cudaMemcpyDeviceToHost));

  std::cout << "Vector X 1-Norm: " << *x_sum << std::endl;
  std::cout << "Vector X Absolute Sum: " << *x_abssum << std::endl;
  std::cout << "Vector X 2-Norm: " << *x_squares << std::endl;
  std::cout << "Vector X Zeros: " << *x_zeros << std::endl;

  //
  // Clean up
  //
  CUDA_ERRCHK(cudaFree(cuda_x));
  CUDA_ERRCHK(cudaFree(cuda_sum));
  CUDA_ERRCHK(cudaFree(cuda_abssum));
  CUDA_ERRCHK(cudaFree(cuda_squares));
  CUDA_ERRCHK(cudaFree(cuda_zeros));
  free(x);
  free(x_sum);
  free(x_abssum);
  free(x_squares);
  free(x_zeros);

  return EXIT_SUCCESS;
}