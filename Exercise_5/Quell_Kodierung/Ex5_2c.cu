#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include "poisson2d.hpp"
#include "timer.hpp"


//Kernel from Lecture slides 5b.

__global void csr_matvec(int N,
int *rowoffsets, int *colindices, double *values, 
double const *x, double *y) {
for (int row = blockDim.x * blockIdx.x + threadIdx.x;
row < N;
row += gridDim.x * blockDim.x) {
double val = 0;
for (int jj = rowoffsets[i]; jj < rowoffsets[i+1]; ++jj) {
val += values[jj] * x[colindices[jj]];
}
y[row] = val;
}
}

// Same Kernels from Exercise before! copy paste basically..
__global__ void cuda_dotp(int N, double *x, double *y, double *result)
{
  __shared__ double shared_mem[1024];
  double dot = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
  {
    dot += x[i] * y[i];
  }
  shared_mem[threadIdx.x] = dot;
  for (int k = blockDim.x / 2; k > 0; k /= 2)
  {
    __syncthreads();
    if (threadIdx.x < k)
    {
      shared_mem[threadIdx.x] += shared_mem[threadIdx.x + k];
    }
  }
  if (threadIdx.x == 0)
  {
    atomicAdd(result, shared_mem[0]);
  }
}

__global__ void kernel_line7(int N, double *x, double *y, double a)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
  {
    x[i] += a * y[i];
  }
}

__global__ void kernel_line8(int N, double *x, double *y, double a)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
  {
    x[i] -= a * y[i];
  }
}

__global__ void kernel_line12(int N, double *x, double *y, double b)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
  {
    x[i] = y[i] + b * x[i];
  }
}