#include "timer.hpp"
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <vector>

__global__ void cuda_5_1c(double *x, double *y, double *z, int N)
{
  unsigned int total_threads = blockDim.x * gridDim.x;
	unsigned int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (unsigned int i = global_tid; i<N; i += total_threads) {
		z[i] = x[i] + y[i];
	}

}

double findMedian(std::vector<double> a,
                  int n)
{
  if (n % 2 == 0) {
      std::nth_element(a.begin(),
                  a.begin() + n / 2,
                  a.end());
      std::nth_element(a.begin(),
                  a.begin() + (n - 1) / 2,
                  a.end());
      return (double)(a[(n - 1) / 2]
                      + a[n / 2])
             / 2.0;
  }
  else {
      std::nth_element(a.begin(),
                  a.begin() + n / 2,
                  a.end());
      return (double)a[n / 2];
    }
}

void fillVectorWithIndices(double *V, int N){
  for (size_t i=0; i<N; ++i){
    V[i] = i;
  }
}

int main() {

int median_int = 10; // iterations to build median for timings
int N_max = 8; // 10 to the N_max'th power is the largest vector for the bandwidth measurement

Timer timer;
std::vector<double> timings, median_timings, N_vec;
for (size_t i=1; i<=N_max; ++i){
  N_vec.push_back(pow(10,i));
}

double *x, *y, *z, *gpu_x, *gpu_y, *gpu_z;

//First For Loop to obtain Timings for varying N
for (size_t i=0; i<N_max; ++i){
  x = (double*)malloc(N_vec[i]*sizeof(double));
  y = (double*)malloc(N_vec[i]*sizeof(double));
	z = (double*)malloc(N_vec[i]*sizeof(double));
  fillVectorWithIndices(x,N_vec[i]);
  fillVectorWithIndices(y,N_vec[i]);
  fillVectorWithIndices(z,N_vec[i]);
  std::cout << x << std::endl;
}

return EXIT_SUCCESS;
}

