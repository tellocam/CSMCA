#include "timer.hpp"
#include "cuda_errchk.hpp"
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <vector>

// findMedian function for any vector lenghts, source geeksforgeeks.com
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


int main() {
Timer timer;

//Allocate and Initialize CPU Doubles
int N = 1e6;
double *x = (double *)malloc(sizeof(double));
double *x_N = (double *)malloc(sizeof(double) * N);
*x = 1;
std::fill(x_N, x_N + N, 1);
double *cuda_x, *cuda_x_N;
cudaMalloc(&cuda_x, sizeof(double));
cudaMalloc(&cuda_x_N, sizeof(double) * N);

// Task 1a start - synchronize and reset timer for for-loop to build time average
std::vector<double> timings_double(100, 0.0);
std::vector<double> timings_N_doubles(100, 0.0);

for (size_t i=0; i<=100; ++i){
  cudaDeviceSynchronize();
  timer.reset();
  cudaMemcpy(cuda_x, x, sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  timings_double[i] = timer.get();
  
}
double timing_double = findMedian(timings_double, 100);


for (size_t i=0; i<=100; ++i){
  cudaDeviceSynchronize();
  timer.reset();
  cudaMemcpy(cuda_x_N, x_N, sizeof(double)*N, cudaMemcpyHostToDevice);
  timings_N_doubles[i] = timer.get();
  cudaDeviceSynchronize();
}
double timing_N_doubles = findMedian(timings_N_doubles, 100);

double latency;
latency = (timing_N_doubles - timing_double*N)/(1-N);

std::cout << "PCI Express gen3 Latency: " << latency << std::endl;

return EXIT_SUCCESS;
}

