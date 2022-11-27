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

int median_int = 20; // iterations to build median for timings
int N_max = 28; // 2 to the N_max'th power is the largest vector for the bandwidth measurement

Timer timer;
std::vector<double> timings, peak_bw, N_vec;
for (size_t i=0; i<=N_max; ++i){
  N_vec.push_back(pow(2,i));
}

double *x, *y, *z, *gpu_x, *gpu_y, *gpu_z;

std::cout << "Effective Bandwidth for varying Vector size addition: " << std::endl;
std::cout << "N, " << "GB/s" << std::endl;


//First For Loop to obtain Timings for varying N
for (size_t i=0; i<N_max; ++i){

  //prepare data on GPU and CPU for Kernel Submission
  x = (double*)malloc(N_vec[i]*sizeof(double));
  y = (double*)malloc(N_vec[i]*sizeof(double));
	z = (double*)malloc(N_vec[i]*sizeof(double));
  fillVectorWithIndices(x,N_vec[i]);
  fillVectorWithIndices(y,N_vec[i]);
  // std::fill(x, x + (int)N_vec[i], 1);
  // std::fill(y, y + (int)N_vec[i], 1);
  std::fill(z, z + (int)N_vec[i], 0);
  cudaMalloc(&gpu_x, N_vec[i]*sizeof(double));
  cudaMalloc(&gpu_y, N_vec[i]*sizeof(double));
  cudaMalloc(&gpu_z, N_vec[i]*sizeof(double));
  cudaMemcpy(gpu_x, x, N_vec[i]*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_y, x, N_vec[i]*sizeof(double), cudaMemcpyHostToDevice);
  // cudaMemcpy(gpu_z, z, N_vec[i]*sizeof(double), cudaMemcpyHostToDevice);


  //First nested for Loop to obtain median of N_max[i]
  for(int j=0; j < median_int; j++){
    cudaDeviceSynchronize();
    timer.reset();
    cuda_5_1c<<<((N_vec[i]+255)/256), 256>>>(gpu_x, gpu_y, gpu_z, N_vec[i]);
    cudaDeviceSynchronize();
    timings.push_back(timer.get());
  }

  cudaMemcpy(z, gpu_z, N_vec[i]*sizeof(double), cudaMemcpyDeviceToHost);

  // obtain median timing for N_vec[i] from all timings_int iterations and clear timings vector for N_vec[i+1]
  // 3 * floor((N - k_values[i])) * sizeof(double) * pow(10, -9) / findMedian(exec_timings, 10);
  peak_bw.push_back((3*N_vec[i]*sizeof(double)*pow(10,-9))/findMedian(timings, median_int));
  timings.clear();

  // print N[i] and peak_bw[i] for copying it into csv later on :-)

  std::cout << N_vec[i] << ", " << peak_bw[i] << std::endl;

  cudaFree(gpu_x);
  cudaFree(gpu_y);
  cudaFree(gpu_z);
  free(x);
  free(y);
  free(z);

}
return EXIT_SUCCESS;
}

