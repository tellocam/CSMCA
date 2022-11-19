#include "timer.hpp"
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <vector>

__global__ void cuda_5_1d(double *atomic_add_result)
{
	atomicAdd(atomic_add_result, threadIdx.x);
}

double findMedian(std::vector<double> a, int n){

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

    int median_int = 100; // iterations to build median for timings
    int N_max = 30; // 2 to the N_max'th power is the largest vector for the bandwidth measurement

    Timer timer;
    std::vector<double> timings, median_timings, N_vec;

    for (size_t i=10; i<=N_max; ++i){
    N_vec.push_back(pow(2,i));
    }

    double *x, *x_gpu;
    x = (double*)malloc(sizeof(double));
    x = 0;

    cudaMalloc(&x_gpu, 1*sizeof(double));
    cudaMemcpy(x_gpu, x, sizeof(double), cudaMemcpyHostToDevice);

   
    std::cout << "Threads Launched, " << "AtomicAdd() / s" << std::endl;
    
    for (size_t i=0; i<N_max; ++i){

        for(int j=0; j < median_int; j++){
            cudaDeviceSynchronize();
            timer.reset();
            cuda_5_1d<<<(int)N_vec[i], 1>>>(x_gpu);
            cudaDeviceSynchronize();
            timings.push_back(timer.get());
        }

    // obtain median timing for N_vec[i] from all timings_int iterations and clear timings vector for N_vec[i+1]
    // 3 * floor((N - k_values[i])) * sizeof(double) * pow(10, -9) / findMedian(exec_timings, 10);
    median_timings.push_back(findMedian(timings, median_int));
    timings.clear();

    std::cout << N_vec[i] << ", " << N_vec[i]/median_timings[i] << std::endl;
    }

    return EXIT_SUCCESS;
}


