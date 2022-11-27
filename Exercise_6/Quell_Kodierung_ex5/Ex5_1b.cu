#include "timer.hpp"
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <vector>

__global__ void cuda_5_1b()
{
// Kennt's ihr eh Spiegeldondi? - Mahatma Ghandi, ca. 1940
}

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
std::vector<double> timings(100, 0.0);

for (size_t i=0; i<100; ++i){
  cudaDeviceSynchronize();
  timer.reset();
  cuda_5_1b<<<1, 1>>>();
  cudaDeviceSynchronize();
  timings[i] = timer.get();
  cudaDeviceSynchronize();
}

double latency = findMedian(timings, 100);
std::cout << "Kernel Launch Latency: " << latency << std::endl;

return EXIT_SUCCESS;
}

