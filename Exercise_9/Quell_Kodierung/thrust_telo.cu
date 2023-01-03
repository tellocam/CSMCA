#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <cstdlib>
#include <numeric>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <iostream>

int main(int argc, char *argv[]) {

    int i, N, N_max, gentxt;
    float dotP;
    N_max = 100;
    gentxt = 0;
    for (i=1; i<argc&&argv[i][0]=='-'; i++) {
    if (argv[i][1]=='N') i++, sscanf(argv[i],"%d",&N); // commandline arg -N for adjusting max. count, if none given N=100
    if (argv[i][1]=='g') i++, sscanf(argv[i], "%d", &gentxt); // commandline arg. -g for generating a txt, if none given, no .txt NOT IMPLEMENTED
    }

  // generate vectors on host of lenght N_max
  thrust::host_vector<float> h_x(N_max);
  thrust::host_vector<float> h_y(N_max);
  thrust::host_vector<float> h_prod1(N_max);
  thrust::host_vector<float> h_prod2(N_max);
  thrust::fill(h_x.begin(), h_x.end(), 1);
  thrust::fill(h_y.begin(), h_y.end(), 2);
  thrust::fill(h_prod1.begin(), h_prod1.end(), 0);
  thrust::fill(h_prod2.begin(), h_prod2.end(), 0);

  // transfer data to the device
  thrust::device_vector<float> d_x = h_x;
  thrust::device_vector<float> d_y = h_y;
  thrust::device_vector<float> d_prod1 = h_prod1;
  thrust::device_vector<float> d_prod2 = h_prod2;

  thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), d_prod1.begin(), thrust::plus<float>());
  thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), d_prod2.begin(), thrust::minus<float>());
  thrust::transform(d_prod1.begin(), d_prod1.end(), d_prod2.begin(),d_prod1.begin(), thrust::multiplies<float>());


  // transfer data back to host
  thrust::copy(d_prod1.begin(), d_prod1.end(), h_prod1.begin());
  dotP = std::accumulate(d_prod1.begin(), d_prod1.end(), 0);

  std::cout << "<x+y, x-y> = " << dotP << " with N = " << N_max << std::endl;

  return 0;
}