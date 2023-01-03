
#include <iostream>
#include <cstdlib>
#include <numeric>
#include <vector>
#include <cstdlib>
#include <algorithm>

#define VIENNACL_WITH_CUDA

#include "viennacl/vector.hpp"
#include "viennacl/linalg/inner_prod.hpp"


int main(int argc, char *argv[]) {
  int i, N, N_max, gentxt;
  double dotP;
  N_max = 100;
  gentxt = 0;
  for (i=1; i<argc&&argv[i][0]=='-'; i++) {
  if (argv[i][1]=='N') i++, sscanf(argv[i],"%d",&N); // commandline arg -N for adjusting max. count, if none given N=100
  if (argv[i][1]=='g') i++, sscanf(argv[i], "%d", &gentxt); // commandline arg. -g for generating a txt, if none given, no .txt NOT IMPLEMENTED
  }

  viennacl::vector<double> x = viennacl::scalar_vector<double>(N_max, 1.0);
  viennacl::vector<double> y = viennacl::scalar_vector<double>(N_max, 2.0);

  viennacl::vector<double> f1 = x + y;
  viennacl::vector<double> f2 = x - y;
  viennacl::vector<double> p = viennacl::linalg::element_prod(f1,f2);

  dotP = std::accumulate(p.begin(), p.end(), 0);
  std::cout << "<x+y, x-y> = " << dotP << " with N = " << N_max << std::endl;

  return EXIT_SUCCESS;
}
