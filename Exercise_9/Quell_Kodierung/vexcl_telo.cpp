// The following three defines are necessary to pick the correct OpenCL version on the machine:
#define VEXCL_HAVE_OPENCL_HPP
#define CL_HPP_TARGET_OPENCL_VERSION  120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#include <cstdlib>
#include <numeric>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <iostream>

#include <stdexcept>
#include <vexcl/vexcl.hpp>

int main(int argc, char *argv[]) {
    int i, N, N_max, gentxt;
    double dotP;
    N_max = 100;
    gentxt = 0;
    for (i=1; i<argc&&argv[i][0]=='-'; i++) {
    if (argv[i][1]=='N') i++, sscanf(argv[i],"%d",&N); // commandline arg -N for adjusting max. count, if none given N=100
    if (argv[i][1]=='g') i++, sscanf(argv[i], "%d", &gentxt); // commandline arg. -g for generating a txt, if none given, no .txt NOT IMPLEMENTED
    }

    vex::Context ctx(vex::Filter::GPU && vex::Filter::DoublePrecision);

    std::cout << ctx << std::endl; // print list of selected devices

    std::vector<double> x(N_max, 1.0), y(N_max, 2.0), prod(N_max, 0);

    vex::vector<double> X(ctx, x);
    vex::vector<double> Y(ctx, y);

    vex::vector<double> F1 = X + Y;
    vex::vector<double> F2 = X - Y;
    vex::vector<double> P = F1 * F2;

    dotP = std::accumulate(P.begin(), P.end(), 0);
    std::cout << "<x+y, x-y> = " << dotP << " with N = " << N_max << std::endl;

    return 0; 
    }