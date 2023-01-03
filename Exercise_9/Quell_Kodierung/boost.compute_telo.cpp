// specify use of OpenCL 1.2:
#define CL_TARGET_OPENCL_VERSION  120
#define CL_MINIMUM_OPENCL_VERSION 120

#include <vector>
#include <algorithm>
#include <iostream>

#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/functional/math.hpp>
    
namespace compute = boost::compute;

BOOST_COMPUTE_FUNCTION(float, boost_multiplication, (float x, float y),
                       {
                           return x * y;
                       });

BOOST_COMPUTE_FUNCTION(float, boost_addition, (float x, float y),
                       {
                           return x + y;
                       });

BOOST_COMPUTE_FUNCTION(float, boost_subtraction, (float x, float y),
                       {
                           return x - 2 * y;
                       });
int main(int argc, char *argv[])
{

    int i, N, N_max, gentxt;
    float dotP;
    N_max = 100;
    gentxt = 0;
    for (i=1; i<argc&&argv[i][0]=='-'; i++) {
    if (argv[i][1]=='N') i++, sscanf(argv[i],"%d",&N); // commandline arg -N for adjusting max. count, if none given N=100
    if (argv[i][1]=='g') i++, sscanf(argv[i], "%d", &gentxt); // commandline arg. -g for generating a txt, if none given, no .txt NOT IMPLEMENTED
    }

    // get default device and setup context
    compute::device device = compute::system::default_device();
    compute::context context(device);
    compute::command_queue queue(context, device);
    
    // generate the 3 vectors: x=(1,1,...,1), y=(2,2,...,2) and dotp
    std::vector<float> host_vector_x(N_max);
    std::vector<float> host_vector_y(N_max);
    std::vector<float> host_vector_prod(N_max);

    std::fill(host_vector_x.begin(), host_vector_x.end(), 1.0);
    std::fill(host_vector_y.begin(), host_vector_y.end(), 2.0);
    
    // create the 3 vectors on the device
    compute::vector<float> device_vector_x(host_vector_x.size(), context);
    compute::vector<float> device_vector_y(host_vector_y.size(), context);
    compute::vector<float> device_vector_prod(host_vector_prod.size(), context);

    // transfer data from the host to the device
    compute::copy(host_vector_x.begin(), host_vector_x.end(), device_vector_x.begin(), queue);
    compute::copy(host_vector_y.begin(), host_vector_y.end(), device_vector_y.begin(), queue);
    
    //calculate x+y within x
    compute::transform(
        device_vector_x.begin(),
        device_vector_x.end(),
        device_vector_y.begin(),
        device_vector_x.begin(),
        boost_addition,
        queue
    );

    //calculate x-2*y within y
    compute::transform(
        device_vector_x.begin(),
        device_vector_x.end(),
        device_vector_y.begin(),
        device_vector_y.begin(),
        boost_subtraction,
        queue
    );
    //after these two steps, vector x should have x_i + y_i in every entry, and vector y should have x_i - y_i in every entry and one can calculated the product in the thrid vector.

    //calculate x_i * y_i
    compute::transform(
        device_vector_x.begin(),
        device_vector_x.end(),
        device_vector_y.begin(),
        device_vector_prod.begin(),
        boost_multiplication,
        queue
    );
    //copy values back to the host
    compute::copy(
        device_vector_prod.begin(), device_vector_prod.end(), host_vector_prod.begin(), queue
    );

    dotP = std::accumulate(device_vector_prod.begin(), device_vector_prod.end(), 0);
    std::cout << "<x+y, x-y> = " << dotP << " with N = " << N_max << std::endl;
    
    return 0;
}