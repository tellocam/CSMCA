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

BOOST_COMPUTE_FUNCTION(double, multiplication, (double x, double y),
                       {
                           return x * y;
                       });
    
int main()
{
    // get default device and setup context
    compute::device device = compute::system::default_device();
    compute::context context(device);
    compute::command_queue queue(context, device);
    
    // generate the 3 vectors: x=(1,1,...,1), y=(2,2,...,2) and dotp
    std::vector<float> host_vector_x(10000);
    std::vector<float> host_vector_y(10000);
    std::vector<float> host_vector_dotp(10000);

    std::generate(host_vector_x.begin(), host_vector_x.end(), 1);
    std::generate(host_vector_y.begin(), host_vector_y.end(), 2);
    
    // create the 3 vectors on the device
    compute::vector<float> device_vector_x(host_vector_x.size(), context);
    compute::vector<float> device_vector_y(host_vector_y.size(), context);
    compute::vector<float> device_vector_dotp(host_vector_dotp.size(), context);

    // transfer data from the host to the device
    compute::copy(host_vector_x.begin(), host_vector_x.end(), device_vector_x.begin(), queue);
    compute::copy(host_vector_y.begin(), host_vector_y.end(), device_vector_y.begin(), queue);
    
    // calculate the square-root of each element in-place
    compute::transform(
        device_vector.begin(),
        device_vector.end(),
        device_vector.begin(),
        compute::sqrt<float>(),
        queue
    );
    
    // copy values back to the host
    compute::copy(
        device_vector.begin(), device_vector.end(), host_vector.begin(), queue
    );

    std::cout << host_vector[0] << ", " << host_vector[1] << ", ..." << std::endl;
    
    return 0;
}