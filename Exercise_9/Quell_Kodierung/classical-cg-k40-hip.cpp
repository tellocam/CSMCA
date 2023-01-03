#include "poisson2d.hpp"
#include "timer.hpp"
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include "hip/hip_runtime.h"

#define HIP_ASSERT(x) (assert((x)==hipSuccess))
 
// y = A * x
__global__ void cuda_csr_matvec_product(int N, int *csr_rowoffsets,
                                        int *csr_colindices, double *csr_values,
                                        double *x, double *y)
{
  for (int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; i < N; i += hipBlockDim_x * hipGridDim_x) {
    double sum = 0;
    for (int k = csr_rowoffsets[i]; k < csr_rowoffsets[i + 1]; k++) {
      sum += csr_values[k] * x[csr_colindices[k]];
    }
    y[i] = sum;
  }
}
 
// x <- x + alpha * y
__global__ void cuda_vecadd(int N, double *x, double *y, double alpha)
{
  for (int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; i < N; i += hipBlockDim_x * hipGridDim_x)
    x[i] += alpha * y[i];
}
 
// x <- y + alpha * x
__global__ void cuda_vecadd2(int N, double *x, double *y, double alpha)
{
  for (int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; i < N; i += hipBlockDim_x * hipGridDim_x)
    x[i] = y[i] + alpha * x[i];
}
 
// result = (x, y)
__global__ void cuda_dot_product(int N, double *x, double *y, double *result_partial)
{
  __shared__ double shared_mem[512];
 
  double dot = 0;
  for (int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; i < N; i += hipBlockDim_x * hipGridDim_x) {
    dot += x[i] * y[i];
  }
 
  shared_mem[hipThreadIdx_x] = dot;
  for (int k = hipBlockDim_x / 2; k > 0; k /= 2) {
    __syncthreads();
    if (hipThreadIdx_x < k) {
      shared_mem[hipThreadIdx_x] += shared_mem[hipThreadIdx_x + k];
    }
  }
 
  if (hipThreadIdx_x == 0) result_partial[hipBlockIdx_x] = shared_mem[0];
}
 
 
 
/** Implementation of the conjugate gradient algorithm.
 *
 *  The control flow is handled by the CPU.
 *  Only the individual operations (vector updates, dot products, sparse
 * matrix-vector product) are transferred to CUDA kernels.
 *
 *  The temporary arrays p, r, and Ap need to be allocated on the GPU for use
 * with CUDA. Modify as you see fit.
 */
void conjugate_gradient(int N, // number of unknows
                        int *csr_rowoffsets, int *csr_colindices,
                        double *csr_values, double *rhs, double *solution)
//, double *init_guess)   // feel free to add a nonzero initial guess as needed
{
  // initialize timer
  Timer timer;
  double partial[256];
 
  // clear solution vector (it may contain garbage values):
  std::fill(solution, solution + N, 0);
 
  // initialize work vectors:
  double alpha, beta;
  double *cuda_solution, *cuda_p, *cuda_r, *cuda_Ap, *cuda_partial;
  hipMalloc(&cuda_p, sizeof(double) * N);
  hipMalloc(&cuda_r, sizeof(double) * N);
  hipMalloc(&cuda_Ap, sizeof(double) * N);
  hipMalloc(&cuda_solution, sizeof(double) * N);
  hipMalloc(&cuda_partial, sizeof(double) * 256);
 
  hipMemcpy(cuda_p, rhs, sizeof(double) * N, hipMemcpyHostToDevice);
  hipMemcpy(cuda_r, rhs, sizeof(double) * N, hipMemcpyHostToDevice);
  hipMemcpy(cuda_solution, solution, sizeof(double) * N, hipMemcpyHostToDevice);
 
  double residual_norm_squared = 0;
  //cuda_dot_product<<<256, 256>>>(N, cuda_r, cuda_r, cuda_partial);
  // IMPLEMENT HIP KERNEL CALL HERE!
  hipLaunchKernelGGL(cuda_dot_product, dim3(265), dim3(265), 0, 0,
                      N, cuda_r, cuda_r, cuda_partial);

  hipMemcpy(partial, cuda_partial, sizeof(double) * 256, hipMemcpyDeviceToHost);
  residual_norm_squared = 0;
  for (size_t i=0; i<256; ++i) residual_norm_squared += partial[i];
 
  double initial_residual_squared = residual_norm_squared;
 
  int iters = 0;
  hipDeviceSynchronize();
  timer.reset();
  while (1) {
 
    // line 4: A*p:
    //cuda_csr_matvec_product<<<512, 512>>>(N, csr_rowoffsets, csr_colindices, csr_values, cuda_p, cuda_Ap);
    // IMPLEMENT HIP KERNEL CALL HERE!
    hipLaunchKernelGGL(cuda_csr_matvec_product, dim3(512), dim3(512), 0, 0,
                        N, csr_rowoffsets, csr_colindices, csr_values, cuda_p, cuda_Ap);
    // lines 5,6:
    //cuda_dot_product<<<256, 256>>>(N, cuda_p, cuda_Ap, cuda_partial);
    // IMPLEMENT HIP KERNEL CALL HERE!
    hipLaunchKernelGGL(cuda_dot_product, dim3(256), dim3(256), 0, 0,
                        N, cuda_p, cuda_Ap, cuda_partial);

    hipMemcpy(partial, cuda_partial, sizeof(double) * 256, hipMemcpyDeviceToHost);
    alpha = 0;
    for (size_t i=0; i<256; ++i) alpha += partial[i];
    alpha = residual_norm_squared / alpha;
 
    // line 7:
    //cuda_vecadd<<<512, 512>>>(N, cuda_solution, cuda_p, alpha);
    // IMPLEMENT HIP KERNEL CALL HERE!
    hipLaunchKernelGGL(cuda_vecadd, dim3(512), dim3(512), 0, 0,
                        N, cuda_solution, cuda_p, alpha);
 
    // line 8:
    //cuda_vecadd<<<512, 512>>>(N, cuda_r, cuda_Ap, -alpha);
    // IMPLEMENT HIP KERNEL CALL HERE!
    hipLaunchKernelGGL(cuda_vecadd, dim3(512), dim3(512), 0, 0,
                        N, cuda_r, cuda_Ap, -alpha);

    // line 9:
    beta = residual_norm_squared;
    //cuda_dot_product<<<256, 256>>>(N, cuda_r, cuda_r, cuda_partial);
    // IMPLEMENT HIP KERNEL CALL HERE!
    hipLaunchKernelGGL(cuda_dot_product, dim3(265), dim3(265), 0, 0,
                        N, cuda_r, cuda_r, cuda_partial);

    hipMemcpy(partial, cuda_partial, sizeof(double) * 256, hipMemcpyDeviceToHost);
    residual_norm_squared = 0;
    for (size_t i=0; i<256; ++i) residual_norm_squared += partial[i];
 
    // line 10:
    if (std::sqrt(residual_norm_squared / initial_residual_squared) < 1e-6) {
      break;
    }
 
    // line 11:
    beta = residual_norm_squared / beta;
 
    // line 12:
    //cuda_vecadd2<<<512, 512>>>(N, cuda_p, cuda_r, beta);
    // IMPLEMENT HIP KERNEL CALL HERE!
    hipLaunchKernelGGL(cuda_vecadd2, dim3(512), dim3(512), 0, 0,
                        N, cuda_p, cuda_r, beta);
 
    if (iters > 10000)
      break; // solver didn't converge
    ++iters;
  }
  hipMemcpy(solution, cuda_solution, sizeof(double) * N, hipMemcpyDeviceToHost);
 
  hipDeviceSynchronize();
  std::cout << "Time elapsed: " << timer.get() << " (" << timer.get() / iters << " per iteration)" << std::endl;
 
  if (iters > 10000)
    std::cout << "Conjugate Gradient did NOT converge within 10000 iterations"
              << std::endl;
  else
    std::cout << "Conjugate Gradient converged in " << iters << " iterations."
              << std::endl;
 
  hipFree(cuda_p);
  hipFree(cuda_r);
  hipFree(cuda_Ap);
  hipFree(cuda_solution);
  hipFree(cuda_partial);
}
 
/** Solve a system with `points_per_direction * points_per_direction` unknowns
 */
void solve_system(int points_per_direction) {
 
  int N = points_per_direction *
          points_per_direction; // number of unknows to solve for
 
  std::cout << "Solving Ax=b with " << N << " unknowns." << std::endl;
 
  //
  // Allocate CSR arrays.
  //
  // Note: Usually one does not know the number of nonzeros in the system matrix
  // a-priori.
  //       For this exercise, however, we know that there are at most 5 nonzeros
  //       per row in the system matrix, so we can allocate accordingly.
  //
  int *csr_rowoffsets = (int *)malloc(sizeof(double) * (N + 1));
  int *csr_colindices = (int *)malloc(sizeof(double) * 5 * N);
  double *csr_values = (double *)malloc(sizeof(double) * 5 * N);
 
  int *cuda_csr_rowoffsets, *cuda_csr_colindices;
  double *cuda_csr_values;
  //
  // fill CSR matrix with values
  //
  generate_fdm_laplace(points_per_direction, csr_rowoffsets, csr_colindices,
                       csr_values);
 
  //
  // Allocate solution vector and right hand side:
  //
  double *solution = (double *)malloc(sizeof(double) * N);
  double *rhs = (double *)malloc(sizeof(double) * N);
  std::fill(rhs, rhs + N, 1);
 
  //
  // Allocate CUDA-arrays //
  //
  hipMalloc(&cuda_csr_rowoffsets, sizeof(double) * (N + 1));
  hipMalloc(&cuda_csr_colindices, sizeof(double) * 5 * N);
  hipMalloc(&cuda_csr_values, sizeof(double) * 5 * N);
  hipMemcpy(cuda_csr_rowoffsets, csr_rowoffsets, sizeof(double) * (N + 1), hipMemcpyHostToDevice);
  hipMemcpy(cuda_csr_colindices, csr_colindices, sizeof(double) * 5 * N,   hipMemcpyHostToDevice);
  hipMemcpy(cuda_csr_values,     csr_values,     sizeof(double) * 5 * N,   hipMemcpyHostToDevice);
 
  //
  // Call Conjugate Gradient implementation with GPU arrays
  //
  conjugate_gradient(N, cuda_csr_rowoffsets, cuda_csr_colindices, cuda_csr_values, rhs, solution);
 
  //
  // Check for convergence:
  //
  double residual_norm = relative_residual(N, csr_rowoffsets, csr_colindices, csr_values, rhs, solution);
  std::cout << "Relative residual norm: " << residual_norm
            << " (should be smaller than 1e-6)" << std::endl;
 
  hipFree(cuda_csr_rowoffsets);
  hipFree(cuda_csr_colindices);
  hipFree(cuda_csr_values);
  free(solution);
  free(rhs);
  free(csr_rowoffsets);
  free(csr_colindices);
  free(csr_values);
}
 
int main() {
 
  solve_system(1000); // solves a system with 100*100 unknowns
 
  return EXIT_SUCCESS;
}
 
