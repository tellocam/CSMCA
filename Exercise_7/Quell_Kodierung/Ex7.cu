#include "poisson2d.hpp"
#include "timer.hpp"
#include <algorithm>
#include <iostream>
#include <stdio.h>
 
// y = A * x
__global__ void cuda_csr_matvec_product(int N, int *csr_rowoffsets,
                                        int *csr_colindices, double *csr_values,
                                        double *x, double *y)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
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
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    x[i] += alpha * y[i];
}
 
// x <- y + alpha * x
__global__ void cuda_vecadd2(int N, double *x, double *y, double alpha)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    x[i] = y[i] + alpha * x[i];
}
 
// result = (x, y)
__global__ void cuda_dot_product(int N, double *x, double *y, double *result_partial)
{
  __shared__ double shared_mem[512];
 
  double dot = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    dot += x[i] * y[i];
  }
 
  shared_mem[threadIdx.x] = dot;
  for (int k = blockDim.x / 2; k > 0; k /= 2) {
    __syncthreads();
    if (threadIdx.x < k) {
      shared_mem[threadIdx.x] += shared_mem[threadIdx.x + k];
    }
  }
 
  if (threadIdx.x == 0) result_partial[blockIdx.x] = shared_mem[0];
}

// Pipelined CG Blue algorithm part (Algorithm-line 2 -line  4)
__global__ void cuda_blue(int N, double *x, double *p, double *Ap, double *r, double *r_ip, double Alpha, double Beta)
{
  __shared__ double shared_memory[512];
  double partial_dot_product = 0;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x){
    // Same Procedure as in Exercise 4 for x, r and p.
    double p_thread = p[i];
    double Ap_thread = Ap[i];
    double r_thread = r[i] - Alpha * Ap_thread;

    x[i] += Alpha * p_thread;
    r[i] = r_thread;
    p[i] = r_thread + Beta * p_thread;
    partial_dot_product += r_thread *r_thread;
  }
  // Now one uses the introduced shared memory to avoid using the atomicAdd() when computing the dot-product <r,r>
  shared_memory[threadIdx.x] = partial_dot_product;
  for (int j = blockDim.x / 2; j > 0; j /= 2) {
    __syncthreads();
    if (threadIdx.x < j) {
      shared_memory[threadIdx.x] += shared_memory[threadIdx.x + j];
  }
  
  if (threadIdx.x == 0) r_ip[blockIdx.x] = shared_memory[0];
  }
}

// Pipelined CG Red algorithm part (Algorithm-line 5 and 6) , bw_dot_ApAp and bw_dot_pAp are blockwise and need to be summed up on the CPU!
__global__ void cuda_red(int N, int *csr_rowoffsets, int *csr_colindices, double *csr_values, double *p, double *Ap, double *bw_dot_ApAp, double *bw_dot_pAp)
{
  __shared__ double sm_ApAp[512];
  __shared__ double sm_pAp[512];
  double dot_ApAp = 0;
  double dot_pAp = 0;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    double sum = 0;
    for (int k = csr_rowoffsets[i]; k < csr_rowoffsets[i + 1]; k++) {
      sum += csr_values[k] * p[csr_colindices[k]];
    }
    Ap[i] = sum;
    dot_ApAp += sum*sum;
    dot_pAp += sum * p[i];
  }

  sm_ApAp[threadIdx.x] = dot_ApAp;
  sm_pAp[threadIdx.x] = dot_pAp;
  for (int k = blockDim.x / 2; k > 0; k /= 2) {
    __syncthreads();
    if (threadIdx.x < k) {
      sm_ApAp[threadIdx.x] += sm_ApAp[threadIdx.x + k];
      sm_pAp[threadIdx.x] += sm_pAp[threadIdx.x + k];
    }
  }
  if (threadIdx.x == 0)
  {
    bw_dot_ApAp[blockIdx.x] = sm_ApAp[0];
    bw_dot_pAp[blockIdx.x] = sm_pAp[0];
  }


}
// This is the pipelined CG function with the adaptations for Exercise 7 where only 2 kernel calls are made
// Still before going into the while-loop where the CG iterations are done, the old Kernels are used to calculate
// initial alpha_0, beta_0 and Ap_0.
void conjugate_gradient_pipelined(int N, // number of unknows
                        int *csr_rowoffsets, int *csr_colindices,
                        double *csr_values, double *rhs, double *solution)
//, double *init_guess)   // feel free to add a nonzero initial guess as needed
{
  // initialize timer
  Timer timer;

  double blocks_lnch = 512; // blocks Launched per Kernel - also result size of blockwise inner product arrays
  double thrds_block = 512; // threads per block launched

    // Initialize array for blockwise inner products <r,r> , <p,Ap> and <Ap,Ap>. bwip = blockwise inner product
  double bwip_rr[(int)blocks_lnch], bwip_pAp[(int)blocks_lnch], bwip_ApAp[(int)blocks_lnch];
 
  // clear solution vector (it may contain garbage values):
  std::fill(solution, solution + N, 0); // Choose x_0 = 0
 
  // initialize work scalars s.a coeff's and Inner Products as well as Vectors.
  double alpha, beta, ip_ApAp, ip_pAp, ip_rr;
  double *cuda_solution, *cuda_p, *cuda_r, *cuda_Ap, *cuda_ip_pAp, *cuda_ip_ApAp, *cuda_ip_rr;
  cudaMalloc(&cuda_p, sizeof(double) * N);
  cudaMalloc(&cuda_r, sizeof(double) * N);
  cudaMalloc(&cuda_Ap, sizeof(double) * N);
  cudaMalloc(&cuda_solution, sizeof(double) * N);
  cudaMalloc(&cuda_ip_pAp, sizeof(double) * blocks_lnch);
  cudaMalloc(&cuda_ip_ApAp, sizeof(double) * blocks_lnch);
  cudaMalloc(&cuda_ip_rr, sizeof(double) * blocks_lnch);
 
  cudaMemcpy(cuda_p, rhs, sizeof(double) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_r, rhs, sizeof(double) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_solution, solution, sizeof(double) * N, cudaMemcpyHostToDevice);

  // Compute <r_0, r_0>,
  
  cuda_dot_product<<<blocks_lnch, thrds_block>>>(N, cuda_r, cuda_r, cuda_ip_rr);
  cudaMemcpy(bwip_rr, cuda_ip_rr, sizeof(double) * blocks_lnch, cudaMemcpyDeviceToHost);
  ip_rr = 0;
  for (size_t i=0; i<blocks_lnch; ++i) ip_rr += bwip_rr[i];
  double initial_residual_squared = ip_rr;

  // Compute Ap_0
  cuda_csr_matvec_product<<<blocks_lnch, thrds_block>>>(N, csr_rowoffsets, csr_colindices, csr_values, cuda_p, cuda_Ap);
  // Compute <p_0, Ap_0> (Blockwise) and retrieve it Blockwise and sum up on CPU
  cuda_dot_product<<<blocks_lnch, thrds_block>>>(N, cuda_p, cuda_Ap, cuda_ip_pAp);
  cudaMemcpy(bwip_pAp, cuda_ip_pAp, sizeof(double) * blocks_lnch, cudaMemcpyDeviceToHost);
  ip_pAp = 0;
  for (size_t i=0; i<blocks_lnch; ++i) ip_pAp += bwip_pAp[i];

  //Compute <Ap_0, Ap_0> (Blockwise) and retrieve it Blockwise and sum up on CPU
  cuda_dot_product<<<blocks_lnch, thrds_block>>>(N, cuda_Ap, cuda_Ap, cuda_ip_ApAp);
  cudaMemcpy(bwip_ApAp, cuda_ip_ApAp, sizeof(double) * blocks_lnch, cudaMemcpyDeviceToHost);
  ip_ApAp = 0;
  for (size_t i=0; i<blocks_lnch; ++i) ip_ApAp += bwip_ApAp[i];

  // Compute alpha, beta according to Algorithm. 
  alpha = ip_rr / ip_pAp;
  beta = (alpha*alpha*ip_ApAp - ip_rr) / ip_rr;

  // Here we have readily available alpha_0, beta_0, and Ap_0 to start with the iterations according to the algorithm.

  int iters = 0;
  cudaDeviceSynchronize();
  timer.reset();
  while (1) {
    // Line 2-4 and partial of line 6, The blue colored part in the algorithm:
    cuda_blue<<<blocks_lnch,thrds_block>>>(N, cuda_solution, cuda_p, cuda_Ap, cuda_r, cuda_ip_rr, alpha, beta);
    // cudaMemcopy for blockwise inner product <r_i, r_i> for CPU calculations of alpha and beta
    cudaMemcpy(bwip_rr, cuda_ip_rr, sizeof(double) * blocks_lnch, cudaMemcpyDeviceToHost);
    // Line 5 and 6, the red colored part in the algorithm
    cuda_red<<<blocks_lnch,thrds_block>>>(N, csr_rowoffsets, csr_colindices, csr_values, cuda_p, cuda_Ap, cuda_ip_ApAp, cuda_ip_pAp);
    // cudaMemcopy for blockwise inner product <Ap_i, Ap_i> and <p_i, Ap_i> for CPU calculations of alpha and beta
    cudaMemcpy(bwip_pAp, cuda_ip_pAp, sizeof(double) * blocks_lnch, cudaMemcpyDeviceToHost);
    cudaMemcpy(bwip_ApAp, cuda_ip_ApAp, sizeof(double) * blocks_lnch, cudaMemcpyDeviceToHost);
    // CPU summation of blockwise inner products.
    ip_rr = bwip_rr[0];
    ip_ApAp = bwip_ApAp[0];
    ip_pAp = bwip_pAp[0];
    for(size_t i=1; i<blocks_lnch; ++i)
    {
      ip_rr += bwip_rr[i];
      ip_ApAp += bwip_ApAp[i];
      ip_pAp += bwip_pAp[i];
    }
    //Check if convergence criterion is fulfilled.
    if (std::sqrt(ip_rr / initial_residual_squared) < 1e-6) {
      break;
    }
    // Computation of alpha and beta for next while iteration.
    alpha = ip_rr / ip_pAp;
    beta = ( alpha*alpha*ip_ApAp - ip_rr) / ip_rr;
 
    if (iters > 10000)
      break; // solver didn't converge
    ++iters;
  }
  cudaMemcpy(solution, cuda_solution, sizeof(double) * N, cudaMemcpyDeviceToHost);
 
  // cudaDeviceSynchronize();
  // std::cout << "Time elapsed: " << timer.get() << " (" << timer.get() / iters << " per iteration)" << std::endl;

  cudaDeviceSynchronize();
  std::cout << std::sqrt(N)<< ", " << timer.get() << ", " << iters << std::endl;
 
  if (iters > 10000)
    std::cout << "Conjugate Gradient did NOT converge within 10000 iterations"
              << std::endl;
  // else
  //   std::cout << "Conjugate Gradient converged in " << iters << " iterations."
  //             << std::endl;
 
  cudaFree(cuda_p);
  cudaFree(cuda_r);
  cudaFree(cuda_Ap);
  cudaFree(cuda_solution);
  cudaFree(cuda_ip_rr);
  cudaFree(cuda_ip_pAp);
  cudaFree(cuda_ip_ApAp);
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
  std::fill(solution, solution + N, 0); // Choose x_0 = 0
 
  // initialize work vectors:
  double alpha, beta;
  double *cuda_solution, *cuda_p, *cuda_r, *cuda_Ap, *cuda_partial;
  cudaMalloc(&cuda_p, sizeof(double) * N);
  cudaMalloc(&cuda_r, sizeof(double) * N);
  cudaMalloc(&cuda_Ap, sizeof(double) * N);
  cudaMalloc(&cuda_solution, sizeof(double) * N);
  cudaMalloc(&cuda_partial, sizeof(double) * 256);
 
  cudaMemcpy(cuda_p, rhs, sizeof(double) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_r, rhs, sizeof(double) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_solution, solution, sizeof(double) * N, cudaMemcpyHostToDevice);
 
  cuda_dot_product<<<256, 256>>>(N, cuda_r, cuda_r, cuda_partial);
  cudaMemcpy(partial, cuda_partial, sizeof(double) * 256, cudaMemcpyDeviceToHost);
  double residual_norm_squared = 0;
  for (size_t i=0; i<256; ++i) residual_norm_squared += partial[i];
 
  double initial_residual_squared = residual_norm_squared;
 
  int iters = 0;
  cudaDeviceSynchronize();
  timer.reset();
  while (1) {
 
    // line 4: A*p:
    cuda_csr_matvec_product<<<512, 512>>>(N, csr_rowoffsets, csr_colindices, csr_values, cuda_p, cuda_Ap);
 
    // lines 5,6:
    cuda_dot_product<<<256, 256>>>(N, cuda_p, cuda_Ap, cuda_partial);
    cudaMemcpy(partial, cuda_partial, sizeof(double) * 256, cudaMemcpyDeviceToHost);
    alpha = 0;
    for (size_t i=0; i<256; ++i) alpha += partial[i];
    alpha = residual_norm_squared / alpha;
 
    // line 7:
    cuda_vecadd<<<512, 512>>>(N, cuda_solution, cuda_p, alpha);
 
    // line 8:
    cuda_vecadd<<<512, 512>>>(N, cuda_r, cuda_Ap, -alpha);
 
    // line 9:
    beta = residual_norm_squared;
    cuda_dot_product<<<256, 256>>>(N, cuda_r, cuda_r, cuda_partial);
    cudaMemcpy(partial, cuda_partial, sizeof(double) * 256, cudaMemcpyDeviceToHost);
    residual_norm_squared = 0;
    for (size_t i=0; i<256; ++i) residual_norm_squared += partial[i];
 
    // line 10:
    if (std::sqrt(residual_norm_squared / initial_residual_squared) < 1e-6) {
      break;
    }
 
    // line 11:
    beta = residual_norm_squared / beta;
 
    // line 12:
    cuda_vecadd2<<<512, 512>>>(N, cuda_p, cuda_r, beta);
 
    if (iters > 10000)
      break; // solver didn't converge
    ++iters;
  }
  cudaMemcpy(solution, cuda_solution, sizeof(double) * N, cudaMemcpyDeviceToHost);
 
  cudaDeviceSynchronize();
  std::cout << std::sqrt(N)<< ", " << timer.get() << ", " << iters << std::endl;
 
  if (iters > 10000)
    std::cout << "Conjugate Gradient did NOT converge within 10000 iterations"
              << std::endl;
  // else
  //   std::cout << "Conjugate Gradient converged in " << iters << " iterations."
  //             << std::endl;
 
  cudaFree(cuda_p);
  cudaFree(cuda_r);
  cudaFree(cuda_Ap);
  cudaFree(cuda_solution);
  cudaFree(cuda_partial);
}
 
/** Solve a system with `points_per_direction * points_per_direction` unknowns
 */
void solve_system(int points_per_direction) {
 
  int N = points_per_direction *
          points_per_direction; // number of unknows to solve for
 
  // std::cout << "Solving Ax=b with " << N << " unknowns." << std::endl;
 
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
  cudaMalloc(&cuda_csr_rowoffsets, sizeof(double) * (N + 1));
  cudaMalloc(&cuda_csr_colindices, sizeof(double) * 5 * N);
  cudaMalloc(&cuda_csr_values, sizeof(double) * 5 * N);
  cudaMemcpy(cuda_csr_rowoffsets, csr_rowoffsets, sizeof(double) * (N + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_csr_colindices, csr_colindices, sizeof(double) * 5 * N,   cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_csr_values,     csr_values,     sizeof(double) * 5 * N,   cudaMemcpyHostToDevice);
 
  //
  // Call Conjugate Gradient implementation with GPU arrays
  //
  conjugate_gradient(N, cuda_csr_rowoffsets, cuda_csr_colindices, cuda_csr_values, rhs, solution);
  //
  // Check for convergence:
  //
  double residual_norm = relative_residual(N, csr_rowoffsets, csr_colindices, csr_values, rhs, solution);

  // std::cout << "Relative residual norm: " << residual_norm
  //           << " (should be smaller than 1e-6)" << std::endl;
 
  cudaFree(cuda_csr_rowoffsets);
  cudaFree(cuda_csr_colindices);
  cudaFree(cuda_csr_values);
  free(solution);
  free(rhs);
  free(csr_rowoffsets);
  free(csr_colindices);
  free(csr_values);
}

void solve_system_pipelined(int points_per_direction) {

  int N = points_per_direction *
          points_per_direction; // number of unknows to solve for
 
  // std::cout << "Solving Ax=b with " << N << " unknowns." << std::endl;
 
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
  cudaMalloc(&cuda_csr_rowoffsets, sizeof(double) * (N + 1));
  cudaMalloc(&cuda_csr_colindices, sizeof(double) * 5 * N);
  cudaMalloc(&cuda_csr_values, sizeof(double) * 5 * N);
  cudaMemcpy(cuda_csr_rowoffsets, csr_rowoffsets, sizeof(double) * (N + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_csr_colindices, csr_colindices, sizeof(double) * 5 * N,   cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_csr_values,     csr_values,     sizeof(double) * 5 * N,   cudaMemcpyHostToDevice);
 
  //
  // Call Conjugate Gradient implementation with GPU arrays
  //
  //conjugate_gradient(N, cuda_csr_rowoffsets, cuda_csr_colindices, cuda_csr_values, rhs, solution);
  conjugate_gradient_pipelined(N, cuda_csr_rowoffsets, cuda_csr_colindices, cuda_csr_values, rhs, solution);
  //
  // Check for convergence:
  //
  double residual_norm = relative_residual(N, csr_rowoffsets, csr_colindices, csr_values, rhs, solution);

  // std::cout << "Relative residual norm: " << residual_norm
  //           << " (should be smaller than 1e-6)" << std::endl;
 
  cudaFree(cuda_csr_rowoffsets);
  cudaFree(cuda_csr_colindices);
  cudaFree(cuda_csr_values);
  free(solution);
  free(rhs);
  free(csr_rowoffsets);
  free(csr_colindices);
  free(csr_values);
}
 
int main() {

  // std::cout << "Classical Conjugate Gradient Algorithm"<< std::endl;
  // std::cout << "N,  "<< "ET,  " << "Iter " << std::endl;
  // solve_system(10);
  // solve_system(50);
  // solve_system(100);
  // solve_system(250);
  // solve_system(500);
  // solve_system(750);
  // solve_system(1000);
  // solve_system(1500);

  std::cout << "Pipelined Conjugate Gradient Algorithm"<< std::endl;
  std::cout << "N,  "<< "ET,  " << "Iter " << std::endl;
  solve_system_pipelined(10);
  solve_system_pipelined(50);
  solve_system_pipelined(100);
  solve_system_pipelined(250);
  solve_system_pipelined(500);
  solve_system_pipelined(750);
  solve_system_pipelined(1000);
  solve_system_pipelined(1500);

  return EXIT_SUCCESS;
}
 
