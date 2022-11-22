#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include "poisson2d.hpp"
#include "timer.hpp"

// Task 2a)
__global__ void CUDA_csr_matvec_product(size_t N, int *csr_rowoffsets, int *csr_colindices, double *csr_values, double *x, double *y)
{
  for (int row = blockDim.x * blockIdx.x + threadIdx.x; row < N; row += gridDim.x * blockDim.x)
  {
    double val = 0;
    for (int jj = csr_rowoffsets[row]; jj < csr_rowoffsets[row + 1]; ++jj)
    {
      val += csr_values[jj] * x[csr_colindices[jj]];
    }
    y[row] = val;
  }
}

// Copy Paste from Exercises!
__global__ void CUDA_dot_product(int N, double *x, double *y, double *result)
{
  __shared__ double shared_mem[1024];

  double dot = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
  {
    dot += x[i] * y[i];
  }

  shared_mem[threadIdx.x] = dot;
  for (int k = blockDim.x / 2; k > 0; k /= 2)
  {
    __syncthreads();
    if (threadIdx.x < k)
    {
      shared_mem[threadIdx.x] += shared_mem[threadIdx.x + k];
    }
  }

  if (threadIdx.x == 0)
  {
    atomicAdd(result, shared_mem[0]);
  }
}

__global__ void vectorAdd_Kernel_7(int N, double *x, double *y, double a)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
  {
    x[i] += a * y[i];
  }
}

__global__ void vectorAdd_Kernel_8(int N, double *x, double *y, double a)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
  {
    x[i] -= a * y[i];
  }
}

__global__ void vectorAdd_Kernel_12(int N, double *x, double *y, double a)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
  {
    x[i] = y[i] + a * x[i];
  }
}

double findMedian(std::vector<float> a)
{
  size_t n = a.size();
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

void conjugate_gradient(size_t N, int *csr_rowoffsets, int *cuda_csr_rowoffsets, int *csr_colindices, int *cuda_csr_colindices,
                        double *csr_values, double *cuda_csr_values, double *rhs, double *cuda_rhs,
                        double *solution, double *cuda_solution)
{
  Timer timer;
  std::vector<float> timings, timings_1, timings_2, timings_3, timings_4;

  const int nBlocks = 1024;
  const int nThreads = 1024;

  std::fill(solution, solution + N, 0);

  double *p = (double *)malloc(sizeof(double) * N);
  double *r = (double *)malloc(sizeof(double) * N);
  double *Ap = (double *)malloc(sizeof(double) * N);

  std::copy(rhs, rhs + N, p);
  std::copy(rhs, rhs + N, r);

  double alpha = 0; // <r, r> / <p, a*p>
  double beta = 0;  // <r, r> / <r_old, r_old>
  double pAp = 0;   // the dot product <p, A*p>
  double rr = 0;    // the dot product <r, r> (i.e. res_norm)
  double rr_old;

  // initialize work vectors on GPU:
  double *cuda_p, *cuda_r, *cuda_rr, *cuda_Ap, *cuda_pAp, *cuda_alpha;
  cudaMalloc(&cuda_p, sizeof(double) * N);
  cudaMalloc(&cuda_r, sizeof(double) * N);
  cudaMalloc(&cuda_Ap, sizeof(double) * N);
  cudaMalloc(&cuda_pAp, sizeof(double));
  cudaMalloc(&cuda_rr, sizeof(double));
  cudaMalloc(&cuda_alpha, sizeof(double));

  // Copy data to GPU:
  cudaMemcpy(cuda_solution, solution, sizeof(double) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_p, p, sizeof(double) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_r, r, sizeof(double) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_pAp, &pAp, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_rr, &rr, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_Ap, Ap, sizeof(double) * N, cudaMemcpyHostToDevice);

  int iterations = 0;
  while (1)
  {

    // line 4: A*p:
    timer.reset();
    CUDA_csr_matvec_product<<<nBlocks, nThreads>>>(N, cuda_csr_rowoffsets, cuda_csr_colindices,
                                                   cuda_csr_values, cuda_p, cuda_Ap);
    timings.push_back(timer.get());

    rr = 0;
    cudaMemcpy(cuda_rr, &rr, sizeof(double), cudaMemcpyHostToDevice);
    pAp = 0;
    cudaMemcpy(cuda_pAp, &pAp, sizeof(double), cudaMemcpyHostToDevice);

    // line 5, 6:
    timer.reset();
    CUDA_dot_product<<<nBlocks, nThreads>>>(N, cuda_p, cuda_Ap, cuda_pAp);
    cudaDeviceSynchronize();
    timings_1.push_back(timer.get());

    CUDA_dot_product<<<nBlocks, nThreads>>>(N, cuda_r, cuda_r, cuda_rr);
    cudaDeviceSynchronize();

    cudaMemcpy(&pAp, cuda_pAp, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&rr, cuda_rr, sizeof(double), cudaMemcpyDeviceToHost);

    // line 6:
    alpha = rr / pAp;

    // line 7, 8:
    timer.reset();
    vectorAdd_Kernel_7<<<nBlocks, nThreads>>>(N, cuda_solution, cuda_p, alpha);
    cudaDeviceSynchronize();
    timings_2.push_back(timer.get());
    timer.reset();
    vectorAdd_Kernel_8<<<nBlocks, nThreads>>>(N, cuda_r, cuda_Ap, alpha);
    cudaDeviceSynchronize();
    timings_3.push_back(timer.get());

    rr_old = rr;
    rr = 0;
    cudaMemcpy(cuda_rr, &rr, sizeof(double), cudaMemcpyHostToDevice);

    // line 9, 10:
    CUDA_dot_product<<<nBlocks, nThreads>>>(N, cuda_r, cuda_r, cuda_rr);
    cudaDeviceSynchronize();
    cudaMemcpy(&rr, cuda_rr, sizeof(double), cudaMemcpyDeviceToHost);

    if (rr < 1e-7)
    {
      cudaMemcpy(solution, cuda_solution, N * sizeof(double), cudaMemcpyDeviceToHost);
      break;
    }
    beta = rr / rr_old;
    timer.reset();
    vectorAdd_Kernel_12<<<nBlocks, nThreads>>>(N, cuda_p, cuda_r, beta);
    cudaDeviceSynchronize();
    timings_4.push_back(timer.get());
 
    if (iterations > 10000)
      break;
    ++iterations;
  }

  cudaMemcpy(p, cuda_p, N * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(r, cuda_r, sizeof(double) * N, cudaMemcpyDeviceToHost);

  if (iterations > 10000)
    std::cout << "CG not converged after 1e5 iterations" << std::endl;
  else
    std::cout << "CG converged after " << iterations << " iterations." << std::endl;

  printf("\nCUDA_csr_matvec_product: %f\n", findMedian(timings));
  printf("\nCUDA_dot_product: %f\n", findMedian(timings_1));
  printf("\nvectorAdd_Kernel_7: %f\n", findMedian(timings_2));
  printf("\nvectorAdd_Kernel_8: %f\n", findMedian(timings_3));
  printf("\nvectorAdd_Kernel_12: %f\n\n", findMedian(timings_4));

  free(p);
  free(r);
  free(Ap);
  cudaFree(cuda_p);
  cudaFree(cuda_r);
  cudaFree(cuda_Ap);
}


void solve_system(size_t CUDA_points_per_direction)
{

  size_t N = CUDA_points_per_direction * CUDA_points_per_direction;
  int *csr_rowoffsets = (int *)malloc(sizeof(double) * (N + 1));
  int *csr_colindices = (int *)malloc(sizeof(double) * 5 * N);
  double *csr_values = (double *)malloc(sizeof(double) * 5 * N);
  generate_fdm_laplace(CUDA_points_per_direction, csr_rowoffsets, csr_colindices, csr_values);
  double *solution = (double *)malloc(sizeof(double) * N);
  double *rhs = (double *)malloc(sizeof(double) * N);

  std::fill(rhs, rhs + N, 1);
  int *cuda_csr_rowoffsets, *cuda_csr_colindices;
  double *cuda_csr_values, *cuda_solution, *cuda_rhs;

  cudaMalloc(&cuda_csr_rowoffsets, sizeof(double) * (N + 1));
  cudaMalloc(&cuda_csr_colindices, sizeof(double) * 5 * N);
  cudaMalloc(&cuda_csr_values, sizeof(double) * 5 * N);
  cudaMalloc(&cuda_solution, sizeof(double) * N);
  cudaMalloc(&cuda_rhs, sizeof(double) * N);
  cudaMemcpy(cuda_csr_rowoffsets, csr_rowoffsets, sizeof(double) * (N + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_csr_colindices, csr_colindices, sizeof(double) * 5 * N, cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_csr_values, csr_values, sizeof(double) * 5 * N, cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_solution, solution, sizeof(double) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_rhs, solution, sizeof(double) * N, cudaMemcpyHostToDevice);

  conjugate_gradient(N, csr_rowoffsets, cuda_csr_rowoffsets, csr_colindices, cuda_csr_colindices,
                     csr_values, cuda_csr_values, rhs, cuda_rhs, solution, cuda_solution);


  double residual_norm = relative_residual(N, csr_rowoffsets, csr_colindices, csr_values, rhs, solution);
  std::cout << "Relative residual norm: " << residual_norm << " (should be smaller than 1e-6)" << std::endl;

  free(solution);
  free(rhs);
  free(csr_rowoffsets);
  free(csr_colindices);
  free(csr_values);
  cudaFree(cuda_solution);
  cudaFree(cuda_rhs);
  cudaFree(cuda_csr_rowoffsets);
  cudaFree(cuda_csr_colindices);
  cudaFree(cuda_csr_values);
}

int main()
{

  Timer timer;
  std::vector<float> timings;
  timer.reset();
  solve_system(1500);
  float t = timer.get();
  printf("\nt = %f\n", t);

  return EXIT_SUCCESS;
}