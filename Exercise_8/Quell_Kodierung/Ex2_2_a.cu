#include <stdio.h>
#include "timer.hpp"

const int threads_per_block = 256;
double dot_cpu(double *a, double *b, int N) {
   double product = 0;
   for (int i = 0; i < N; i++)
   product = product + a[i] * b[i];
   return product;
}
__global__ void dotVec_one(double *x, double *y, double *partial_z, int N) {
	__shared__ double temp_arr[threads_per_block];
	double thread_product = 0;
	unsigned int global_tid = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int local_tid  = threadIdx.x;
	unsigned int total_threads = blockDim.x * gridDim.x;
	for (unsigned int i=global_tid; i<N; i+=total_threads) {
		thread_product += x[i] * y[i];
	}
	temp_arr[local_tid] = thread_product;
	for (unsigned int stride = blockDim.x/2; stride>0; stride/=2) {
		__syncthreads();
		if (threadIdx.x < stride) {
			temp_arr[threadIdx.x] += temp_arr[threadIdx.x + stride];
		}
	}
	if (threadIdx.x == 0) {
		partial_z[blockIdx.x] = temp_arr[0];
	}
}
__global__ void dotVec_two(double *partial_z) {
	for (int stride = blockDim.x/2; stride>0; stride/=2) {
		__syncthreads();
		if (threadIdx.x < stride)
			partial_z[threadIdx.x] += partial_z[threadIdx.x+stride];
	}
}
int main(void)
{
	// Task a //
	double *x, *y, *z;
	double *gpu_x, *gpu_y, *gpu_partial_z;
	Timer timer;
	int k = 0;
	int N_values_d[19] = { 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728};
	printf("\nsize,time\n");
	while(k <= 19) {
		int N = N_values_d[k];
		x = (double*)malloc(N*sizeof(double));
		y = (double*)malloc(N*sizeof(double));
		z = (double*)malloc(threads_per_block*sizeof(double));
		for (int i = 0; i < N; i++) {
			x[i] = 1.0;
			y[i] = 1.0;
		}
		cudaMalloc(&gpu_x, N*sizeof(double)); 
		cudaMalloc(&gpu_y, N*sizeof(double));
		cudaMalloc(&gpu_partial_z, threads_per_block*sizeof(double));
		cudaMemcpy(gpu_x, x, N*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_y, y, N*sizeof(double), cudaMemcpyHostToDevice);
		timer.reset();
		for (int n=0; n<5; n++) { 
			dotVec_one<<<256, threads_per_block>>>(gpu_x, gpu_y, gpu_partial_z, N);	
			cudaDeviceSynchronize();
			dotVec_two<<<1, threads_per_block>>>(gpu_partial_z);
			cudaMemcpy(z, gpu_partial_z, threads_per_block*sizeof(double), cudaMemcpyDeviceToHost);
		}
		printf("%g,%g\n", z[0], 0.2*timer.get());
		cudaFree(gpu_x);
		cudaFree(gpu_y);
		cudaFree(gpu_partial_z);
		free(x);
		free(y);
		free(z);
		k++;	
	}
	return EXIT_SUCCESS;
}
