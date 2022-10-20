#include <stdio.h>
#include "timer.hpp"

__global__ void initVec(double *vec1, double *vec2, int N) {
	unsigned int total_threads = blockDim.x * gridDim.x;
	unsigned int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	for (unsigned int i = global_tid; i<N; i += total_threads) {
		vec1[i] = (double)(i);
		vec2[i] = (double)(N-i-1);
	}
}

int main(void)
{
	int k=0, i=6;
	int N_values[i] = { 1000000, 2000000, 4000000, 8000000, 16000000, 32000000 };
	double *x, *y, *gpu_x, *gpu_y;
	Timer timer;	
	float option_1=0;
	k = 0;
	printf("\nsize,option1\n");
	while(k < i) {
		int N = N_values[k];
		for (int n=0; n<5; n++) {
			timer.reset();
			x = (double*)malloc(N*sizeof(double));
			y = (double*)malloc(N*sizeof(double));
			cudaMalloc(&gpu_x, N*sizeof(double)); 
			cudaMalloc(&gpu_y, N*sizeof(double));
			initVec<<<256, 256>>>(gpu_x, gpu_y, N);
			cudaDeviceSynchronize();
			option_1 += timer.get();
		}
		printf("%d,%g\n", N, 0.2*option_1);
		cudaFree(gpu_x);
		cudaFree(gpu_y);
		free(x);
		free(y);
		k++;		
	}
	float option_2=0;
	k = 0;
	printf("\nsize,option2\n");
	while(k < i) {
		int N = N_values[k];
		for (int n=0; n<5; n++) {
			timer.reset();
			cudaDeviceSynchronize();
			x = (double*)malloc(N*sizeof(double));
			y = (double*)malloc(N*sizeof(double));
			for (int i = 0; i < N; i++) {
				x[i] = (double)(i);
				y[i] = (double)(N-i-1);
			}
			cudaMalloc(&gpu_x, N*sizeof(double)); 
			cudaMalloc(&gpu_y, N*sizeof(double));
			cudaMemcpy(gpu_x, x, N*sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(gpu_y, y, N*sizeof(double), cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();
			option_2 += timer.get();
		}
		printf("%d,%g\n", N, 0.2*option_2);
		cudaMemcpy(x, gpu_x, N*sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(y, gpu_y, N*sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(gpu_x);
		cudaFree(gpu_y);
		free(x);
		free(y);
		k++;
	}
	return EXIT_SUCCESS;
}

