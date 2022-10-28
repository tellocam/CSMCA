#include <stdio.h>
#include "timer.hpp"
__global__ void addVec_kth(double *x, double *y, double *z, int N, int k) {
	// something wong with threads and thread id, adapt it till it wokrs! //
	unsigned int total_threads = blockDim.x * gridDim.x;
	unsigned int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (unsigned int i = global_tid; i<N/k; i += total_threads) {
		z[i*k] = x[i*k] + y[i*k];
	}

}
int main(void)
{
	// Task a //
	double *x, *y, *z, *gpu_x, *gpu_y, *gpu_z;
	float t_ex1a = 0;
	Timer timer;
	int N = pow(10.0, 8.0);
	int k_values[10] = {0, 1, 3, 7, 11, 15, 23, 31, 48, 63}; 
	x = (double*)malloc(N*sizeof(double));
	y = (double*)malloc(N*sizeof(double));
	z = (double*)malloc(N*sizeof(double));
	for (int i = 0; i < N; i++) {
		x[i] = (double)(i);
		y[i] = (double)(N-i-1);
	}
	cudaMalloc(&gpu_x, N*sizeof(double)); 
	cudaMalloc(&gpu_y, N*sizeof(double));
	cudaMalloc(&gpu_z, N*sizeof(double));
	cudaMemcpy(gpu_x, x, N*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_y, y, N*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(z, gpu_z, N*sizeof(double), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 10; i++) {
		for (int m = 0; m < 10; m++) {
			timer.reset();
			addVec_kth<<<256, 256>>>(gpu_x, gpu_y, gpu_z, N, k_values[i]);
			t_ex1a = timer.get();
		}
		// Here still average calculated -> implement Median //
	printf("%d,%g\n", k_values[i], 0.1*t_ex1a);
	}

	cudaFree(gpu_x);
	cudaFree(gpu_y);
	cudaFree(gpu_z);
	free(x);
	free(y);
	free(z);

	// // Task d //
	// int k = 0;
	// int N_values[10] = { 100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000, 3000000 };
	// printf("\nsize,time\n");
	// while(k < 10) {
	// 	float t_kernel=0;
	// 	int N = N_values[k];
	// 	x = (double*)malloc(N*sizeof(double));
	// 	y = (double*)malloc(N*sizeof(double));
	// 	z = (double*)malloc(N*sizeof(double));
	// 	for (int i = 0; i < N; i++) {
	// 		x[i] = (double)(i);
	// 		y[i] = (double)(N-i-1);
	// 	}
	// 	cudaMalloc(&gpu_x, N*sizeof(double)); 
	// 	cudaMalloc(&gpu_y, N*sizeof(double));
	// 	cudaMalloc(&gpu_z, N*sizeof(double));
	// 	cudaMemcpy(gpu_x, x, N*sizeof(double), cudaMemcpyHostToDevice);
	// 	cudaMemcpy(gpu_y, y, N*sizeof(double), cudaMemcpyHostToDevice);
	// 	timer.reset();
	// 	for (int n=0; n<5; n++) {
	// 		addVec<<<256, 256>>>(gpu_x, gpu_y, gpu_z, N);
	// 		cudaDeviceSynchronize();
	// 	}
	// 	t_kernel += timer.get();
	// 	printf("%d,%g\n", N, 0.2*t_kernel);
	// 	cudaMemcpy(z, gpu_z, N*sizeof(double), cudaMemcpyDeviceToHost);
	// 	cudaFree(gpu_x);
	// 	cudaFree(gpu_y);
	// 	cudaFree(gpu_z);
	// 	free(x);
	// 	free(y);
	// 	free(z);
	// 	k++;
	// }
	// // Task e //
	// N = 10000000;
	// k = 0;
	// int params[7] = { 16, 32, 64, 128, 256, 512, 1024};
	// printf("\nsqrt(threads),time\n");
	// while(k < 7) {
	// 	float t_kernel=0;
	// 	int param = params[k];
	// 	x = (double*)malloc(N*sizeof(double));
	// 	y = (double*)malloc(N*sizeof(double));
	// 	z = (double*)malloc(N*sizeof(double));
	// 	for (int i = 0; i < N; i++) {
	// 		x[i] = (double)(i);
	// 		y[i] = (double)(N-i-1);
	// 	}
	// 	cudaMalloc(&gpu_x, N*sizeof(double)); 
	// 	cudaMalloc(&gpu_y, N*sizeof(double));
	// 	cudaMalloc(&gpu_z, N*sizeof(double));
	// 	cudaMemcpy(gpu_x, x, N*sizeof(double), cudaMemcpyHostToDevice);
	// 	cudaMemcpy(gpu_y, y, N*sizeof(double), cudaMemcpyHostToDevice);
	// 	timer.reset();
	// 	for (int n=0; n<5; n++) {
	// 		addVec<<<param, param>>>(gpu_x, gpu_y, gpu_z, N);
	// 		cudaDeviceSynchronize();
	// 	}
	// 	t_kernel += timer.get();
	// 	printf("%d,%g\n", param, 0.2*t_kernel);
	// 	cudaMemcpy(z, gpu_z, N*sizeof(double), cudaMemcpyDeviceToHost);
	// 	cudaFree(gpu_x);
	// 	cudaFree(gpu_y);
	// 	cudaFree(gpu_z);
	// 	free(x);
	// 	free(y);
	// 	free(z);
	// 	k++;
	//}
	return EXIT_SUCCESS;
}

