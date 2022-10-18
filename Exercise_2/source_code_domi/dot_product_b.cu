#include <stdio.h>
#include "timer.hpp"

const int threads_per_block = 256;

double dotVec_cpu(double *a, double *b, int N) {
   double product = 0;
   for (int i = 0; i < N; i++)
   product = product + a[i] * b[i];
   return product;
}

__global__ void dotVec_gpu(double *x, double *y, double *partial_z, int N) {
	// 'temp_arr' can be accessed by all threads within a block
	// each thread in a block has a slot in 'temp_arr'
	__shared__ double temp_arr[threads_per_block];
	// local variable for each thread
	double thread_product = 0;
	// global thread ID (within grid)
	unsigned int global_tid = threadIdx.x + blockIdx.x * blockDim.x;
	// local thread ID (within block)
	unsigned int local_tid  = threadIdx.x;
	// total number of threads in grid
	unsigned int total_threads = blockDim.x * gridDim.x;
	
	// compute local thread products and account for case of
	// vector size exceeding total number of threads
	for (unsigned int i=global_tid; i<N; i+=total_threads) {
		thread_product += x[i] * y[i];
	}
	
	// write local thread product to shared memory
	temp_arr[local_tid] = thread_product;
	
	// reduction within each block in shared memory
	for (unsigned int stride = blockDim.x/2; stride>0; stride/=2) {
		__syncthreads();
		if (threadIdx.x < stride) {
			temp_arr[threadIdx.x] += temp_arr[threadIdx.x + stride];
		}
	}
	
	// only first thread of block writes result
	if (threadIdx.x == 0) {
		partial_z[blockIdx.x] = temp_arr[0];
	}
}



int main(void)
{
	/////////////////////////
	// Basic CUDA Task (b) //
	/////////////////////////
	
	double *x, *y, z, *partial_z;
	double *d_x, *d_y, *d_partial_z;
	Timer timer;
	
	int k = 0;
	int N_vals_d[10] = { 100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000, 3000000 };

	printf("\nsize,time\n");
	while(k < 10) {
		int N = N_vals_d[k];

		// Allocate host memory and initialize
		x = (double*)malloc(N*sizeof(double));
		y = (double*)malloc(N*sizeof(double));
		partial_z = (double*)malloc(256*sizeof(double));

		for (int i = 0; i < N; i++) {
			x[i] = 1.0;
			y[i] = 1.0;
		}

		// Allocate device memory and copy host data over
		cudaMalloc(&d_x, N*sizeof(double)); 
		cudaMalloc(&d_y, N*sizeof(double));
		cudaMalloc(&d_partial_z, 256*sizeof(double));

		cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_y, y, N*sizeof(double), cudaMemcpyHostToDevice);

		timer.reset();
		for (int n=0; n<10; n++) {
		
			dotVec_gpu<<<256, 256>>>(d_x, d_y, d_partial_z, N);

			// copy data back (implicit synchronization point)
			cudaMemcpy(partial_z, d_partial_z, 256*sizeof(double), cudaMemcpyDeviceToHost);
		  
			// finish on cpu
			z = 0;
			for(int i=0; i<256; i++) {
				z += partial_z[i];
			}
		}
		
		printf("%d,%g\n", N, 0.1*timer.get());

		// tidy up host and device memory
		cudaFree(d_x);
		cudaFree(d_y);
		cudaFree(d_partial_z);
		free(x);
		free(y);
		free(partial_z);
		
		k++;
	}

	return EXIT_SUCCESS;
}
