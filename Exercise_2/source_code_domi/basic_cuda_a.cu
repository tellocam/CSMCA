#include <stdio.h>
#include "timer.hpp"

int main(void)
{
	int k=0, i=8;
	int N_values[i] = { 1000000, 2000000, 4000000, 8000000, 16000000, 32000000, 64000000, 128000000};
	double *gpu_x;
	float t_malloc=0, t_free=0;
	Timer timer;
	cudaDeviceSynchronize();
	printf("\nsize, malloc, free\n");
	while(k < i) {
		int N = N_values[k];
		for (int n=0; n<5; n++) {
			timer.reset();
			cudaMalloc(&gpu_x, N*sizeof(double)); 
			cudaDeviceSynchronize();
			t_malloc += timer.get();
			timer.reset();
			cudaFree(gpu_x);
			cudaDeviceSynchronize();
			t_free += timer.get();
		} 
		printf("%d,%g,%g\n", N, 0.2*t_malloc, 0.2*t_free);
		k++;
	}
	return EXIT_SUCCESS;
}

