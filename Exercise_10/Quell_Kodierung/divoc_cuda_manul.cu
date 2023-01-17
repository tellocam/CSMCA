/**
 * 360.252 - Computational Science on Many-Core Architectures
 * WS 2022/23, TU Wien
 *
 * Simplistic simulator for a disease of very immediate concern (DIVOC). Inspired by COVID-19 simulations.
 *
 * DISCLAIMER: This simulator is for educational purposes only.
 * It may be arbitrarily inaccurate and should not be used for drawing any conclusions about any actual virus.
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "timer.hpp"


#include <cuda_runtime.h>
#include <cublas_v2.h>
// #include <stdio.h>
#include <cmath>
#include <iostream>
// #include "timer.hpp"
#include "cuda_errchk.hpp"
#include <algorithm>
#include <vector>

#ifndef M_PI
    #define M_PI  3.14159265358979323846
#endif

//
// Data container for simulation input
//
typedef struct {
    
    size_t population_size;    // Number of people to simulate
    
    //// Configuration
    int mask_threshold;        // Number of cases required for masks
    int lockdown_threshold;    // Number of cases required for lockdown
    int infection_delay;       // Number of days before an infected person can pass on the disease
    int infection_days;        // Number of days an infected person can pass on the disease
    int starting_infections;   // Number of infected people at the start of the year
    int immunity_duration;     // Number of days a recovered person is immune
    
    // for each day:
    int    *contacts_per_day;           // number of other persons met each day to whom the disease may be passed on
    double *transmission_probability;   // how likely it is to pass on the infection to another person
 
} SimInput_t;
 
void init_input(SimInput_t *input) {
    input->population_size = 8916845;  // Austria's population in 2020 according to Statistik Austria
    
    input->mask_threshold      = 5000;
    input->lockdown_threshold  = 50000;
    input->infection_delay     = 5;     // 5 to 6 days incubation period (average) according to WHO
    input->infection_days      = 3;     // assume three days of passing on the disease
    input->starting_infections = 10;
    input->immunity_duration   = 180;   // half a year of immunity
    
    input->contacts_per_day = (int*)malloc(sizeof(int) * 365);
    input->transmission_probability = (double*)malloc(sizeof(double) * 365);
    for (int day = 0; day < 365; ++day) {
        input->contacts_per_day[day] = 6;             // arbitrary assumption of six possible transmission contacts per person per day, all year
        input->transmission_probability[day] = 0.2
                                            + 0.1 * cos((day / 365.0) * 2 * M_PI);   // higher transmission in winter, lower transmission during summer
    }
}

__global__ void cuda_init_input(SimInput_t *input, int *cpd, double *tp) {
    input->population_size = 8916845;  // Austria's population in 2020 according to Statistik Austria
    
    input->mask_threshold      = 5000;
    input->lockdown_threshold  = 50000;
    input->infection_delay     = 5;     // 5 to 6 days incubation period (average) according to WHO
    input->infection_days      = 3;     // assume three days of passing on the disease
    input->starting_infections = 10;
    input->immunity_duration   = 180;   // half a year of immunity

    input->contacts_per_day = cpd;
    input->transmission_probability = tp;

    for (int day = blockIdx.x*blockDim.x + threadIdx.x; day < 365; day += gridDim.x*blockDim.x) {
        input->contacts_per_day[day] = 6;             // arbitrary assumption of six possible transmission contacts per person per day, all year
        input->transmission_probability[day] = 0.2
                                            + 0.1 * cos((day / 365.0) * 2 * M_PI);   // higher transmission in winter, lower transmission during summer
    }
}

__global__ void cuda_print(SimInput_t *input) {
    printf("Threshold on GPU: %d \n", input->mask_threshold);
}
 
typedef struct {
    // for each day:
    int *active_infections;     // number of active infected on that day (including incubation period)
    int *lockdown;              // 0 if no lockdown on that day, 1 if lockdown
    
    // for each person:
    int *is_infected;      // 0 if healty, 1 if currently infected
    int *infected_on;      // day of infection. negative if not yet infected. January 1 is Day 0.
    
} SimOutput_t;
 
//
// Initializes the output data structure (values to zero, allocate arrays)
//

void init_output(SimOutput_t *output, int population_size) {
    output->active_infections = (int*)malloc(sizeof(int) * 365);
    output->lockdown          = (int*)malloc(sizeof(int) * 365);
    for (int day = 0; day < 365; ++day) {
        output->active_infections[day] = 0;
        output->lockdown[day] = 0;
    }
    
    output->is_infected       = (int*)malloc(sizeof(int) * population_size);
    output->infected_on       = (int*)malloc(sizeof(int) * population_size);
    // printf("init_output()\n");
    for (int i=0; i<population_size; ++i) {
        output->is_infected[i] = 0;
        output->infected_on[i] = 0;
    }
}

__global__ void cuda_init_output(SimOutput_t *output, SimInput_t *input, int *ai, int *ld, int *ii, int *io) {
    output->active_infections = ai;
    output->lockdown          = ld;
     for (int day = blockIdx.x*blockDim.x + threadIdx.x; day < 365; day += gridDim.x*blockDim.x) {
        output->active_infections[day] = 0;
        output->lockdown[day] = 0;
    }
    output->is_infected       = ii;
    output->infected_on       = io;
    
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < input->population_size; i += gridDim.x*blockDim.x) {
        output->is_infected[i] = (i < input->starting_infections) ? 1 : 0;
        output->infected_on[i] = (i < input->starting_infections) ? 0 : -1;
    }
}

__global__ void cuda_print_out(SimOutput_t *output) {
    printf("Infected on GPU: %d \n", output->is_infected[0]);
}

__global__ void cuda_determine_infections(const SimInput_t *input, SimOutput_t *output, int *numInfected, int *numRecovered, int day) {
    // Step 1: determine number of infections and recoveries
    int num_infected_current = 0;
    int num_recovered_current = 0;
    __shared__ int shared_inf[256];
    __shared__ int shared_rec[256];

    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < input->population_size; i += gridDim.x*blockDim.x) {
        if (output->is_infected[i] > 0) {
            if (output->infected_on[i] > day - input->infection_delay - input->infection_days
            && output->infected_on[i] <= day - input->infection_delay)   // currently infected and incubation period over
            {
                num_infected_current += 1;
                // printf("person %d is infected on day %d (info in thread %d in block %d)\n", i, day, threadIdx.x, blockIdx.x);
            }
            else if (output->infected_on[i] < day - input->infection_delay - input->infection_days)
            {
                num_recovered_current += 1;
                // printf("person %d is recovered on day %d (info in thread %d in block %d)\n", i, day);
            }
        }
    }
 
    // write to shared mem
    shared_inf[threadIdx.x] = num_infected_current;
    shared_rec[threadIdx.x] = num_recovered_current;
    // blockwise reduction
    for(unsigned int stride = blockDim.x/2; stride > 0 ; stride/=2){
        __syncthreads(); // synchronize threads within thread block
        if(threadIdx.x < stride){
            shared_inf[threadIdx.x] += shared_inf[threadIdx.x + stride];
            shared_rec[threadIdx.x] += shared_rec[threadIdx.x + stride];
        }
    }
    if (threadIdx.x == 0) {
        numInfected[blockIdx.x] = shared_inf[0];
        numRecovered[blockIdx.x] = shared_rec[0];
    }
}

// reduction between blocks
__global__ void cuda_reduction(int *input) { 
    __shared__ int shared_mem[256];
    shared_mem[threadIdx.x] = input[threadIdx.x];
    for(unsigned int stride = blockDim.x/2; stride > 0; stride/=2){
        __syncthreads(); // synchronize threads within thread block
        if(threadIdx.x < stride){
            shared_mem[threadIdx.x] += shared_mem[threadIdx.x + stride];
        }
    }
    if(threadIdx.x == 0) input[0] = shared_mem[0];
}

__global__ void cuda_lockdown(const SimInput_t *input, SimOutput_t *output, int *numInfected, int day) {

        if (numInfected[0] > input->lockdown_threshold) {
            output->lockdown[day] = 1;
        }
        if (day > 0 && output->lockdown[day-1] == 1) { // end lockdown if number of infections has reduced significantly
            output->lockdown[day] = (numInfected[0] < input->lockdown_threshold / 3) ? 0 : 1;
        }
}

__global__ void cuda_adjust_params(const SimInput_t *input, SimOutput_t *output, int *numInfected, int day) {
    // double contacts_today = input->contacts_per_day[day];
    // double transmission_probability_today = input->transmission_probability[day];
    if (numInfected[0] > input->mask_threshold) { // transmission is reduced with masks. Arbitrary factor: 2
        input->transmission_probability[day] /= 2.0;
    }
    if (output->lockdown[day]) { // contacts are significantly reduced in lockdown. Arbitrary factor: 4
        input->contacts_per_day[day] /= 4;
    }
}

__global__ void cuda_pass_infection_a(const SimInput_t *input, SimOutput_t *output, double *cuda_rand, int day){
    // if(threadIdx.x < 10 && blockIdx.x == 0) printf("check if infectious: person = %d, yes/no = %d\n", threadIdx.x, output->is_infected[threadIdx.x]);
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < input->population_size; i += gridDim.x*blockDim.x)  // loop over population
    {
        if (   output->is_infected[i] > 0
            && output->infected_on[i] >  day - input->infection_delay - input->infection_days  // currently infected
            && output->infected_on[i] <= day - input->infection_delay)                         // already infectious
        {
            // printf("person %d is infectious on day %d (info in thread %d in block %d)\n", i, day, threadIdx.x, blockIdx.x);
            // pass on infection to other persons with transmission probability
            for (int j=0; j<input->contacts_per_day[day]; ++j) 
            {
                double r = cuda_rand[((i*day*6+j)*2) % (input->population_size * 6 * 2)];  // random number between 0 and 1 imported from cpu
                // double r = 0.0;
                if (r < input->transmission_probability[day]) {

                    r = cuda_rand[(i*6+j)*2+1];       // new random number to determine a random other person to transmit the virus to
                    // r = 0.5;
                    int other_person = r * input->population_size;
                    if (output->is_infected[other_person] == 0     // other person is not infected
                    || output->infected_on[other_person] < day - input->immunity_duration) { // other person has no more immunity
                        output->is_infected[other_person] = 1;
                        output->infected_on[other_person] = day;
                    }
                }
            } // for contacts_per_day
        } // if currently infected
    } // for i
}

__global__ void cuda_pass_infection_b(const SimInput_t *input, SimOutput_t *output, double *cuda_rand, int day){
    int A = 1103515245;
    int C = 12345;
    int M = std::pow(2, 31);
    double r = cuda_rand[blockIdx.x*blockDim.x + threadIdx.x];
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < input->population_size; i += gridDim.x*blockDim.x)  // loop over population
    {
        if (   output->is_infected[i] > 0
            && output->infected_on[i] >  day - input->infection_delay - input->infection_days  // currently infected
            && output->infected_on[i] <= day - input->infection_delay)                         // already infectious
        {
            // pass on infection to other persons with transmission probability
            for (int j=0; j<input->contacts_per_day[day]; ++j) 
            {
                r = (((long)(A * r + C)) % M);  // random number between 0 and M imported 
                // if(day == 125) printf("r/M = %f", r/M);
                if ((r/M) < input->transmission_probability[day]) {
                    r = (((long)(A * r + C)) % M);       // new random number to determine a random other person to transmit the virus to
                    int other_person = (r / M) * input->population_size;
                    if (output->is_infected[other_person] == 0     // other person is not infected
                    || output->infected_on[other_person] < day - input->immunity_duration) { // other person has no more immunity
                        output->is_infected[other_person] = 1;
                        output->infected_on[other_person] = day;
                    }
                }
            } // for contacts_per_day
        } // if currently infected
    } // for i
}

__global__ void cuda_get_rand(double *cuda_rand, int day){
    int A = 1103515245;
    int C = 12345;
    int M = std::pow(2, 31); 
    // x_(n+1) = (A * x_n + C) mod M
    cuda_rand[0] = day;
    for (int i = 1; i < 256*256; i++)
    {
        cuda_rand[i] = ((long)(A * cuda_rand[i-1] + C) % M);
    }
}
 
void run_simulation(const SimInput_t *input, SimOutput_t *output) {
    
    //
    // Run simulation
    //

    // Step 1
    //
    int *cuda_numInfected, *cuda_numRecovered;
    CUDA_ERRCHK(cudaMalloc(&cuda_numInfected, sizeof(int)*256));
    CUDA_ERRCHK(cudaMalloc(&cuda_numRecovered, sizeof(int)*256));
    int *numRecovered = (int*)malloc(sizeof(int) * 256);
    int *numInfected = (int*)malloc(sizeof(int) * 256);

    // double *cuda_ctd, *cuda_tpd;
    // CUDA_ERRCHK(cudaMalloc(&cuda_ctd, sizeof(double)));
    // CUDA_ERRCHK(cudaMalloc(&cuda_tpd, sizeof(double)));

    int population_size;//, contacts_per_day;
    cudaMemcpy(&population_size, &input->population_size, sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(&contacts_per_day, &input->contacts_per_day, sizeof(int), cudaMemcpyDeviceToHost);

    // sähe den Samen für (a)
    srand(0);

    //create some randomnes for (a)
    double *cpu_rand = (double *)malloc(sizeof(double) * population_size * 6 * 2);
    for(int i = 0; i < population_size * 6 * 2; i++)
    {
        cpu_rand[i] = ((double)rand())/(double)RAND_MAX;
    }
    double *cuda_rand_a;
    cudaMalloc(&cuda_rand_a, sizeof(double) * population_size * 6 * 2);
    cudaMemcpy(cuda_rand_a, cpu_rand, sizeof(double) * population_size * 6 * 2, cudaMemcpyHostToDevice);

    // create some randomnes for (b)
    double *cuda_rand_b;
    cudaMalloc(&cuda_rand_b, sizeof(double) * 256 * 256);

    for (int day=0; day<365; ++day)  // loop over all days of the year
    {
        // get todays infected
        cudaDeviceSynchronize();
        cuda_determine_infections<<<256,256>>>(input, output, cuda_numInfected, cuda_numRecovered, day);
        cudaDeviceSynchronize();
        cuda_reduction<<<1,256>>>(cuda_numInfected);
        cudaDeviceSynchronize();
        cuda_reduction<<<1,256>>>(cuda_numRecovered);
        cudaDeviceSynchronize();
        int infected;
        cudaMemcpy(numInfected, cuda_numInfected, sizeof(int),cudaMemcpyDeviceToHost);
        infected = numInfected[0];
        printf("Infected on day %d: %d\n", day, numInfected[0]);

        // check for lockdown
        cuda_lockdown<<<1,1>>>(input, output, cuda_numInfected, day);
        cudaDeviceSynchronize();

        //
        // Step 2: determine today's transmission probability and contacts based on pandemic situation
        //

        cuda_adjust_params<<<1,1>>>(input, output, cuda_numInfected, day);
        cudaDeviceSynchronize();
        //
        // Step 3: pass on infections within population
        //
        
        // for (a)
        cudaDeviceSynchronize();
        cuda_pass_infection_a<<<256,256>>>(input, output, cuda_rand_a, day);
        cudaDeviceSynchronize();

        // // for (b)
        // cudaDeviceSynchronize();
        // cuda_get_rand<<<1,1>>>(cuda_rand_b, day);
        // cudaDeviceSynchronize();
        // cuda_pass_infection_b<<<265,265>>>(input, output, cuda_rand_b, day);
        // cudaDeviceSynchronize();

    } // for day
}

 
int main(int argc, char **argv) {
 
    SimInput_t input;
    SimOutput_t output;

    init_input(&input); // works :D 
    init_output(&output, input.population_size);

    // GPU INIT

    SimInput_t *cuda_input; 
    CUDA_ERRCHK(cudaMalloc(&cuda_input, sizeof(SimInput_t))); // double pointer !!!
    int *cuda_cpd;
    CUDA_ERRCHK(cudaMalloc(&cuda_cpd, sizeof(int)*365));
    double *cuda_tp;
    CUDA_ERRCHK(cudaMalloc(&cuda_tp, sizeof(double)*365));
    cuda_init_input<<<1,1>>>(cuda_input, cuda_cpd, cuda_tp);
    cudaDeviceSynchronize();
    // cuda_print<<<2,2>>> (cuda_input);

    // get pop size for cudamalloc to init output
    int population_size;
    cudaMemcpy(&population_size, &cuda_input->population_size, 
                                sizeof(int), cudaMemcpyDeviceToHost);

    SimOutput_t *cuda_output; 
    CUDA_ERRCHK(cudaMalloc(&cuda_output, sizeof(SimOutput_t))); // double pointer !!!
    int *cuda_ai, *cuda_ld, *cuda_ii, *cuda_io;
    CUDA_ERRCHK(cudaMalloc(&cuda_ai, sizeof(int)*365));
    CUDA_ERRCHK(cudaMalloc(&cuda_ld, sizeof(int)*365));
    CUDA_ERRCHK(cudaMalloc(&cuda_ii, sizeof(int)*population_size)); // popsize = 8916845
    CUDA_ERRCHK(cudaMalloc(&cuda_io, sizeof(int)*population_size)); // popsize = 8916845
    cuda_init_output<<<1,1>>>(cuda_output, cuda_input, cuda_ai, cuda_ld, cuda_ii, cuda_io);
    cudaDeviceSynchronize();
    // cuda_print_out<<<2,2>>> (cuda_output);

    




    
    
    Timer timer;
    srand(0); // initialize random seed for deterministic output
    timer.reset();
    // run_simulation(&input, &output);
    run_simulation(cuda_input, cuda_output);

    printf("Simulation time: %d, %g,\n", input.population_size, timer.get());
    
    return EXIT_SUCCESS;
}