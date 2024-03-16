
#include "common.h"

#include "timer.h"

__global__ void vecMax_kernel(double* a, double* b, double* c, unsigned int M) {

    // TODO
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i<M){
        if(a[i]>b[i])
            c[i]=a[i];
        else 
            c[i]=b[i];
    }

}

void vecMax_gpu(double* a, double* b, double* c, unsigned int M) {

    Timer timer;

    // Allocate GPU memory
    
    startTime(&timer);
    double *a_d,*b_d,*c_d;
    cudaMalloc((void**)&a_d,sizeof(double)*M);
    cudaMalloc((void**)&b_d,sizeof(double)*M);
    cudaMalloc((void**)&c_d,sizeof(double)*M);
    // TODO





    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    // Copy data to GPU
    startTime(&timer);
    cudaMemcpy(a_d,a,sizeof(double)*M,cudaMemcpyHostToDevice);
    cudaMemcpy(b_d,b,sizeof(double)*M,cudaMemcpyHostToDevice);
    
    // TODO



    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");

    // Call kernel
    startTime(&timer);
    const unsigned int nb_threads_per_block=512;
    const unsigned int nb_blocks=(M+nb_threads_per_block-1)/nb_threads_per_block;
    vecMax_kernel<<<nb_blocks,nb_threads_per_block>>>(a_d,b_d,c_d,M);

    // TODO




    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time", GREEN);

    // Copy data from GPU
    startTime(&timer);
    cudaMemcpy(c,c_d,sizeof(double)*M,cudaMemcpyDeviceToHost);
    // TODO


    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

    // Free GPU memory
    startTime(&timer);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    // TODO



    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");

}

