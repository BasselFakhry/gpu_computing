
#include "common.h"

#include "timer.h"

__global__ void mm_kernel(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {

    // TODO
    unsigned int row=blockIdx.y*blockDim.y+threadIdx.y;
    unsigned int col=blockIdx.x*blockDim.x+threadIdx.x;
    float sum=0.0f;
    if(row<M && col<N){
        for(int i=0;i<K;++i){
            sum+=A[row*K+i]*B[i*N+col];
        }
        C[N*row+col]=sum;
    }
}

void mm_gpu(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {

    Timer timer;

    // Allocate GPU memory
    startTime(&timer);
    float *A_d,*B_d,*C_d;
    cudaMalloc((void**)&A_d,sizeof(float)*M*K);
    cudaMalloc((void**)&B_d,sizeof(float)*N*K);
    cudaMalloc((void**)&C_d,sizeof(float)*M*N);
    // TODO






    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    // Copy data to GPU
    startTime(&timer);
    cudaMemcpy(A_d,A,sizeof(float)*M*K,cudaMemcpyHostToDevice);
    cudaMemcpy(B_d,B,sizeof(float)*N*K,cudaMemcpyHostToDevice);
    // TODO






    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");

    // Call kernel
    startTime(&timer);
    dim3 nb_threads_per_block(32,32);
    dim3 nb_blocks((N+nb_threads_per_block.x -1 )/nb_threads_per_block.x,
(M+nb_threads_per_block.y-1)/nb_threads_per_block.y);
    mm_kernel<<<nb_blocks,nb_threads_per_block>>>(A_d,B_d,C_d,M,N,K);

    // TODO






    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time", GREEN);

    // Copy data from GPU
    startTime(&timer);
    cudaMemcpy(C,C_d,sizeof(float)*M*N,cudaMemcpyDeviceToHost);

    // TODO






    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

    // Free GPU memory
    startTime(&timer);
    cudaFree(C_d);
    cudaFree(B_d);
    cudaFree(A_d);
    // TODO






    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");

}

