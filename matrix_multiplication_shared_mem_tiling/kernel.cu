
#include "common.h"

#include "timer.h"

#define TILE_DIM 32

__global__ void mm_tiled_kernel(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {

    // TODO
    unsigned int row=blockDim.y*blockIdx.y+threadIdx.y;
    unsigned int col=blockDim.x*blockIdx.x+threadIdx.x;

    __shared__ float A_s[TILE_DIM][TILE_DIM];
    __shared__ float B_s[TILE_DIM][TILE_DIM];

    float sum=0.0f;
    
    for(unsigned int tile=0;tile<(K+TILE_DIM-1)/TILE_DIM;++tile){
         unsigned int col_t_a=tile*TILE_DIM+threadIdx.x;
         unsigned int row_t_a=threadIdx.y+row/TILE_DIM;
         unsigned int col_t_b=threadIdx.x+col/TILE_DIM;
         unsigned int row_t_b=tile*TILE_DIM+threadIdx.y;
         if(col_t_a<K && row_t_a<M && row<M )
            A_s[threadIdx.y][threadIdx.x]=A[row*K+tile*TILE_DIM+threadIdx.x];
	     else A_s[threadIdx.y][threadIdx.x]=0.0f;
         if(col_t_b<N && row_t_b<K && col<N)
            B_s[threadIdx.y][threadIdx.x]=B[(threadIdx.y+tile*TILE_DIM)*N+col];
         else B_s[threadIdx.y][threadIdx.x]=0.0f;
         __syncthreads();
            
         for(unsigned int i=0;i<TILE_DIM;++i){
             sum+=A_s[threadIdx.y][i]*B_s[i][threadIdx.x];
         }
            
         __syncthreads();
   }
   if(row<M && col<N)
       	 C[row*N+col]=sum;
    
    






}

void mm_gpu(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {

    Timer timer;

    // Allocate GPU memory
    startTime(&timer);
    float *A_d,*B_d,*C_d;
    cudaMalloc((void**)&A_d,sizeof(float)*M*K);
    cudaMalloc((void**)&B_d,sizeof(float)*K*N);
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
    dim3 nb_blocks((N+nb_threads_per_block.x-1)/nb_threads_per_block.x,
                    (M+nb_threads_per_block.y-1)/nb_threads_per_block.y);
    mm_tiled_kernel<<<nb_blocks,nb_threads_per_block>>>(A_d,B_d,C_d,M,N,K);    
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
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    // TODO






    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");

}

