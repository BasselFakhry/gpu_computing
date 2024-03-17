
#include "common.h"
#define c_factor 4
#include "timer.h"

__global__ void histogram_private_kernel(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {

    // TODO
    unsigned int i=blockDim.x*blockIdx.x+threadIdx.x;
    __shared__ unsigned int hist[NUM_BINS];
    if(threadIdx.x<NUM_BINS)
        hist[threadIdx.x]=0;
    __syncthreads();
    if(i<width*height){
        unsigned int b=image[i];
        atomicAdd(&hist[b],1); 
    }
    __syncthreads();
    if(threadIdx.x==0){
        for(int j=0;j<NUM_BINS;++j)
            atomicAdd(&bins[j],hist[j]);
    }

}

void histogram_gpu_private(unsigned char* image_d, unsigned int* bins_d, unsigned int width, unsigned int height) {

    // TODO
    unsigned int nb_threads_per_block=1024;
    unsigned int nb_blocks= (height*width+nb_threads_per_block-1)/nb_threads_per_block;
    histogram_private_kernel<<<nb_blocks,nb_threads_per_block>>>(image_d,bins_d,width,height);

}

__global__ void histogram_private_coarse_kernel(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {

    // TODO
    unsigned int i=blockDim.x*blockIdx.x*c_factor+threadIdx.x;
    __shared__ unsigned int hist[NUM_BINS];
    if(threadIdx.x<NUM_BINS)
        hist[threadIdx.x]=0;
    __syncthreads();
    if(i<height*width){
        unsigned int bound=(blockDim.x*c_factor)*(blockIdx.x+1);
        for(int j=i;j<bound&&j<height*width;j+=blockDim.x){
            unsigned int b=image[j];
            atomicAdd(&hist[b],1); 
        }
    }
    __syncthreads();
    if(threadIdx.x==0){
        for(int j=0;j<NUM_BINS;++j)
            atomicAdd(&bins[j],hist[j]);
    }
    

}

void histogram_gpu_private_coarse(unsigned char* image_d, unsigned int* bins_d, unsigned int width, unsigned int height) {

    // TODO
    unsigned int nb_threads_per_block=1024;
    unsigned int nb_blocks=(height*width+(nb_threads_per_block*c_factor-1))/(nb_threads_per_block*c_factor);

    histogram_private_coarse_kernel<<<nb_blocks,nb_threads_per_block>>>(image_d, bins_d, width, height);


}

