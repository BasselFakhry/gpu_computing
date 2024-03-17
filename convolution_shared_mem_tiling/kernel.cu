
#include "common.h"

#include "timer.h"

#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))

__constant__ float filter_c[FILTER_DIM][FILTER_DIM];

__global__ void convolution_tiled_kernel(float* input, float* output, unsigned int width, unsigned int height) {

    // TODO
    int row=blockIdx.y * OUT_TILE_DIM +threadIdx.y - FILTER_RADIUS;
    int col=blockIdx.x*OUT_TILE_DIM+threadIdx.x - FILTER_RADIUS;
    __shared__ float input_s[IN_TILE_DIM][IN_TILE_DIM];
    if(row<height && row>=0 && col<width && col>=0){
        input_s[threadIdx.y][threadIdx.x]=input[row*width+col];
    }else{
        input_s[threadIdx.y][threadIdx.x]=0;
    }
    __syncthreads();
    
    if(row<height && col<width ){
        if(threadIdx.x>=FILTER_RADIUS && threadIdx.x<IN_TILE_DIM-FILTER_RADIUS && threadIdx.y>=FILTER_RADIUS && threadIdx.y<IN_TILE_DIM-FILTER_RADIUS){
            float sum=0.0f;
            for(unsigned int i=0;i<FILTER_DIM;++i){
                for(unsigned int j=0;j<FILTER_DIM;++j){
                    int r=threadIdx.y+i-FILTER_RADIUS;
                    int c=threadIdx.x+j-FILTER_RADIUS;
                    sum+=filter_c[i][j]*input_s[r][c];
                }
            }
            output[row*width+col]=sum;
        }
    }
}
void copyFilterToGPU(float filter[][FILTER_DIM]) {

    // Copy filter to constant memory
    cudaMemcpyToSymbol(filter_c,filter,sizeof(float)*FILTER_DIM*FILTER_DIM);
    // TODO

}

void convolution_tiled_gpu(float* input_d, float* output_d, unsigned int width, unsigned int height) {
    // Call kernel
    dim3 nb_threads_per_block(IN_TILE_DIM,IN_TILE_DIM);
    dim3 nb_of_blocks((width+OUT_TILE_DIM-1)/OUT_TILE_DIM,
                        (height+OUT_TILE_DIM-1)/OUT_TILE_DIM);
    convolution_tiled_kernel<<<nb_of_blocks,nb_threads_per_block>>>(input_d,output_d,width,height);
    // TODO



}

