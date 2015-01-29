#include "luaT.h"
#include "THC.h"
#include "cuda.h"

#include <thrust/transform.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

#define CUDA_MAX_THREADS 1024   // this is safe, in reality 256 is our limit

//no-overlap
__global__ void output_kernel_unpooling(float *input, float *output, float *input_dx, float *input_dy,
                              int input_n, int input_h, int input_w, int output_h, int output_w,
                              int kH, int kW){
    float* ptr_input_plane = input + blockIdx.x * input_w * input_h;
    float* ptr_input_plane_dx = input_dx + blockIdx.x * input_w * input_h;
    float* ptr_input_plane_dy = input_dy + blockIdx.x * input_w * input_h;
    float* ptr_output_plane = output + blockIdx.x * output_w * output_h;

    int xin = threadIdx.x;
    int yin = threadIdx.y;
    const int xin_step = blockDim.x;
    const int yin_step = blockDim.y;
    int xout_start = threadIdx.x * kW;
    int yout_start = threadIdx.y * kH;
    const int xout_step = blockDim.x * kW;
    const int yout_step = blockDim.y * kH;
    int xout_end = output_w;
    int yout_end = output_h;

    for (int yout = yout_start; yout < yout_end; yout += yout_step){
        for (int xout = xout_start; xout < xout_end; xout += xout_step){
           float* ptr_output = ptr_output_plane + xout + yout * output_w;
           float* ptr_input = ptr_input_plane + xin + yin * input_w;
           float* ptr_input_dx = ptr_input_plane_dx + xin + yin * input_w;
           float* ptr_input_dy = ptr_input_plane_dy + xin + yin * input_w;

           if (xin < input_w && yin < input_h){
              for (int ky = 0; ky < kH && yout + ky < output_h; ky++) {
                 for (int kx = 0; kx < kW && xout + kx < output_w; kx++){
                    float* ptr_output_pool = ptr_output + kx + ky * output_w;
                    if (ky == *ptr_input_dy-1 && kx == *ptr_input_dx-1)
                       *ptr_output_pool = *ptr_input;
                    else
                       *ptr_output_pool = 0;
                 }   
              }     
           } // endif
           xin += xin + xin_step;
        } // end xout
        yin += yin + yin_step;
    } // end yout
}

__global__ void gradInput_kernel_unpooling(float* gradInput_p, float* gradOutput, float* input_dx, float* input_dy,
                                 int input_n, int input_h, int input_w, int output_h, int output_w, 
                                 int kH, int kW){
   float* ptr_gradInput_plane_p = gradInput_p + blockIdx.x * input_w * input_h;
   float* ptr_gradOutput_plane = gradOutput + blockIdx.x * output_w * output_h;
   float* ptr_input_plane_dx = input_dx + blockIdx.x * input_w * input_h;
   float* ptr_input_plane_dy = input_dy + blockIdx.x * input_w * input_h;

   //int xout_start = threadIdx.x * kW;
   //int yout_start = threadIdx.x * kH;
   const int xout_step = blockDim.x * kW;
   const int yout_step = blockDim.x * kH;
   //int xout_end = output_w;
   //int yout_end = output_h;
   
   int xout = threadIdx.x * kW;
   int yout = threadIdx.y * kH;
   int xin_start = threadIdx.x;
   int yin_start = threadIdx.y;
   const int xin_step = blockDim.x;
   const int yin_step = blockDim.y;
   int xin_end = input_w;
   int yin_end = input_h;

   for (int yin = yin_start; yin < yin_end; yin += yin_step){
       for (int xin = xin_start; xin < xin_end; xin += xin_step){
           float* ptr_gradInput_p = ptr_gradInput_plane_p + xin + yin * input_w;
           float* ptr_gradOutput = ptr_gradOutput_plane + xout + yout * output_w;
           float* ptr_input_dx = ptr_input_plane_dx + xin + yin * input_w;
           float* ptr_input_dy = ptr_input_plane_dy + xin + yin * input_w;

           *ptr_gradInput_p = *(ptr_gradOutput + (int)*ptr_input_dx - 1 + ((int)*ptr_input_dy - 1) * output_w);

           xout += xout + xout_step;
       } // end for xin
       yout += yout_step;
   } // end for yin
}


static int cunn_SpatialMaxUnpoolingPos_updateOutput(lua_State *L){
    THCudaTensor* input_p = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
    THCudaTensor* output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
    THCudaTensor* input_dx = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "input_dx", "torch.CudaTensor");
    THCudaTensor* input_dy = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "input_dy", "torch.CudaTensor");

    int kW = luaT_getfieldcheckint(L, 1, "kW");
    int kH = luaT_getfieldcheckint(L, 1, "kH");
    float* output_data;
    float* input_dx_data;
    float* input_dy_data;
    float* input_p_data;

    long nInputCols = input_p -> size[3];
    long nInputRows = input_p -> size[2];
    long nInputPlane = input_p -> size[1];
    long nBatch = input_p -> size[0];
    long nOutputCols = nInputCols * kW;
    long nOutputRows = nInputRows * kH;

    luaL_argcheck(L, nOutputCols >= kW && nOutputRows >= kH, 2, "input_p image smaller than kernel size");

    input_p = THCudaTensor_newContiguous(input_p);
    input_p_data = THCudaTensor_data(input_p);

    THCudaTensor_resize4d(output, nBatch, nInputPlane, nOutputRows, nOutputCols);
    THCudaTensor_resize4d(input_dx, nBatch, nInputPlane, nOutputRows, nOutputCols);
    THCudaTensor_resize4d(input_dy, nBatch, nInputPlane, nOutputRows, nOutputCols);

    output_data = THCudaTensor_data(output);
    input_dx_data = THCudaTensor_data(input_dx);
    input_dy_data = THCudaTensor_data(input_dy);

    dim3 blocks(nInputPlane*nBatch, 1);
    dim3 threads(32,8);
    
    output_kernel_unpooling <<<blocks, threads>>> (input_p_data, output_data, input_dx_data, input_dy_data, nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols, kH, kW);

    THCudaTensor_free(input_p);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("error in SpatialMaxPoolingPos.updateOutput: %s\n", cudaGetErrorString(err));
        THError("aborting");
    }
    return 1;
}

static int cunn_SpatialMaxUnpoolingPos_updateGradInput(lua_State *L){
    THCudaTensor* gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
    THCudaTensor* gradInput_p = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput_p", "torch.CudaTensor");
    THCudaTensor* input_p = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
    THCudaTensor* input_dx = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "input_dx", "torch.CudaTensor");
    THCudaTensor* input_dy = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "input_dy", "torch.CudaTensor");
    
    int kW = luaT_getfieldcheckint(L, 1, "kW");
    int kH = luaT_getfieldcheckint(L, 1, "kH");

    float* gradInput_p_data;
    float* gradOutput_data;
    //float* gradOutput_dx_data;
    //float* gradOutput_dy_data;
    float* input_dx_data;
    float* input_dy_data;

    long nInputCols = input_p->size[3];
    long nInputRows = input_p->size[2];
    long nInputPlane = input_p->size[1];
    long nbatch = input_p->size[0];
    long nOutputCols = gradOutput->size[3];
    long nOutputRows = gradOutput->size[2];

    THCudaTensor_resizeAs(gradInput_p, input_p);
    THCudaTensor_zero(gradInput_p);

    gradOutput_data = THCudaTensor_data(gradOutput);
    input_dx_data = THCudaTensor_data(input_dx);
    input_dy_data = THCudaTensor_data(input_dy);
    gradInput_p_data = THCudaTensor_data(gradInput_p);

    dim3 blocks(nInputPlane*nbatch, 1);
    dim3 threads(32,8);
    
    gradInput_kernel_unpooling<<<blocks, threads>>> (gradInput_p_data, gradOutput_data, input_dx_data, input_dy_data,
                                           nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols, kH, kW);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("error in SSMPoolingOffsets_updateGradInput: %s\n", cudaGetErrorString(err));
        THError("aborting");
    }
    return 1;
}

static const struct luaL_Reg cunn_SpatialMaxUnpoolingPos__ [] = {
  {"SpatialMaxUnpoolingPos_updateOutput", cunn_SpatialMaxUnpoolingPos_updateOutput},
  {"SpatialMaxUnpoolingPos_updateGradInput", cunn_SpatialMaxUnpoolingPos_updateGradInput},
  {NULL, NULL}
};

void cunn_SpatialMaxUnpoolingPos_init(lua_State* L){
    luaL_openlib(L, "jz", cunn_SpatialMaxUnpoolingPos__, 0);
}
