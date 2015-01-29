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
__global__ void output_kernel(float *input, float *output, float *input_dx, float *input_dy,
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
    int xout_end = output_h;

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

__global__ void gradInput_kernel(float* gradInput, float* gradOutput, float* input_dx, float* input_dy,
                                 int input_n, int input_h, int input_w, int output_h, int output_w, 
                                 int kH, int kW){
   float* ptr_gradInput_plane = gradInput + blockIdx.x * input_w * input_h;
   float* ptr_gradOutput_plane = gradOutput + blockIdx.x * output_w * output_h;
   float* ptr_output_plane_dx = output_dx + blockIdx.x * output_w * output_h;
   float* ptr_output_plane_dy = output_dy + blockIdx.x * output_w * output_h;

   int xin_start = threadIdx.x * kW;
   int yin_start = threadIdx.y * kH;
   const int xin_step = blockDim.x * kW;
   const int yin_step = blockDim.y * kH;
   int xin_end = (input_w/kW) * kW;
   int yin_end = (input_h/kH) * kH;
    
   int xout = threadIdx.x;
   int yout = threadIdx.y;
   const int xout_step = blockDim.x;
   const int yout_step = blockDim.y;

   for (int yin = yin_start; yin < yin_end; yin += yin_step){
       for (int xin = xin_start; xin < xin_end; xin += xin_step){
           float* ptr_gradInput = ptr_gradInput_plane + xin + yin * input_w;
           float* ptr_gradOuput = ptr_gradOutput_plane + xout + yout * output_w;
           float* ptr_output_dx = ptr_output_plane_dx + xout + yout * output_w;
           float* ptr_output_dy = ptr_output_plane_dy + xout + yout * output_w;

           for (int ky = 0; ky < kH && yin + ky < input_h; ky++){
            for (int kx = 0; kx < kW && xin + kx < input_w; kx++){
                float* ptr_gradInput_pool = ptr_gradInput + kx + ky * input_w;
                if(kx == *ptr_output_dx-1 && ky == *ptr_output_dy-1)
                    *ptr_gradInput_pool = *ptr_gradOuput;
                else
                    *ptr_gradInput_pool = 0;
            } // end for kx
           } // end for ky
           xout += xout + xout_step;
       } // end for xin
       yout += yout_step;
   } // end for yin
}


static int cunn_SpatialMaxPoolingPos_updateOutput(lua_State *L){
    THCudaTensor* input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
    THCudaTensor* output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output_p", "torch.CudaTensor");
    THCudaTensor* dx = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output_dx", "torch.CudaTensor");
    THCudaTensor* dy = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output_dy", "torch.CudaTensor");

    int kW = luaT_getfieldcheckint(L, 1, "kW");
    int kH = luaT_getfieldcheckint(L, 1, "kH");
    float* output_data;
    float* output_dx;
    float* output_dy;
    float* input_data;

    long nInputCols = input -> size[3];
    long nInputRows = input -> size[2];
    long nInputPlane = input -> size[1];
    long nBatch = input -> size[0];
    long nOutputCols = nInputCols / kW;
    long nOutputRows = nInputRows / kH;

    luaL_argcheck(L, input->size[1] == nInputPlane, 2, "invalid number of input planes");
    luaL_argcheck(L, nInputCols >= kW && nInputRows >= kH, 2, "input image smaller than kernel size");

    input = THCudaTensor_newContiguous(input);
    input_data = THCudaTensor_data(input);

    THCudaTensor_resize4d(output, nBatch, nInputPlane, nOutputRows, nOutputCols);
    THCudaTensor_resize4d(dx, nBatch, nInputPlane, nOutputRows, nOutputCols);
    THCudaTensor_resize4d(dy, nBatch, nInputPlane, nOutputRows, nOutputCols);

    output_data = THCudaTensor_data(output);
    output_dx = THCudaTensor_data(dx);
    output_dy = THCudaTensor_data(dy);

    dim3 blocks(nInputPlane*nBatch, 1);
    dim3 threads(32,8);
    
    output_kernel <<<blocks, threads>>> (input_data, output_data, output_dx, output_dy, nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols, kH, kW);

    THCudaTensor_free(input);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("error in SpatialMaxPoolingPos.updateOutput: %s\n", cudaGetErrorString(err));
        THError("aborting");
    }
    return 1;
}

static int cunn_SpatialMaxPoolingPos_updateGradInput(lua_State *L){
    THCudaTensor* gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
    THCudaTensor* gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
    THCudaTensor* input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
    THCudaTensor* output_dx = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output_dx", "torch.CudaTensor");
    THCudaTensor* output_dy = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output_dy", "torch.CudaTensor");
    
    int kW = luaT_getfieldcheckint(L, 1, "kW");
    int kH = luaT_getfieldcheckint(L, 1, "kH");

    float* gradInput_data;
    float* gradOutput_data;
    float* gradOutput_dx_data;
    float* gradOutput_dy_data;

    long nInputCols = input->size[3];
    long nInputRows = input->size[2];
    long nInputPlane = input->size[1];
    long nbatch = input->size[0];
    long nOutputCols = gradOutput->size[3];
    long nOutputRows = gradOutput->size[2];

    THCudaTensor_resizeAs(gradInput, input);
    THCudaTensor_zero(gradInput);

    gradOutput_data = THCudaTensor_data(gradOutput);
    gradOutput_dx_data = THCudaTensor_data(output_dx);
    gradOutput_dy_data = THCudaTensor_data(output_dy);
    gradInput_data = THCudaTensor_data(gradInput);

    dim3 blocks(nInputPlane*nbatch, 1);
    dim3 threads(32,8);
    
    gradInput_kernel<<<blocks, threads>>> (gradInput_data, gradOutput_data, gradOutput_dx_data, gradOutput_dy_data,
                                           nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols, kH, kW);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("error in SSMPoolingOffsets_updateGradInput: %s\n", cudaGetErrorString(err));
        THError("aborting");
    }
    return 1;
}

static const struct luaL_Reg cunn_SpatialMaxPoolingPos__ [] = {
  {"SpatialMaxPoolingPos_updateOutput", cunn_SpatialMaxPoolingPos_updateOutput},
  {"SpatialMaxPoolingPos_updateGradInput", cunn_SpatialMaxPoolingPos_updateGradInput},
  {NULL, NULL}
};

void cunn_SpatialMaxPoolingPos_init(lua_State* L){
    luaL_openlib(L, "jz", cunn_SpatialMaxPoolingPos__, 0);
}
