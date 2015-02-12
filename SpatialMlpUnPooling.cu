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
__global__ void output_kernel(float *input, float* output, float* weight, int input_h, int input_w,
                              int output_h, int output_w, int kW, int kH){
   float* ptr_input_plane = input + (blockIdx.x + gridDim.x * blockIdx.y) * input_w * input_h;
   float* ptr_output_plane = output + (blockIdx.x + gridDim.x * blockIdx.y) * output_w * output_h;
   float* weight_plane = weight + blockIdx.x * kW * kH;

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

   for (int yout = yout_start; yout < yout_end ; yout += yout_step){
      for (int xout = xout_start; xout < xout_end; xout += xout_step){
         float* ptr_output = ptr_output_plane + xout + yout * output_w;
         float* ptr_input = ptr_input_plane + xin + yin * input_w;
         
         if (xin < input_w && yin < input_h){
            for (int ky = 0; ky < kH && yout + ky < output_h; ky++){
               for (int kx = 0; kx < kW && xout + kx < output_w; kx++){
                  float* ptr_output_elem = ptr_output + kx + ky * output_w;
                  float* weight_plane_elem = weight_plane + kx + ky * kW;
                  *ptr_output_elem = (*ptr_input) * (*weight_plane_elem);
               }   
            }   
         } // end if
         xin += xin_step;
      } // end for xin
      yin += yin_step;
   } // end for yin
}

__global__ void gradInput_kernel(float* input, float* grad_output, float* grad_input, float* weight, int input_h,
                                 int input_w, int output_h, int output_w, int kW, int kH){
   float* ptr_grad_output_plane = grad_output + (blockIdx.x + gridDim.x * blockIdx.y) * output_w * output_h;
   float* ptr_grad_input_plane = grad_input + (blockIdx.x + gridDim.x * blockIdx.y) * input_w * input_h;
   float* weight_plane = weight + blockIdx.x * kW * kH;

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
         float* ptr_grad_input_plane_elem = ptr_grad_input_plane + xin + yin * input_w;
         float* ptr_grad_output_plane_elem = ptr_grad_output_plane + xout + yout * output_w;
         if (xin < input_w && yin < input_h){
            for (int ky = 0; ky < kH && yout + ky < output_h; ky++){
               for (int kx = 0; kx < kW && xout + kx < output_w; kx++){
                  float* weight_plane_elem = weight_plane + kx + ky * kW;
                  float* ptr_grad_output_plane_elem_ = ptr_grad_output_plane_elem + kx + ky * kW;
                  *ptr_grad_input_plane_elem += (*weight_plane_elem) * (*ptr_grad_output_plane_elem_);
               }  
            }  
         }
         xin += xin_step;
      } // end for xout   
      yin += yin_step;
   } // end for yout
}

__global__ void accGrad_kernel(float* input, float* grad_output, float* grad_weight, float* weight, int input_h,
                               int input_w, int output_h, int output_w, int kW, int kH, float scale){
   float* ptr_input_plane = input + (blockIdx.x + gridDim.x * blockIdx.y) * input_w * input_h;
   float* ptr_grad_output_plane = grad_output + (blockIdx.x + gridDim.x * blockIdx.y) * output_w * output_h;
   float* grad_weight_plane = grad_weight + blockIdx.x * kW * kH;

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
         float* ptr_grad_output_plane_elem = ptr_grad_output_plane + xout + yout * output_w;
         float* ptr_input_plane_elem = ptr_input_plane + xin + yin * input_w;
         if (xin < input_w && yin < input_h){
            for (int ky = 0; ky < kH && yout + ky < output_h; ky++){
               for (int kx = 0; kx < kW && xout + kx < output_w; kx++){
                  float* grad_weight_plane_elem = grad_weight_plane + kx + ky * kW;
                  float* ptr_grad_output_plane_elem_ = ptr_grad_output_plane_elem + kx + ky * kW;
                  float tmp = scale * (*ptr_input_plane_elem) * (*ptr_grad_output_plane_elem_);
                  atomicAdd(grad_weight_plane_elem, tmp);
               }  
            }  
         }
         xin += xin_step;
      } // end for xout   
      yin += yin_step;
   } // end for yout
}

static int cunn_SpatialMlpUnPooling_updateOutput(lua_State *L){
    THCudaTensor* input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
    THCudaTensor* output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
    THCudaTensor* weight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
    int kW = luaT_getfieldcheckint(L, 1, "kW");
    int kH = luaT_getfieldcheckint(L, 1, "kH");
    float* output_data;
    float* input_data;
    float* weight_data;

    long nInputCols = input -> size[3];
    long nInputRows = input -> size[2];
    long nInputPlane = input -> size[1];
    long nBatch = input -> size[0];
    long nOutputCols = nInputCols * kW;
    long nOutputRows = nInputRows * kH;

    luaL_argcheck(L, nOutputCols >= kW && nOutputRows >= kH, 2, "input image smaller than kernel size");

    input = THCudaTensor_newContiguous(input);

    input_data = THCudaTensor_data(input);
    weight_data = THCudaTensor_data(weight);

    THCudaTensor_resize4d(output, nBatch, nInputPlane, nOutputRows, nOutputCols);

    output_data = THCudaTensor_data(output);

    dim3 blocks(nInputPlane, nBatch);
    dim3 threads(32,8);
    
    output_kernel <<<blocks, threads>>> (input_data, output_data, weight_data, nInputRows, nInputCols, nOutputRows, nOutputCols, kW, kH);

    THCudaTensor_free(input);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("error in SpatialMaxPoolingPos.updateOutput: %s\n", cudaGetErrorString(err));
        THError("aborting");
    }
    return 1;
}

static int cunn_SpatialMlpUnPooling_updateGradInput(lua_State *L){
    THCudaTensor* input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
    THCudaTensor* gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
    THCudaTensor* weight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
    THCudaTensor* gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
    int kW = luaT_getfieldcheckint(L, 1, "kW");
    int kH = luaT_getfieldcheckint(L, 1, "kH");

    float* input_data;
    float* weight_data;
    float* gradOutput_data;
    float* gradInput_data;

    long nInputCols = input -> size[3];
    long nInputRows = input -> size[2];
    long nInputPlane = input -> size[1];
    long nBatch = input -> size[0];
    long nOutputCols = nInputCols * kW;
    long nOutputRows = nInputRows * kH;

    luaL_argcheck(L, nOutputCols >= kW && nOutputRows >= kH, 2, "input image smaller than kernel size");

    THCudaTensor_resizeAs(gradInput, input);
    THCudaTensor_zero(gradInput);

    input = THCudaTensor_newContiguous(input);
    gradOutput = THCudaTensor_newContiguous(gradOutput);
    input_data = THCudaTensor_data(input);
    gradOutput_data = THCudaTensor_data(gradOutput);
    gradInput_data = THCudaTensor_data(gradInput);
    weight_data = THCudaTensor_data(weight);

    dim3 blocks(nInputPlane, nBatch);
    dim3 threads(32,8);
    
    gradInput_kernel <<<blocks, threads>>> (input_data, gradOutput_data, gradInput_data, weight_data, nInputRows, nInputCols, nOutputRows, nOutputCols, kW, kH);

    THCudaTensor_free(input);
    THCudaTensor_free(gradOutput);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("error in SpatialMaxPoolingPos.updateOutput: %s\n", cudaGetErrorString(err));
        THError("aborting");
    }
    return 1;
}

static int cunn_SpatialMlpUnPooling_accGradParameters(lua_State *L){
    THCudaTensor* input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
    THCudaTensor* gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
    THCudaTensor* weight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
    THCudaTensor* gradWeight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradWeight", "torch.CudaTensor");
    float scale = luaL_checknumber(L, 4); 
    int kW = luaT_getfieldcheckint(L, 1, "kW");
    int kH = luaT_getfieldcheckint(L, 1, "kH");

    float* input_data;
    float* weight_data;
    float* gradOutput_data;
    float* gradWeight_data;

    long nInputCols = input -> size[3];
    long nInputRows = input -> size[2];
    long nInputPlane = input -> size[1];
    long nBatch = input -> size[0];
    long nOutputCols = nInputCols * kW;
    long nOutputRows = nInputRows * kH;

    luaL_argcheck(L, nOutputCols >= kW && nOutputRows >= kH, 2, "input image smaller than kernel size");

    THCudaTensor_resizeAs(gradWeight, input);
    THCudaTensor_zero(gradWeight);

    input = THCudaTensor_newContiguous(input);
    gradOutput = THCudaTensor_newContiguous(gradOutput);

    input_data = THCudaTensor_data(input);
    gradOutput_data = THCudaTensor_data(gradOutput);
    gradWeight_data = THCudaTensor_data(gradWeight);
    weight_data = THCudaTensor_data(weight);

    dim3 blocks(nInputPlane, nBatch);
    dim3 threads(32,8);
    
    accGrad_kernel <<<blocks, threads>>> (input_data, gradOutput_data, gradWeight_data, weight_data, nInputRows, nInputCols, nOutputRows, nOutputCols, kW, kH, scale);

    THCudaTensor_free(input);
    THCudaTensor_free(gradOutput);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("error in SpatialMaxPoolingPos.updateOutput: %s\n", cudaGetErrorString(err));
        THError("aborting");
    }
    return 1;
}

static const struct luaL_Reg cunn_SpatialMlpUnPooling__ [] = {
   {"SpatialMlpUnPooling_updateOutput", cunn_SpatialMlpUnPooling_updateOutput},
   {"SpatialMlpUnPooling_updateGradInput", cunn_SpatialMlpUnPooling_updateGradInput},
   {"SpatialMlpUnPooling_accGradParameters", cunn_SpatialMlpUnPooling_accGradParameters},
   {NULL, NULL}
};

void cunn_SpatialMlpUnPooling_init(lua_State* L){
   luaL_openlib(L, "jz", cunn_SpatialMlpUnPooling__, 0) ;
}
