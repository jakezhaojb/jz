#include "luaT.h"
#include "THC.h"
#include "cuda.h"

#include <thrust/transform.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

#define CUDA_MAX_THREADS 1024   // this is safe, in reality 256 is our limit

// no-overlop
__global__ void output_kernel(float *input, float* output, float* weight, int input_h, int input_w,
                              int output_h, int output_w, int kW, int kH){
   float* ptr_input_plane = input + (blockIdx.x + gridDim.x * blockIdx.y) * input_w * input_h;
   float* ptr_output_plane = output + (blockIdx.x + gridDim.x * blockIdx.y) * output_w * output_h;
   float* weight_plane = weight + blockIdx.x * kW * kH;

   int xout = threadIdx.x;
   int yout = threadIdx.y; 
   const int xout_step = blockDim.x;
   const int yout_step = blockDim.y;
   int xin_start = threadIdx.x * kW;
   int yin_start = threadIdx.y * kH;
   const int xin_step = blockDim.x * kW;
   const int yin_step = blockDim.y * kH;
   int xin_end = (input_w/kW) * kW;  //TODO could this be right?
   int yin_end = (input_h/kH) * kH;

   for (int yin = yin_start; yin < yin_end ; yin += yin_step){
      for (int xin = xin_start; xin < xin_end; xin += xin_step){
         float* ptr_input = ptr_input_plane + xin + yin * input_w;
         float* ptr_output = ptr_output_plane + xout + yout * output_w;
         
         if (xout < output_w && yout < output_h){
            for (int ky = 0; ky < kH && yin + ky < input_h; ky++){
               for (int kx = 0; kx < kW && xin + kx < input_w; kx++){
                  float* weight_plane_elem = weight_plane + kx + ky * kW;
                  float* ptr_input_elem = ptr_input + kx + ky * input_w;
                  *ptr_output += (*ptr_input_elem) * (*weight_plane_elem);
               }   
            }   
         } // end if
         xout += xout_step;
      } // end for xout
      yout += yout_step;
   } // end for yout
}


__global__ void grad_input_kernel(float* input, float* grad_output, float* grad_input, float* weight, int input_h,
                                 int input_w, int output_h, int output_w, int kW, int kH){
   float* ptr_grad_output_plane = grad_output + (blockIdx.x + gridDim.x * blockIdx.y) * output_w * output_h;
   float* ptr_grad_input_plane = grad_input + (blockIdx.x + gridDim.x * blockIdx.y) * input_w * input_h;
   float* weight_plane = weight + blockIdx.x * kW * kH;

    int xout = threadIdx.x;
    int yout = threadIdx.y;
    const int xout_step = blockDim.x;
    const int yout_step = blockDim.y;
    int xin_start = threadIdx.x * kW;
    int yin_start = threadIdx.y * kH;
    const int xin_step = blockDim.x * kW;
    const int yin_step = blockDim.y * kH;
    int xin_end = (input_w/kW) * kW;  //TODO could this be right?
    int yin_end = (input_h/kH) * kH;

   for (int yin = yin_start; yin < yin_end; yin += yin_step){
       for (int xin = xin_start; xin < xin_end; xin += xin_step){
           float* ptr_grad_input = ptr_grad_input_plane + xin + yin * input_w;
           float* ptr_grad_output_elem = ptr_grad_output_plane + xout + yout * output_w;

         if (xout < output_w && yout < output_h){
           for (int ky = 0; ky < kH && yin + ky < input_h; ky++){
            for (int kx = 0; kx < kW && xin + kx < input_w; kx++){
                float* ptr_grad_input_elem = ptr_grad_input + kx + ky * input_w;
                float* weight_plane_elem = weight_plane + kx + ky * kW;
                *ptr_grad_input_elem = (*weight_plane_elem) * (*ptr_grad_output_elem);
            } // end for kx
           } // end for ky
         }
         xout += xout_step;
       } // end for xin
      yout += yout_step;
   } // end for yin
}


__global__ void accGrad_kernel(float* input, float* grad_output, float* grad_weight, float* weight, int input_h,
                               int input_w, int output_h, int output_w, int kW, int kH, float scale){
   float* ptr_input_plane = input + (blockIdx.x + gridDim.x * blockIdx.y) * input_w * input_h;
   float* ptr_grad_output_plane = grad_output + (blockIdx.x + gridDim.x * blockIdx.y) * output_w * output_h;
   float* grad_weight_plane = grad_weight + blockIdx.x * kW * kH;

   int xout = threadIdx.x;
   int yout = threadIdx.y;
   const int xout_step = blockDim.x;
   const int yout_step = blockDim.y;
   int xin_start = threadIdx.x * kW;
   int yin_start = threadIdx.y * kH;
   const int xin_step = blockDim.x * kW;
   const int yin_step = blockDim.y * kH;
   int xin_end = (input_w/kW) * kW;  //TODO could this be right?
   int yin_end = (input_h/kH) * kH;

   for (int yin = yin_start; yin < yin_end; yin += yin_step){
      for (int xin = xin_start; xin < xin_end; xin += xin_step){
         float* ptr_grad_output_plane_elem = ptr_grad_output_plane + xout + yout * output_w;
         float* ptr_input_plane_elem = ptr_input_plane + xin + yin * input_w;
         if (xout < input_w && yout < input_h){
            for (int ky = 0; ky < kH && yin + ky < input_h; ky++){
               for (int kx = 0; kx < kW && xin + kx < input_w; kx++){
                  float* grad_weight_plane_elem = grad_weight_plane + kx + ky * kW;
                  float* ptr_input_plane_elem_elem = ptr_input_plane_elem + kx + ky * input_w;
                  float tmp = scale * (*ptr_input_plane_elem_elem) * (*ptr_grad_output_plane_elem);
                  atomicAdd(grad_weight_plane_elem, tmp);
               }  
            }  
         }
         xout += xout_step;
      } // end for xin   
      yout += yout_step;
   } // end for yin
}


static int cunn_SpatialMlpPooling_updateOutput(lua_State *L){
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
    long nOutputCols = nInputCols / kW;
    long nOutputRows = nInputRows / kH;

    luaL_argcheck(L, nInputCols >= kW && nInputRows >= kH, 2, "input image smaller than kernel size");

    input = THCudaTensor_newContiguous(input);

    input_data = THCudaTensor_data(input);
    weight_data = THCudaTensor_data(weight);

    THCudaTensor_resize4d(output, nBatch, nInputPlane, nOutputRows, nOutputCols);
    THCudaTensor_zero(output);

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


static int cunn_SpatialMlpPooling_updateGradInput(lua_State *L){
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
    long nOutputCols = nInputCols / kW;
    long nOutputRows = nInputRows / kH;

    luaL_argcheck(L, nInputCols >= kW && nInputRows >= kH, 2, "input image smaller than kernel size");

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
    
    grad_input_kernel <<<blocks, threads>>> (input_data, gradOutput_data, gradInput_data, weight_data, nInputRows, nInputCols, nOutputRows, nOutputCols, kW, kH);

    THCudaTensor_free(input);
    THCudaTensor_free(gradOutput);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("error in SpatialMaxPoolingPos.updateOutput: %s\n", cudaGetErrorString(err));
        THError("aborting");
    }
    return 1;
}

static int cunn_SpatialMlpPooling_accGradParameters(lua_State *L){
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
    long nOutputCols = nInputCols / kW;
    long nOutputRows = nInputRows / kH;

    luaL_argcheck(L, nInputCols >= kW && nInputRows >= kH, 2, "input image smaller than kernel size");

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

static const struct luaL_Reg cunn_SpatialMlpPooling__ [] = {
   {"SpatialMlpPooling_updateOutput", cunn_SpatialMlpPooling_updateOutput},
   {"SpatialMlpPooling_updateGradInput", cunn_SpatialMlpPooling_updateGradInput},
   {"SpatialMlpPooling_accGradParameters", cunn_SpatialMlpPooling_accGradParameters},
   {NULL, NULL}
};

void cunn_SpatialMlpPooling_init(lua_State* L){
   luaL_openlib(L, "jz", cunn_SpatialMlpPooling__, 0) ;
}
