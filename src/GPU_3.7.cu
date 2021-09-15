#include <iostream>
#include <cuda.h>
#include <cuda_fp16.h>
#include "../inc/cuda_test.h"
#define DATA_TYPE  float

template<typename T>
__global__ void base(T * src, T * dst, int num, T data) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x; 
	if (idx < num) {
	  float tmp_old = src[idx];
	  for(int i = 0; i < 100; i++) { 
	    float tmp = tmp_old  * data;
	    tmp_old = tmp;
	  }
	  dst[idx] = src[idx];
  }
}

template<typename T>
__global__ void latency(T * src, T * dst, int num, T data) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x; 
	if (idx < num) {
	  T tmp_old = src[idx];
	  for(int i = 0; i < 100; i++) { 
	    T tmp = tmp_old / data;
	    tmp_old = tmp;
	  }
	  dst[idx] = tmp_old;
  }
}

template<typename T>
__global__ void select(T * src, T * dst, int num, T data) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x; 
	if (idx < num) {
	  if (idx % 2 == 0) {     
	     float tmp_old = src[idx];
	     for(int i = 0; i < 100; i++) { 
	       float tmp = tmp_old  * data;
	       tmp_old = tmp;
	     }
	     dst[idx] = tmp_old;
	  } else {      
	     float tmp_old = src[idx];
	     for(int i = 0; i < 100; i++) { 
	       float tmp = tmp_old  * data;
	       tmp_old = tmp;
	     }
	     dst[idx] = tmp_old;
	  }
  }
}

template<typename T>
__global__ void blocks(T * src, T * dst, int num, T data) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x; 
	if (idx < num) {
	  float tmp_old = src[idx];
	  for(int i = 0; i < 100; i++) { 
	    float tmp = tmp_old  * data;
	    tmp_old = tmp;
	  }
	  dst[idx] = tmp_old;
  }
}

template<typename T>
__global__ void for_4(T* src1, T* src2, T* dst1, int num) {
   float4 * src = (float4*) src1;
   float4 * dx  = (float4*) src2;
   float4 * dst = (float4*) dst1;
   int i = blockDim.x * blockIdx.x + threadIdx.x;
   int loop = num / 4; 
   float4 tmp;

   for (int t = i; t < loop; t += blockDim.x * blockDim.x) {
      tmp = src[t];
      dst[t].x = (tmp.x >  0.0f) * dx[i].x;
      dst[t].y = (tmp.y >  0.0f) * dx[i].y;
      dst[t].z = (tmp.z >  0.0f) * dx[i].z;
      dst[t].w = (tmp.w >  0.0f) * dx[i].w;
   }
} 

void init(DATA_TYPE *src_h, DATA_TYPE*src2_h, DATA_TYPE* dst_cpu, int num) {
    for (int i = 0; i < num; i++) {
       src_h[i] = -0.99;//i % 3 *( (-1) ^ i); 
       src2_h[i] = -1.25;//i % 3 *( (-1) ^ i); 
    }
}

int main(void) {
   size_t N = 1 << 20;
   float dev = 0;
   cudaSetDevice(dev);
   DATA_TYPE *dst_d_h = (DATA_TYPE*)malloc( N * sizeof(DATA_TYPE));
   DATA_TYPE *src1_h  = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));
   DATA_TYPE *src2_h  = (DATA_TYPE*)malloc( N * sizeof(DATA_TYPE));
   DATA_TYPE *dst_h   = (DATA_TYPE*)malloc( N * sizeof(DATA_TYPE));
   init(src1_h,src2_h, dst_h, N);

   DATA_TYPE* src1_d, * src2_d, *dst_d;
   cudaMalloc((DATA_TYPE**)&src1_d,N * sizeof(DATA_TYPE));
   cudaMalloc((DATA_TYPE**)&src2_d,N * sizeof(DATA_TYPE));
   cudaMalloc((DATA_TYPE**)&dst_d, N * sizeof(DATA_TYPE));
   cudaMemcpy(src1_d, src1_h,   N * sizeof(DATA_TYPE),cudaMemcpyHostToDevice);
   cudaMemcpy(src2_d, src2_h,  N * sizeof(DATA_TYPE),cudaMemcpyHostToDevice);
   int block = 32;
   int grid = (N + block - 1) / block;
   for (int i = 0; i < 1000; i++) { 
     block = 32;
     grid = (N + block - 1) / block;
     latency<DATA_TYPE><<<grid, block>>>(src1_d, dst_d, N, (float)(1000));
     base<DATA_TYPE><<<grid, block>>>(src1_d, dst_d, N, (float)(1/1000));
     select<DATA_TYPE><<<grid, block>>>(src1_d, dst_d, N, (float)(1/1000));
     block = 128;
     grid = (N + block - 1) / block;
     blocks<DATA_TYPE><<<grid, block>>>(src1_d, dst_d, N, (float)(1/1000));
   }
   cudaMemcpy(dst_d_h, dst_d, N * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
   cudaFree(src1_d);
   cudaFree(src2_d);
   cudaFree(dst_d);
   free(src1_h);
   free(src2_h);
   free(dst_h);
   free(dst_d_h);
   return 0;
}
