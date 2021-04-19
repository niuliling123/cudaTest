#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <limits>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include<addTest.h>
#define block_size 4
#define block_size_x 8
#define block_size_y 1
#define BLOCK_SIZE  512
// GPU
void check_error(void)
{
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    exit(err);
  }
}

// GPU
template<typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * 4) aligned_vector {
      scalar_t val[4];
};
__global__ void gpuCopy(float*x, float* y, int nx, int ny, int block_size_t) {
   extern  __shared__ float staticShared[];
    for(int i = threadIdx.x; i < ny; i += blockDim.x) {
      #pragma unroll
      for(int m =0; m < block_size; m++) {
        int idx = threadIdx.x * block_size + m;
        staticShared[idx] = x[i * nx + m + blockIdx.x * block_size];
      }
      __syncthreads();
      #pragma unroll
      for(int j = 0; j < block_size; j ++) {
        int idx = threadIdx.x * block_size + j;
        int out_id = (blockIdx.x * block_size  + j) * ny + i; 
        y[out_id] = staticShared[idx];
      }
    }
}
__global__ void gpuCopy_thread(float*x, float* y, int nx, int ny) {
   extern  __shared__ float staticShared[];
   int num = ny / block_size_y;

   for(int in = 0; in < num ; in++) {
      #pragma unroll
      for(int m = 0; m < block_size_y; m++) {
        #pragma unroll
        for(int ix = 0; ix < block_size_x; ix++) {
          int idx_x = nx * (in * block_size_y + m);
          int idx_x_small = ix + (blockIdx.x * blockDim.x + threadIdx.x) * block_size_x;
          int idx_src = idx_x + idx_x_small;
          int idx_shared = m * blockDim.x * block_size_x + threadIdx.x * block_size_x + ix;
          staticShared[idx_shared] = x[idx_src]; 
        }
      } 

      #pragma unroll
      for(int ix = 0; ix < block_size_x; ix++) {
        #pragma unroll
        for(int m = 0; m < block_size_y; m ++) {
           int idx_y = in * block_size_y + m;
           int idx_higher = ((blockIdx.x * blockDim.x + threadIdx.x) * block_size_x + ix) * ny;
           int idx_dst = idx_y + idx_higher;
           int idx_shared =  m * blockDim.x * block_size_x + threadIdx.x * block_size_x + ix;
           y[idx_dst] = staticShared[idx_shared];
        }
      
      }
   }
}

__global__ void gpuReduce_block_size(float*x, float* y, int nx, int ny) {
   extern  __shared__ float staticShared[];
   for (int iy = 0; iy < ny; iy++) {
     #pragma unroll
     for(int ix = 0; ix < block_size_x; ix++) {
       int idx_src = iy * nx + blockIdx.x * blockDim.x * block_size_x + threadIdx.x * block_size_x + ix;
       staticShared[ix] += x[idx_src]; 
     }
   }
    #pragma unroll
   for (int ix = 0; ix < block_size_x; ix++) {
     int idx_dst = blockIdx.x * blockDim.x * block_size_x + threadIdx.x * block_size_x + ix;
     y[idx_dst] = staticShared[ix];
   }
}
// register
__global__ void gpuReduce_vec_4(float*x, float* y, int nx, int ny) {
   using Vec = aligned_vector<float, 4>;
   Vec *src = reinterpret_cast<Vec *>(x); 
   Vec *dst = reinterpret_cast<Vec *>(y); 
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
     Vec temp;
     Vec tp;
   if (idx + 3 > nx) return;
   for(int t = idx; t < nx/4; t += gridDim.x * blockDim.x) {
     temp.val[0] = 0.0f;
     temp.val[1] = 0.0f;
     temp.val[2] = 0.0f;
     temp.val[3] = 0.0f;
     for (int iy = 0; iy < ny; iy++) {
         tp = src[t + (iy * nx)/4];
         #pragma unroll
         for (int ix = 0; ix < 4; ix++) {
            temp.val[ix] += tp.val[ix]; 
         }
     }
     dst[t] = temp;
   }
}
__global__ void gpuReduce_share(float*x, float* y, int nx, int ny) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   int idy = blockIdx.y * BLOCK_SIZE;
   extern  __shared__ float staticShared[];
   float temp = 0.0f;
   if (idx > nx) return;
   for (int iy = 0; iy < BLOCK_SIZE && idy + iy < ny; iy ++) {
      int id = (idy + iy) * nx + idx;
      staticShared[threadIdx.x] += x[id]; 
     // float tp = x[id];
     // temp += tp;
   }
   y[idx + blockIdx.y * nx] = staticShared[threadIdx.x];
}

__global__ void gpuReduce_base(float*x, float* y, int nx, int ny) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   float temp = 0.0f;
   if (idx > nx) return;
   for (int iy = 0; iy < ny; iy ++) {
      int id = iy * nx + idx;
      float tp = x[id];
      temp += tp;
   }
   y[idx + blockIdx.y * nx] = temp;
}
// reduce_y + reduce_global 
__global__ void gpuReduce_y(float*x, float* y, int nx, int ny) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   int idy = blockIdx.y * BLOCK_SIZE;
   extern  __shared__ float staticShared[];
   float temp = 0.0f;
   if (idx > nx) return;
   for (int iy = 0; iy < BLOCK_SIZE && idy + iy < ny; iy ++) {
      int id = (idy + iy) * nx + idx;
      float tp = x[id];
      temp += tp;
   }
   y[idx + blockIdx.y * nx] = temp;
}
__global__ void gpuReduce_global(float*x, float* y, int nx, int ny) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   int idy = blockIdx.y * BLOCK_SIZE;
   float temp = 0.0f;
   if (idx > nx) return;
   for (int iy = 0; iy < BLOCK_SIZE && idy + iy < ny; iy ++) {
      int id = (idy + iy) * nx + idx;
      float tp = x[id];
      temp += tp;
   }
   y[idx + blockIdx.y * nx] = temp;
}

// CPU
void cpuCopy(float *x, float * y, int nx, int ny) {
   for(int i = 0; i < nx; i ++) {
     for (int j = 0; j < ny; j ++) {
       int idx_dst = i * ny + j;
       int idx_src = j * nx + i;
       y[idx_dst] = x[idx_src];
     
     }
   }
}

void cpuReduce(float *x, float * y, int nx, int ny) {
   for(int i = 0; i < nx; i ++) {
     float sum = 0;
     for (int j = 0; j < ny; j ++) {
       int idx_src = j * nx + i;
       sum += x[idx_src];
     }
     y[i] = sum; 
   }
}
int main(int block_size_t, char *argv[]) {
   int dev = 0;
   initDevice(dev);
  // int nx = 75264;
  // int ny = 2048;
  // int ny=     512    , nx =   2048; 
  // int ny=     128    , nx =   1024; 
   int ny=     30522  , nx =   1024; 
 //  int ny=     1024   , nx =   16  ; 
   int num = nx * ny;
   float* x_h = (float *)malloc(num * sizeof(float));
   float* y_h = (float *)malloc(num * sizeof(float));
   float* gpu_h = (float *)malloc(num * sizeof(float));
   initialData(x_h, num);
   for(int i = 0; i < num ; i++) {
     gpu_h[i] = 0;
     y_h[i] = 0;
     x_h[i] = 1 % 10;
   } 
   float *x_d, *y_d;
   CHECK(cudaMalloc((float**)&x_d, num * sizeof(float)));
   CHECK(cudaMalloc((float**)&y_d, num * sizeof(float)));

   CHECK(cudaMemcpy(x_d, x_h, num * sizeof(float), cudaMemcpyHostToDevice));
   CHECK(cudaMemcpy(y_d, y_h, num * sizeof(float), cudaMemcpyHostToDevice));
   int block = 32;
   int grid = (nx + block_size -1) / block_size;
   double start, end;
   start = cpuSecond(); 

   int block_num =  64;
   int grid_2 = (nx / block_size_x + block_num - 1)/ block_num;

   int block_4 =  32;
   int grid_4 = (nx / 4 + block_4 - 1)/ block_4;
  
   int block_1 =  32;
   int grid_1 = (nx + block_1 - 1)/ block_1;
   int grid_y = (ny + BLOCK_SIZE - 1)/BLOCK_SIZE;
   dim3 grid3(grid_1, grid_y);
   dim3 block3(block_1, 1);
   std::chrono::high_resolution_clock::time_point t1, t2;
   std::vector<std::vector<double>> timings(5);
   gpuReduce_vec_4<<<grid_4, block_4>>>(x_d, y_d, nx, ny);
   cudaDeviceSynchronize();
   for (int i = 0; i < 1000; i++) {
    
     gpuReduce_block_size<<<grid_2, block_num, sizeof(float) * block_size_x * block_num>>>(x_d, y_d, nx, ny);
     cudaDeviceSynchronize();

     gpuReduce_vec_4<<<grid_4, block_4>>>(x_d, y_d, nx, ny);
     cudaDeviceSynchronize();

     gpuReduce_base<<<grid3, block3>>>(x_d, y_d, nx, ny);
     cudaDeviceSynchronize();

     t1 = std::chrono::high_resolution_clock::now();
     gpuReduce_y<<<grid3, block3, sizeof(float) * block_1>>>(x_d, y_d, nx, ny);
     check_error();
     cudaDeviceSynchronize();
     check_error();
     t2 = std::chrono::high_resolution_clock::now();
     timings[0].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());


     gpuReduce_global<<<grid_1, block_1>>>(y_d, y_d, nx, grid_y);
     cudaDeviceSynchronize();
   }
   cudaDeviceSynchronize();
   end = cpuSecond();
   cpuReduce(x_h,y_h, nx, ny);   
   double average = std::accumulate(timings[0].begin()+1, timings[0].end(), 0.0) / (double)(1000);
   //cpuCopy(x_h,y_h, nx, ny);   
   printf("Kernel Time is %f s %f ms\n", end - start, average * 1000);
   CHECK(cudaMemcpy(gpu_h, y_d, num * sizeof(float), cudaMemcpyDeviceToHost));
  // for(int i = 0; i < nx * grid_y; i++) {
  //    if(gpu_h[i] != 512) printf("error %d  %f %d\n", i / nx, gpu_h[26624], i);
  // }
   checkResult(y_h, gpu_h, nx);
   cudaFree(x_d);
   cudaFree(y_d);
   free(x_h);
   free(y_h);
   free(gpu_h);
   return 0; 
}
