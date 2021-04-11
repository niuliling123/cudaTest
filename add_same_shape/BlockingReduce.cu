#include<iostream>
#include<addTest.h>
#define block_size 4
#define block_size_x 8
#define block_size_y 1
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
__global__ void gpuReduce_base(float*x, float* y, int nx, int ny) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   float temp = 0.0f;
   if (idx > nx) return;
   for (int iy = 0; iy < ny; iy++) {
       float tp = x[idx + (iy * nx)];
       temp += tp; 
   }
   y[idx] = temp;
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
 //  int ny=     30522  , nx =   1024; 
   int ny=     1024   , nx =   16  ; 
   int num = nx * ny;
   float* x_h = (float *)malloc(num * sizeof(float));
   float* y_h = (float *)malloc(num * sizeof(float));
   float* gpu_h = (float *)malloc(num * sizeof(float));
   initialData(x_h, num);
   for(int i = 0; i < num ; i++) {
     gpu_h[i] = 0;
     y_h[i] = 0;
     x_h[i] = i % 10;
   } 
   float *x_d, *y_d;
   CHECK(cudaMalloc((float**)&x_d, num * sizeof(float)));
   CHECK(cudaMalloc((float**)&y_d, num * sizeof(float)));

   CHECK(cudaMemcpy(x_d, x_h, num * sizeof(float), cudaMemcpyHostToDevice));
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

   for (int i = 0; i < 1000; i++) {
    
     //gpuCopy<<<grid, block, sizeof(float) * block * block_size>>>(x_d, y_d, nx, ny, block_size_t);
     //cudaDeviceSynchronize();

     //gpuCopy_thread<<<grid_2, block_num, sizeof(float) * block_size_x * block_size_y * block_num>>>(x_d, y_d, nx, ny);
     //cudaDeviceSynchronize();

     gpuReduce_block_size<<<grid_2, block_num, sizeof(float) * block_size_x * block_num>>>(x_d, y_d, nx, ny);
     cudaDeviceSynchronize();

     gpuReduce_vec_4<<<grid_4, block_4>>>(x_d, y_d, nx, ny);
     cudaDeviceSynchronize();

     gpuReduce_base<<<grid_1, block_1>>>(x_d, y_d, nx, ny);
     cudaDeviceSynchronize();
   }
   cudaDeviceSynchronize();
   end = cpuSecond();
   cpuReduce(x_h,y_h, nx, ny);   
   //cpuCopy(x_h,y_h, nx, ny);   
   printf("Kernel Time is %f s\n", end - start);
   CHECK(cudaMemcpy(gpu_h, y_d, num * sizeof(float), cudaMemcpyDeviceToHost));
   checkResult(y_h, gpu_h, num);
   cudaFree(x_d);
   cudaFree(y_d);
   free(x_h);
   free(y_h);
   free(gpu_h);
   return 0; 
}
