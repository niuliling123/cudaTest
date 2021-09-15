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
#define BLOCK_SIZE   64
#define hl 1

#define VecSize 4
// GPU
unsigned hight_bit(unsigned x){//0010 1100 0000 0000 0000 0000 0000 0000 0000 0001
    x= x|(x>>1);              //0011 1110 0000 0000 0000 0000 0000 0000 0000 0000
    x= x|(x>>2);              //0011 1111 1000 0000 0000 0000 0000 0000 0000 0000
    x= x|(x>>4);              //0011 1111 1111 1000 0000 0000 0000 0000 0000 0000
    x= x|(x>>8);              //0011 1111 1111 1111 1111 1000 0000 0000 0000 0000
    x= x|(x>>16);             //0011 1111 1111 1111 1111 1111 1111 1111 1111 1111
    x= x|(x>>32);
// return max(1, x - (x >> 1));
// 如果数特别大， 这里感觉会溢出， 所以这里只使用于小于数据最大值1/2的数。
   return (x+1) >> 1;        //0100 0000 0000 0000 0000 0000 0000 0000 0000 0000
}
// returns floor(log2(n))
static inline int last_pow2(int n) {
   n |= (n >>  1);
   n |= (n >>  2);
   n |= (n >>  4);
   n |= (n >>  8);
   n |= (n >> 16);
   return std::max(1, n - (n >> 1));
}

static inline int GetLastPow2(int n) {
  n = log2(n);
  n = max(0, n);
  return std::pow(2, n);
}

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
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
      scalar_t val[vec_size];
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
   if (idx > nx) return;
   int t = idx;
   // for(int t = idx; t < nx; t += gridDim.x * blockDim.x) {
     temp.val[0] = 0.0f;
     temp.val[1] = 0.0f;
     temp.val[2] = 0.0f;
     temp.val[3] = 0.0f;
     for (int iy = 0; iy < ny; iy++) {
         tp = src[t + (iy * nx)];
         #pragma unroll
         for (int ix = 0; ix < 4; ix++) {
            temp.val[ix] += tp.val[ix]; 
         }
     }
     dst[t] = temp;
   //}
}

template <typename Tx, typename Ty>
__global__ void reduce_900 (const Tx *x, Ty * y, int nx, int ny) {
  int block_size_t = ny + 1;
  
  using VecTx = aligned_vector<Tx, VecSize>;

  const VecTx *src = reinterpret_cast<const VecTx *> (x);
  VecTx *dst = reinterpret_cast<VecTx *> (y);
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  VecTx tp;
  VecTx reduce_var;

  reduce_var.val[0] = 0.0f;
  reduce_var.val[1] = 0.0f;
  // reduce_var.val[2] = 0.0f;
  // reduce_var.val[3] = 0.0f;

  if (idx < nx) {
    for (int iy = 0; iy < ny; iy++) {
      int id = (iy) * nx + idx;
      tp = src[id];
      #pragma unroll
      for(int i = 0; i < VecSize; i++) {
        reduce_var.val[i] += static_cast<Ty>(tp.val[i]);
      }

    }
      dst[idx] = reduce_var; 
  }

}

__global__ void gpuReduce_base(float*x, float* y, int nx, int ny, int nz) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   float temp = 0.0f;
   if (idx < nx) {
     for (int iz = 0; iz < nz; iz++) {
       for (int iy = 0; iy < ny; iy ++) {
          int id = iy * nx + idx + blockIdx.y * ny * nx + iz * nx * ny;
          float tp = x[id];
          temp += tp;
       }

       y[idx + blockIdx.y * nx + iz * gridDim.y * nx] = temp;
     }
   }
}
//
__global__ void gpuReduce_vec_4_base(float*x, float* y, int nx, int ny) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx > nx) return;
   using Vec = aligned_vector<float, 4>;
   Vec *src = reinterpret_cast<Vec *>(x); 
   Vec *dst = reinterpret_cast<Vec *>(y); 
   Vec temp;
   Vec tp;
   temp.val[0] = 0.0f;
   temp.val[1] = 0.0f;
   temp.val[2] = 0.0f;
   temp.val[3] = 0.0f;
   for (int iy = 0; iy < ny; iy ++) {
      int id = iy * nx + idx;
      tp = src[id];
      #pragma unroll
      for (int ix = 0; ix < 4; ix++) {
         temp.val[ix] += tp.val[ix]; 
      }
   }
   dst[idx] = temp;
}

__global__ void gpuReduce_vec_4_y(float*x, float* y, int nx, int block_size_m) {
   using Vec = aligned_vector<float, 4>;
   Vec *src = reinterpret_cast<Vec *>(x); 
   Vec *dst = reinterpret_cast<Vec *>(y); 
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   int idy = blockIdx.y * block_size_m;

   Vec temp;
   Vec tp;

   if (idx > nx) return;
   temp.val[0] = 0.0f;
   temp.val[1] = 0.0f;
   temp.val[2] = 0.0f;
   temp.val[3] = 0.0f;

   for (int iy = 0; iy < block_size_m; iy ++) {
       tp = src[(idy + iy ) *nx + idx];
       #pragma unroll
       for (int ix = 0; ix < 4; ix++) {
          temp.val[ix] += tp.val[ix]; 
       }
   }
   dst[idx + blockIdx.y * nx] = temp;

}

__global__ void gpuReduce_global_vec_4(float*x, float* y, int nx, int ny) {
   using Vec = aligned_vector<float, 4>;
   Vec *src = reinterpret_cast<Vec *>(x); 
   Vec *dst = reinterpret_cast<Vec *>(y); 
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   Vec temp;
   Vec tp;
   if (idx > nx) return;
   temp.val[0] = 0.0f;
   temp.val[1] = 0.0f;
   temp.val[2] = 0.0f;
   temp.val[3] = 0.0f;
   for (int iy = 0; iy < ny; iy ++) {
       tp = src[(iy ) *nx + idx];
       #pragma unroll
       for (int ix = 0; ix < 4; ix++) {
          temp.val[ix] += tp.val[ix]; 
       }
   }
   dst[idx + blockIdx.y * nx] = temp;
}
// reduce_y + reduce_global 
__global__ void gpuReduce_y(float*x, float* y, int nx, int ny, int block_size_reduce_ny) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   int idy = blockIdx.y * block_size_reduce_ny;
   extern  __shared__ float staticShared[];
   float temp = 0.0f;
   int loop = min(ny -idy, block_size_reduce_ny);
   if (idx > nx) return;
   for (int iy = 0; iy < loop; iy ++) {
      int id = (idy + iy) * nx + idx + blockIdx.z * nx * ny;
      float tp = x[id];
      temp += tp;
   }
   y[idx + blockIdx.y * nx + blockIdx.z * gridDim.y * nx] = temp;
}
__global__ void gpuReduce_global(float*x, float* y, int nx, int ny) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   float temp = 0.0f;
   if (idx > nx) return;
   for (int iy = 0; iy < ny; iy ++) {
      int id = iy * nx + idx + blockIdx.y * nx * ny;
      float tp = x[id];
      temp += tp;
   }
   y[idx + blockIdx.y * nx] = temp;
}

// CPU
void cpuCopy(float *x, float * y, int nx, int ny) { 
   for(int i = 0; i < nx; i++) {
     for (int j = 0; j < ny; j ++) {
       int idx_dst = i * ny + j;
       int idx_src = j * nx + i;
       y[idx_dst] = x[idx_src];
     
     }
   }
}

void cpuReduce(float *x, float * y, int nx, int ny, int nz) {
    for(int t =0; t < nz; t++) {
   for(int i = 0; i < nx; i ++) {
     float sum = 0;
     for (int j = 0; j < ny; j ++) {
       int idx_src = j * nx + i;
       sum += x[idx_src];
     }
     y[i + t * nx] = sum; 
   } }
}

template <typename T>
struct IdentityFunctor {
   explicit inline IdentityFunctor() {}

  __device__ inline T operator()(const T& x) const { return x; }
};

template <typename T>
struct CustomSum {
  __device__ __forceinline__ T operator()(const T &a, const T &b) const {
    return b + a;
  }
};

template <typename Tx, typename Ty, typename ReduceOp, typename TransformOp>
__global__ void ReduceFirstDim_t(const Tx* x, Ty* y, ReduceOp reducer,
                               TransformOp transformer, int ny,
                               int nx) {
  //int block_size_p = reduce_num;
  //int idx = blockIdx.x * blockDim.x + threadIdx.x;
  //int idy = blockIdx.y * block_size_p;
  //Ty init = (Ty)(0.0f);
  //Ty reduce_var = init;

  //if (idx < left_num) {
  //  for (int iy = 0; iy < block_size_p && idy + iy < reduce_num; iy++) {
  //    int id = (idy + iy) * left_num + idx;
  //    reduce_var = reducer(reduce_var, static_cast<Ty>(x[id]));
  //  }
  //  y[idx + blockIdx.y * left_num] = static_cast<Ty>(transformer(reduce_var));
  //}
  int block_size_p = ny;
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   int idy = blockIdx.y * block_size_p;
   float reduce_var = 0.0f;
   if (idx < nx) {
     int base = ny - idy;
     base = base > block_size_p ? block_size_p : base;
     for (int iy = 0; iy < base; iy++) {
        int id = (idy + iy) * nx + idx;
        reduce_var = reducer(reduce_var, static_cast<Ty>(x[id]));
     }
     y[idx] = reduce_var;
   }
}

int main(int argc, char *argv[]) {
   int block_size_t = 0;

   if (argc > 1) {
       block_size_t = atoi(argv[1]);
   }
   int dev = 0;
   initDevice(dev);
   cudaSetDevice(dev);
   cudaDeviceProp deviceProp;
   cudaGetDeviceProperties(&deviceProp,dev);
   int ny = 2048, nx = 75264;
   int nz = 16;
   switch(block_size_t) {
     case 0:
       ny = 8, nx = 128;
       break;
     case 1:
       ny = 512  , nx =   2048; 
       break;
     case 2:
       ny = 128  , nx =   1024; 
       break;
     case 3:
       ny = 30522, nx =   1024; 
       break;
     case 4:
       ny = 1024 , nx =   16  ; 
       break;
     case 5:
       ny = 256  ; nx = 12800;
       break;
     case 6:
       ny = 256  ; nx = 10240;
       break;
     case 7:
       ny = 1024 ; nx = 1280 ;
       break;
     case 8:
       ny = 32768; nx = 1280 ;
       break;
     case 9:
       ny = 30522; nx = 10240;
       break;
     case 10:
       ny = 256  ; nx = 10240;
       break;
     case 11:
       ny = 1024 ; nx = 1280 ;
       break;
     case 12:
       ny = 32768; nx = 1280 ;
       break;
     case 13:
       ny = 30522; nx = 10240;
       break;
     case 14:
       ny = 2560 ; nx = 10240;
       break;
     case 15:
       ny = 10240; nx = 1280 ;
       break;
     case 16:
       ny = 32768; nx = 2560 ;
       break;
     case 17:
       ny = 30522; nx = 1024 ;
       break;
    default:
         printf("this is default\n");
       break;
   }
   printf("%d this is block_size_t %d %d\n", block_size_t, ny, nx);
   // int ny = 2048, nx = 75264;
   // int ny = 512  , nx =   2048; 
   // int ny = 128  , nx =   1024; 
   // int ny = 30522, nx =   1024; 
   // int ny = 1024 , nx =   16  ; 
   // int ny = 256  ; nx = 12800;
   // int ny = 256  ; nx = 10240;
   // int ny = 1024 ; nx = 1280 ;
   // int ny = 32768; nx = 1280 ;
   // int ny = 30522; nx = 10240;
   // int ny = 256  ; nx = 10240;
   // int ny = 1024 ; nx = 1280 ;
   // int ny = 32768; nx = 1280 ;
   // int ny = 30522; nx = 10240;
   // int ny = 2560 ; nx = 10240;
   // int ny = 10240; nx = 1280 ;
   // int ny = 32768; nx = 2560 ;
   // int ny = 30522; nx = 1024 ;
// ny = ny * 2; 
   int num = nx * ny * nz * 16;
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

   int block_num =  64;
   int grid_2 = (nx / block_size_x + block_num - 1)/ block_num;

   int block_4 =  32;
   int grid_4 = (nx / 4 + block_4 - 1)/ block_4;
  
   int block_5 =  32;
   int grid_5 = (nx / VecSize + block_4 - 1)/ block_4;

   int block_1 = 32 ;
   int grid_1 = (nx/4 + block_1 - 1)/ block_1;
   int grid_0 = (nx + block_1 - 1)/ block_1;
   int grid_3 = (nx + block_1 - 1)/ block_1;
   printf("%d %d\n", grid_0, block_1);
   std::chrono::high_resolution_clock::time_point t1, t2;
   std::vector<std::vector<double>> timings(5);

   int max_threads = deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount;
   int num_block = (max_threads / (grid_0 * 32));
   int block_size_reduce_ny = GetLastPow2(ny * nz / num_block);

   if (block_size_reduce_ny <= 1)  {
     block_size_reduce_ny = GetLastPow2(sqrt(ny));
   } else if (block_size_reduce_ny * 2 < ny) {
     block_size_reduce_ny *= 2;
   }

   block_size_reduce_ny = max(block_size_reduce_ny, 32);
   printf("A   this is block_size_reduce_ny %d %d\n", block_size_reduce_ny, num_block);
   // block_size_reduce_ny = 32;
   // block_size_reduce_ny = atoi(argv[2]);
   
   printf("B   this is block_size_reduce_ny %d\n", block_size_reduce_ny);

// int grid_y = (ny + BLOCK_SIZE - 1)/BLOCK_SIZE;
   int grid_y = (ny + block_size_reduce_ny - 1) / block_size_reduce_ny;

   dim3 grid3(grid_3, grid_y, nz);
   dim3 block3(block_1, 1);


   dim3 grid5(grid_4, grid_y);
   dim3 block5(block_1, 1, 1);

   // gpuReduce_vec_4<<<grid_4, block_4>>>(x_d, y_d, nx, ny);
   // cudaDeviceSynchronize();
   for (int i = 0; i < 1000; i++) {

    
     // gpuReduce_block_size<<<grid_2, block_num, sizeof(float) * block_size_x * block_num>>>(x_d, y_d, nx, ny);
     // cudaDeviceSynchronize();

     // gpuReduce_vec_4<<<grid_4, block_4>>>(x_d, y_d, nx/4, ny);
     // cudaDeviceSynchronize();

     // reduce_900<float, float><<<grid_5, block_5>>>(x_d, y_d, nx/VecSize, ny);
     // cudaDeviceSynchronize();

     // ReduceFirstDim_t<float, float, CustomSum<float>, IdentityFunctor<float>><<<grid_0, block_1>>>(x_d, y_d,
     //         CustomSum<float>(), IdentityFunctor<float>(), ny, nx);
     // cudaDeviceSynchronize();
     if (grid_y == 1)  {  
       dim3 pp(grid_0, nz);
       dim3 tt(block_1, 1);
       // gpuReduce_base<<<grid_0, block_1>>>(x_d, y_d, nx, ny);
       gpuReduce_base<<<pp, tt>>>(x_d, y_d, nx, ny, 16);
     }
     cudaDeviceSynchronize();
     // gpuReduce_vec_4_base<<<grid_1, block_1>>>(x_d, y_d, nx/4, ny);
     // cudaDeviceSynchronize();

     t1 = std::chrono::high_resolution_clock::now();

      if(grid_y > 1) {
        gpuReduce_y<<<grid3, block3, sizeof(float) * block_1>>>(x_d, y_d, nx, ny, block_size_reduce_ny);
        // gpuReduce_vec_4_y<<<grid5, block5>>>(x_d, y_d, nx/4, block_size_reduce_ny);
      }

     check_error();
     cudaDeviceSynchronize();
     check_error();
     t2 = std::chrono::high_resolution_clock::now();
     timings[0].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

     if(grid_y > 1) {
       dim3 temp(grid3.x, nz);
       dim3 mp(32, 1);
       gpuReduce_global<<<temp, mp>>>(y_d, y_d, nx, grid_y);
      // gpuReduce_global_vec_4<<<grid5.x, block5>>>(x_d, y_d, nx/4, grid5.y);
       // printf("\n** grid_3_x:  %d grid_3_y: %d block_3_x: %d block_3_y: %d\n", temp.x, temp.y, mp.x, mp.y);
     }
   }

   if (grid_y) {
     printf("grid_3_x %d grid_3_y %d block_3_x %d block_3_y %d", grid3.x, grid3.y, block3.x, block3.y);
   }
   cudaDeviceSynchronize();
   cpuReduce(x_h,y_h, nx, ny, nz);   
   double average = std::accumulate(timings[0].begin()+1, timings[0].end(), 0.0) / (double)(1000);
   //cpuCopy(x_h,y_h, nx, ny);   
   double start, end;
   unsigned int m = 4099;
   unsigned int t = 0;
   start = cpuSecond() * 1000; 
   for(int i = 0; i < 1000; i++) {
    int tp = last_pow2(4099);      
   }
   end = cpuSecond() * 1000;
   printf("\n * %u this is m \n", hight_bit(m));
   printf("--------> %f ms\n", end - start);

   start = cpuSecond() * 1000; 
   for(int i = 0; i < 1000; i++) {
    int tp = GetLastPow2(4099);      
   }
   end = cpuSecond() * 1000;
   printf("->>>>>>>> %f ms\n", end - start);

   start = cpuSecond() * 1000; 
   for(int i = 0; i < 1000; i++) {
    t = hight_bit(4099);      
   }
   end = cpuSecond() * 1000;
   printf("-<<<<<<<< %f ms\n", end - start);

   CHECK(cudaMemcpy(gpu_h, y_d, num * sizeof(float), cudaMemcpyDeviceToHost));
  // for(int i = 0; i < nx * grid_y; i++) {
  //    if(gpu_h[i] != 512) printf("error %d  %f %d\n", i / nx, gpu_h[26624], i);
  // }
   checkResult(y_h, gpu_h, nx * nz);
   cudaFree(x_d);
   cudaFree(y_d);
   free(x_h);
   free(y_h);
   free(gpu_h);
   return 0; 
}
