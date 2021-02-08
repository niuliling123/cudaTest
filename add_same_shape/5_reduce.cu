#include<iostream>
#include<addTest.h>
__global__ void  reduceUnroll4(int *src, int *dst, int num) {
  unsigned int tid = threadIdx.x;
  unsigned int id = blockIdx.x * blockDim.x * 4 + threadIdx.x;
  if (tid >= num) return;
  int* data = src + blockIdx.x * blockDim.x * 4;
  if (id + 3 * blockDim.x < num) {
     src[id] += src[id + blockDim.x];
     src[id] += src[id + blockDim.x * 2];
     src[id] += src[id + blockDim.x * 3];
  }
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      data[tid] += data[tid + stride]; 
    }
    __syncthreads();
  }
  if (tid == 0) 
    dst[blockIdx.x] = data[0];
}
__global__ void  reduceUnroll8(int *src, int *dst, int num) {
  unsigned int tid = threadIdx.x;
  unsigned int id = blockIdx.x * blockDim.x * 8 + threadIdx.x;
  if (tid >= num) return;
  int* data = src + blockIdx.x * blockDim.x * 8;
  if (id + 7 * blockDim.x < num) {
     src[id] += src[id + blockDim.x];
     src[id] += src[id + blockDim.x * 2];
     src[id] += src[id + blockDim.x * 3];
     src[id] += src[id + blockDim.x * 4];
     src[id] += src[id + blockDim.x * 5];
     src[id] += src[id + blockDim.x * 6];
     src[id] += src[id + blockDim.x * 7];
  }
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      data[tid] += data[tid + stride]; 
    }
    __syncthreads();
  }
  if (tid == 0) 
    dst[blockIdx.x] = data[0];
}
__global__ void  reduceUnrollWarps8(int *src, int *dst, int num) {
  unsigned int tid = threadIdx.x;
  unsigned int id = blockIdx.x * blockDim.x * 8 + threadIdx.x;
  if (tid >= num) return;
  int* data = src + blockIdx.x * blockDim.x * 8;
  if (id + 7 * blockDim.x < num) {
     src[id] += src[id + blockDim.x];
     src[id] += src[id + blockDim.x * 2];
     src[id] += src[id + blockDim.x * 3];
     src[id] += src[id + blockDim.x * 4];
     src[id] += src[id + blockDim.x * 5];
     src[id] += src[id + blockDim.x * 6];
     src[id] += src[id + blockDim.x * 7];
  }
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
    if (tid < stride) {
      data[tid] += data[tid + stride]; 
    }
    __syncthreads();
  }
  if (tid < 32) {
    volatile int *vmen = data;
    vmen[tid] += vmen[tid + 32];
    vmen[tid] += vmen[tid + 16];
    vmen[tid] += vmen[tid + 8];
    vmen[tid] += vmen[tid + 4];
    vmen[tid] += vmen[tid + 2];
    vmen[tid] += vmen[tid + 1];
  }
  if (tid == 0) 
    dst[blockIdx.x] = data[0];
}
__global__ void  reduceCompleteUnrollWarps8(int *src, int *dst, int num) {
  unsigned int tid = threadIdx.x;
  unsigned int id = blockIdx.x * blockDim.x * 8 + threadIdx.x;
  if (tid >= num) return;
  int* data = src + blockIdx.x * blockDim.x * 8;
  if (id + 7 * blockDim.x < num) {
     src[id] += src[id + blockDim.x];
     src[id] += src[id + blockDim.x * 2];
     src[id] += src[id + blockDim.x * 3];
     src[id] += src[id + blockDim.x * 4];
     src[id] += src[id + blockDim.x * 5];
     src[id] += src[id + blockDim.x * 6];
     src[id] += src[id + blockDim.x * 7];
  }
  __syncthreads();
  if (blockDim.x >= 1024 && tid < 512)
      data[tid] += data[tid + 512];
  __syncthreads();
  if (blockDim.x >= 512 && tid < 256)
      data[tid] += data[tid + 256];
  __syncthreads();
  if (blockDim.x >= 256 && tid < 128)
      data[tid] += data[tid + 128];
  __syncthreads();
  if (blockDim.x >= 128 && tid < 64)
      data[tid] += data[tid + 64];
  __syncthreads();

  if (tid < 32) {
    volatile int *vmen = data;
    vmen[tid] += vmen[tid + 32];
    vmen[tid] += vmen[tid + 16];
    vmen[tid] += vmen[tid + 8];
    vmen[tid] += vmen[tid + 4];
    vmen[tid] += vmen[tid + 2];
    vmen[tid] += vmen[tid + 1];
  }
  if (tid == 0) 
    dst[blockIdx.x] = data[0];
}
__global__ void  reduceUnroll2(int *src, int *dst, int num) {
  unsigned int tid = threadIdx.x;
  unsigned int id = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  if (tid >= num) return;
  int* data = src + blockIdx.x * blockDim.x * 2;
  if (id + blockDim.x < num) {
     src[id] += src[id + blockDim.x];
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      data[tid] += data[tid + stride]; 
    }
    __syncthreads();
  }
  if (tid == 0) 
    dst[blockIdx.x] = data[0];
}


__global__ void  reduceNeighboredLess(int* src, int *dst, int num) {
  unsigned int tid = threadIdx.x;
  unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  int* data = src + blockIdx.x * blockDim.x;
  if (id >= num) return;
  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    int idx = 2 * tid * stride;
    if (idx < blockDim.x) {
      data[idx] += data[idx + stride]; 
    }
    __syncthreads();
  }
  if (tid == 0) 
    dst[blockIdx.x] = data[0];
}

__global__ void  reduceInterieaved(int* src, int *dst, int num) {
   // set threadId 
  unsigned int tid = threadIdx.x;
  unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= num) return;
  int* data = src + blockIdx.x * blockDim.x;
  for (int stride = blockDim.x / 2 ; stride > 0; stride >>= 1) {
    if (tid < stride) {
      data[tid] += data[tid + stride];  
    } 
    __syncthreads();
  }
  if (tid == 0) {
      dst[blockIdx.x] = data[0];
  }
}
__global__ void  reduceNeighbored(int* src, int *dst, int num) {
   // set threadId 
   unsigned int id_thread = threadIdx.x;
   if (id_thread >= num) return;
   int* data = src + blockIdx.x * blockDim.x;
   for (int stride = 1; stride < blockDim.x; stride *= 2) {
     if ((id_thread % (2 * stride)) == 0) {
       data[id_thread] += data[stride + id_thread];
     }
     __syncthreads();
   }
   if (id_thread == 0) {
       dst[blockIdx.x] = data[0];
   }
}

// CPU
int reduceNeighbored_cpu(int *data, int num) {
   if (num == 1) return data[0];
   int const stride = num / 2;
   for (int i = 0; i < stride; i++) {
     data[i] += data[i + stride]; 
   }
   if (num % 2 == 1) {
      data[0] += data[num - 1];
   }
   return reduceNeighbored_cpu(data, stride);
}


int main(void) {
   int dev = 0;
   initDevice(dev);
   int num = 1 << 20;

   int* x_h = (int *)malloc(num * sizeof(int));
   int* dst_cpu = (int *)malloc(num * sizeof(int));
   int* dst_dev_cpu = (int *)malloc(num * sizeof(int));
   for(int i = 0; i < num; i++) {
     x_h[i] = i % 3; 
   }
    
   int *x_d, *dst_d;
   CHECK(cudaMalloc((int**)&x_d, num * sizeof(int)));
   CHECK(cudaMalloc((int**)&dst_d, num * sizeof(int)));
   CHECK(cudaMemcpy(x_d, x_h, num * sizeof(int), cudaMemcpyHostToDevice));
   int block = 1024;
   int grid = (num + block -1) / block;
   printf("grid : %d , block : %d\n", grid, block);

   reduceNeighbored<<<grid, block>>>(x_d, dst_d, num); 
   CHECK(cudaMemcpy(dst_dev_cpu, dst_d, num * sizeof(int), cudaMemcpyDeviceToHost));
   int sum_dev = 0;
   for (int i = 0; i < grid; i++) {
     sum_dev += dst_dev_cpu[i];
   }
   reduceNeighbored_cpu(x_h, num);
   if (sum_dev != x_h[0])
   printf("Error kernel data device: %d  host:%d \n", sum_dev, x_h[0]);

   CHECK(cudaMemcpy(x_d, x_h, num * sizeof(int), cudaMemcpyHostToDevice));
   reduceNeighboredLess<<<grid, block>>>(x_d, dst_d, num); 
   CHECK(cudaMemcpy(dst_dev_cpu, dst_d, num * sizeof(int), cudaMemcpyDeviceToHost));
   sum_dev = 0;
   for (int i = 0; i < grid; i++) {
     sum_dev += dst_dev_cpu[i];
   }
   reduceNeighbored_cpu(x_h, num);
   if (sum_dev != x_h[0])
   printf("Error Less kernel data device: %d  host:%d \n", sum_dev, x_h[0]);
   
   CHECK(cudaMemcpy(x_d, x_h, num * sizeof(int), cudaMemcpyHostToDevice));
   reduceInterieaved<<<grid, block>>>(x_d, dst_d, num); 
   CHECK(cudaMemcpy(dst_dev_cpu, dst_d, num * sizeof(int), cudaMemcpyDeviceToHost));
   sum_dev = 0;
   for (int i = 0; i < grid; i++) {
     sum_dev += dst_dev_cpu[i];
   }
   reduceNeighbored_cpu(x_h, num);
   if (sum_dev != x_h[0])
   printf("Error interieaved kernel data device: %d  host:%d \n", sum_dev, x_h[0]);

   CHECK(cudaMemcpy(x_d, x_h, num * sizeof(int), cudaMemcpyHostToDevice));
   reduceUnroll2<<<grid / 2, block>>>(x_d, dst_d, num); 
   CHECK(cudaMemcpy(dst_dev_cpu, dst_d, num * sizeof(int), cudaMemcpyDeviceToHost));
   sum_dev = 0;
   for (int i = 0; i < grid / 2; i++) {
     sum_dev += dst_dev_cpu[i];
   }
   reduceNeighbored_cpu(x_h, num);
   if (sum_dev != x_h[0])
   printf("Error interieaved kernel data device: %d  host:%d \n", sum_dev, x_h[0]);

   CHECK(cudaMemcpy(x_d, x_h, num * sizeof(int), cudaMemcpyHostToDevice));
   reduceUnroll4<<<grid / 4, block>>>(x_d, dst_d, num); 
   CHECK(cudaMemcpy(dst_dev_cpu, dst_d, num * sizeof(int), cudaMemcpyDeviceToHost));
   sum_dev = 0;
   for (int i = 0; i < grid / 4; i++) {
     sum_dev += dst_dev_cpu[i];
   }
   reduceNeighbored_cpu(x_h, num);
   if (sum_dev != x_h[0])
   printf("Error unroll4 kernel data device: %d  host:%d \n", sum_dev, x_h[0]);
   
   CHECK(cudaMemcpy(x_d, x_h, num * sizeof(int), cudaMemcpyHostToDevice));
   reduceUnroll8<<<grid / 8, block>>>(x_d, dst_d, num); 
   CHECK(cudaMemcpy(dst_dev_cpu, dst_d, num * sizeof(int), cudaMemcpyDeviceToHost));
   sum_dev = 0;
   for (int i = 0; i < grid / 8; i++) {
     sum_dev += dst_dev_cpu[i];
   }
   reduceNeighbored_cpu(x_h, num);
   if (sum_dev != x_h[0])
   printf("Error unroll8 kernel data device: %d  host:%d \n", sum_dev, x_h[0]);
   
   CHECK(cudaMemcpy(x_d, x_h, num * sizeof(int), cudaMemcpyHostToDevice));
   reduceUnrollWarps8<<<grid / 8, block>>>(x_d, dst_d, num); 
   CHECK(cudaMemcpy(dst_dev_cpu, dst_d, num * sizeof(int), cudaMemcpyDeviceToHost));
   sum_dev = 0;
   for (int i = 0; i < grid / 8; i++) {
     sum_dev += dst_dev_cpu[i];
   }
   reduceNeighbored_cpu(x_h, num);
   if (sum_dev != x_h[0])
   printf("Error warps8 kernel data device: %d  host:%d \n", sum_dev, x_h[0]);

   CHECK(cudaMemcpy(x_d, x_h, num * sizeof(int), cudaMemcpyHostToDevice));
   reduceCompleteUnrollWarps8<<<grid / 8, block>>>(x_d, dst_d, num); 
   CHECK(cudaMemcpy(dst_dev_cpu, dst_d, num * sizeof(int), cudaMemcpyDeviceToHost));
   sum_dev = 0;
   for (int i = 0; i < grid / 8; i++) {
     sum_dev += dst_dev_cpu[i];
   }
   reduceNeighbored_cpu(x_h, num);
   if (sum_dev != x_h[0])
   printf("Error Completewarps8 kernel data device: %d  host:%d \n", sum_dev, x_h[0]);
   cudaFree(x_d);
   cudaFree(dst_d);
   free(x_h);
   free(dst_cpu);
   free(dst_dev_cpu);
   return 0; 
}
