#include<iostream>
#include<addTest.h>

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
   cudaFree(x_d);
   cudaFree(dst_d);
   free(x_h);
   free(dst_cpu);
   free(dst_dev_cpu);
   return 0; 
}
