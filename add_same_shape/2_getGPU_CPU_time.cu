#include<iostream>
#include<addTest.h>

// GPU
__global__ void gpuCopy(float*x, float* y, int num) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx < num) {
     y[idx] = x[idx];
   }
}

// CPU
void cpuCopy(float *x, float * y, int num) {
   for (int i = 0; i < num; i++) {
      y[i] = x[i];
   }
}

int main(void) {
   int dev = 0;
   initDevice(dev);
   int num = 1<<20;
   float* x_h = (float *)malloc(num * sizeof(float));
   float* y_h = (float *)malloc(num * sizeof(float));
   float* gpu_h = (float *)malloc(num * sizeof(float));
   initialData(x_h, num);
   
   float *x_d, *y_d;
   CHECK(cudaMalloc((float**)&x_d, num * sizeof(float)));
   CHECK(cudaMalloc((float**)&y_d, num * sizeof(float)));
   CHECK(cudaMemcpy(x_d, x_h, num * sizeof(float), cudaMemcpyHostToDevice));

   int block = 1024;
   int grid = (num + block - 1) / block;
   double start, end;

   start = cpuSecond(); 
   gpuCopy<<<grid, block>>>(x_d, y_d, num);
   cudaDeviceSynchronize();
   end = cpuSecond();
   
   printf("Kernel Time is %f s\n", end - start);
   CHECK(cudaMemcpy(gpu_h, y_d, num * sizeof(float), cudaMemcpyDeviceToHost));
   cudaFree(x_d);
   cudaFree(y_d);
   free(x_h);
   free(y_h);
   free(gpu_h);
   return 0; 
}
