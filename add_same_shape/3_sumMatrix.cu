#include<iostream>
#include<addTest.h>

__global__ void sumMatrix(float* x, float *y, float * dst, int nx, int ny) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   int idy = blockIdx.y * blockDim.y + threadIdx.y;

   int id = idx + idy * nx;
   if (id < nx * ny && idx < nx && idy < ny) {
     dst[id] = x[id] + y[id];
   }
}

// GPU
__global__ void gpuCopy(float*x, float* y, int num) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx < num) {
     y[idx] = x[idx];
   }
}

// CPU
void cpuSumMatrix(float *x, float * y, float * dst, int nx, int ny) {
     for (int i = 0; i < ny * nx; i++) {
       dst[i] = x[i] + y[i];
     }
}

void launchKernel_1block1grid(float *x, float *y, float *dst, int nx, int ny) {
   // 1d block 1d grid
   int block = 32;
   int grid = (nx * ny + block - 1) / block;
   printf("11 grid x %d y %d, block x %d y %d", grid, grid, block, block);
   sumMatrix<<<grid, block>>>(x, y, dst, nx * ny, 1);
}

void launchKernel_1block2grid(float *x, float *y, float *dst, int nx, int ny) {
   // 1d block 1d iid
   int num = 32;
   dim3 block(num);
   dim3 grid((nx + block.x -1) / block.x, ny);
   printf("12 grid x %d y %d, block x %d y %d", grid.x, grid.y, block.x, block.y);
   sumMatrix<<<grid, block>>>(x, y, dst, nx , ny);
}
void launchKernel_2block2grid(float *x, float *y, float *dst, int nx, int ny) {
   // 1d block 1d grid
   int num_x = 32;
   int num_y = 16;
   dim3 block(num_x, num_y);
   dim3 grid((nx + block.x -1) / block.x, (ny + block.y -1) / block.y);
   printf("22 grid x %d y %d, block x %d y %d", grid.x, grid.y, block.x, block.y);
   sumMatrix<<<grid, block>>>(x, y, dst, nx , ny);
}

int main(void) {
   int dev = 0;
   initDevice(dev);

   int nx = 1 << 10;
   int ny = 1 << 10;
   int num = nx * ny;

   float* x_h = (float *)malloc(num * sizeof(float));
   float* y_h = (float *)malloc(num * sizeof(float));
   float* dst_cpu = (float *)malloc(num * sizeof(float));
   float* dst_dev_cpu = (float *)malloc(num * sizeof(float));
   initialData(x_h, num);
   initialData(y_h, num);
   
   float *x_d, *y_d, *dst_d;
   CHECK(cudaMalloc((float**)&x_d, num * sizeof(float)));
   CHECK(cudaMalloc((float**)&y_d, num * sizeof(float)));
   CHECK(cudaMalloc((float**)&dst_d, num * sizeof(float)));
   CHECK(cudaMemcpy(x_d, x_h, num * sizeof(float), cudaMemcpyHostToDevice));
   CHECK(cudaMemcpy(y_d, y_h, num * sizeof(float), cudaMemcpyHostToDevice));
   
   // 1d block 1d grid
   // launchKernel_1block1grid(x_d, y_d, dst_d, nx, ny);
   // 2d block 1d grid
   // launchKernel_1block2grid(x_d, y_d, dst_d, nx, ny);
   // 2d block 2d grid
   launchKernel_2block2grid(x_d, y_d, dst_d, nx, ny);
   CHECK(cudaMemcpy(dst_dev_cpu, dst_d, num * sizeof(float), cudaMemcpyDeviceToHost));

   double start, end;
   start = cpuSecond(); 
   cpuSumMatrix(x_h, y_h, dst_cpu, nx, ny);   
   // cudaDeviceSynchronize();
   end = cpuSecond();
   printf("Kernel Time is %f s\n", end - start);
   checkResult(dst_dev_cpu, dst_cpu, nx * ny);
   cudaFree(x_d);
   cudaFree(y_d);
   cudaFree(dst_d);
   free(x_h);
   free(y_h);
   free(dst_cpu);
   free(dst_dev_cpu);
   return 0; 
}
