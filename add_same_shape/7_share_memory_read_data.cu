#include <iostream>
#include <math.h>
#include <addTest.h>

#define BDIMX 32 
#define BDIMY 32

#define BDIM_X 32
#define BDIM_Y 16 

#define IPAD  1

// row read row
__global__ void setRowReadRow(int *out) {
   __shared__ int tile[BDIMY][BDIMX];

   unsigned int idx = threadIdx.y * blockDim.x * threadIdx.x;

   tile[threadIdx.y][threadIdx.x] = idx;
   __syncthreads();
   out[idx] = tile[threadIdx.y][threadIdx.x];
}

// row read row
__global__ void setColReadCol(int *out) {
   __shared__ int tile[BDIMY][BDIMX];

   unsigned int idx = threadIdx.y * blockDim.x * threadIdx.x;

   tile[threadIdx.x][threadIdx.y] = idx;
   __syncthreads();
   out[idx] = tile[threadIdx.x][threadIdx.y];
}
__global__ void setColReadColIpad(int *out) {
   __shared__ int tile[BDIMY][BDIMX + IPAD];

   unsigned int idx = threadIdx.y * blockDim.x * threadIdx.x;

   tile[threadIdx.x][threadIdx.y] = idx;
   __syncthreads();
   out[idx] = tile[threadIdx.x][threadIdx.y];
}

// GPU
__global__ void setRowReadColRect(int *out) {
   __shared__ int tile[BDIM_Y][BDIM_X];

   unsigned int idx = threadIdx.y * blockDim.x * threadIdx.x;
   unsigned int x = idx % blockDim.x;
   unsigned int y = idx % blockDim.y;
   tile[y][x] = idx;
   __syncthreads();
   out[idx] = tile[x][y];
}
// row read row
__global__ void setRowReadCol(int *out) {
   __shared__ int tile[BDIMY][BDIMX];

   unsigned int idx = threadIdx.y * blockDim.x * threadIdx.x;

   tile[threadIdx.y][threadIdx.x] = idx;
   __syncthreads();
   out[idx] = tile[threadIdx.x][threadIdx.y];
}
// row read row
__global__ void setRowReadColIpad(int *out) {
   __shared__ int tile[BDIMY][BDIMX + IPAD];

   unsigned int idx = threadIdx.y * blockDim.x * threadIdx.x;

   tile[threadIdx.y][threadIdx.x] = idx;
   __syncthreads();
   out[idx] = tile[threadIdx.x][threadIdx.y];
}

// row read row
__global__ void setColReadRow(int *out) {
   __shared__ int tile[BDIMY][BDIMX];

   unsigned int idx = threadIdx.y * blockDim.x * threadIdx.x;

   tile[threadIdx.x][threadIdx.y] = idx;
   __syncthreads();
   out[idx] = tile[threadIdx.y][threadIdx.x];
}
// row read row dynnamic
__global__ void setRowReadRowDyn(int *out) {

   extern __shared__ int tile[];
   unsigned int idx = threadIdx.y * blockDim.x * threadIdx.x;
   tile[idx] = idx;
   __syncthreads();
   out[idx] = tile[idx];
}

// row read row dynnamic
__global__ void setRowReadColDyn(int *out) {

   extern __shared__ int tile[];
   unsigned int idx = threadIdx.y * blockDim.x * threadIdx.x;
   unsigned int idy = threadIdx.x * blockDim.y * threadIdx.y;
   tile[idx] = idx;
   __syncthreads();
   out[idx] = tile[idy];

}
// test the effect of first address alignment on kernel performancet 
void launchKernel(int* src1, int * src2, int *dst, size_t N, int type, dim3 grid, dim3 block) {
  
  switch(type) {
    case 0:
      setRowReadRow<<<grid, block>>>(dst);
      break;
    case 1:
      setColReadRow<<<grid, block>>>(dst);
      break;
    case 2:
      setColReadCol<<<grid, block>>>(dst);
      break;
    case 3:
      setRowReadCol<<<grid, block>>>(dst);
      break;
    case 4:
      setRowReadRowDyn<<<grid, block, (BDIMY) * BDIMX * sizeof(int)>>>(dst);
      break;
    case 5:
      setRowReadColDyn<<<grid, block, (BDIMY) * BDIMX * sizeof(int)>>>(dst);
      break;
    case 6:
      setRowReadColIpad<<<grid, block>>>(dst);
      break;
    case 7:
      setColReadColIpad<<<grid, block>>>(dst);
      break;
    case 8:
      setRowReadColRect<<<grid, block>>>(dst);
      break;
    default:
      std::cout<<" this is default in launchKernel" << std::endl;
      break;
  }
}
void compute(int * dst, size_t num, int loop, kernel_t type, dim3 grid, dim3 block) {
  int * src1, *src2;
  for (int i = 0; i < 1; i++) {

    launchKernel(src1, src2, dst, num, 0, grid, block);
    cudaDeviceSynchronize();

    launchKernel(src1, src2, dst, num, 1, grid, block);
    cudaDeviceSynchronize();

    launchKernel(src1, src2, dst, num, 2, grid, block);
    cudaDeviceSynchronize();

    launchKernel(src1, src2, dst, num, 3, grid, block);
    cudaDeviceSynchronize();

    // launchKernel(src1, src2, dst, num, 4, grid, block);
    // cudaDeviceSynchronize();

    // launchKernel(src1, src2, dst, num, 5, grid, block);
    // cudaDeviceSynchronize();
    launchKernel(src1, src2, dst, num, 6, grid, block);
    cudaDeviceSynchronize();

    launchKernel(src1, src2, dst, num, 7, grid, block);
    cudaDeviceSynchronize();

    launchKernel(src1, src2, dst, num, 8, grid, block);
    cudaDeviceSynchronize();
  }
}

int main(void) {
   // set num of input , offset of src and dst, and loop_num and kernelType
   size_t N = 32 * 32;
   int loop_num = 100;
   kernel_t kernel_type = MEM_ADDR;
   // device
   int dev = 0;
   cudaSetDevice(dev);
   // malloc for host
   int *src1_cpu = (int *)malloc(N * sizeof(int));
   int *src2_cpu = (int *)malloc(N * sizeof(int));
   int *dst_cpu = (int *)malloc(N * sizeof(int));
   int *dst_dev_cpu = (int *)malloc(N * sizeof(int));
   memset(dst_cpu, 0, N * sizeof(int));
   memset(dst_dev_cpu, 0, N * sizeof(int));
   // init host
   // malloc for device
   int *dst_dev;
   cudaMalloc((int**)&dst_dev, N * sizeof(int));
   // memcpy from host to device
   // update N;
   size_t count = N;
   dim3 block(BDIMY,BDIMX);
   dim3 grid(1,1);
   // gpu compute
   compute(dst_dev, count, loop_num,  kernel_type, grid, block);
   cudaDeviceSynchronize();
   cudaMemcpy(dst_dev_cpu, dst_dev, N * sizeof(int), cudaMemcpyDeviceToHost);
   // cpu compute
   // free device
   cudaFree(dst_dev);

   // free cpu
   free(src1_cpu);
   free(src2_cpu);
   free(dst_cpu);
   free(dst_dev_cpu);
   return 0;
}
