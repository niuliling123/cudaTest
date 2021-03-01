#include <iostream>
#include <math.h>
#include <addTest.h>
__global__ void transformNaiveRowDiagonal(float * MatA,float * MatB,int nx,int ny)
{
        int block_y=blockIdx.x;
        int block_x=(blockIdx.x+blockIdx.y)%gridDim.x;
        int ix=threadIdx.x+blockDim.x*block_x;
        int iy=threadIdx.y+blockDim.y*block_y;
        int idx_row=ix+iy*nx;
        int idx_col=ix*ny+iy;
        if (ix<nx && iy<ny) {
           MatB[idx_col]=MatA[idx_row];
        }
}
__global__ void transformNaiveColDiagonal(float * MatA,float * MatB,int nx,int ny)
{
        int block_y=blockIdx.x;
        int block_x=(blockIdx.x+blockIdx.y)%gridDim.x;
        int ix=threadIdx.x+blockDim.x*block_x;
        int iy=threadIdx.y+blockDim.y*block_y;
        int idx_row=ix+iy*nx;
        int idx_col=ix*ny+iy;
        if (ix<nx && iy<ny) {
          MatB[idx_row]=MatA[idx_col];
        }
}


void transformMatrix2D_CPU(float * matA, float * matB, int nx, int ny) {
  for(int i = 0; i < ny; i++) {
    for(int j = 0; j < nx; j++) {
      matB[i + j * ny] = matA[j + i * nx];
    }
  
  }
}

__global__ void transRow(float* matA, float *matB, int nx, int ny) {
   int ix = threadIdx.x + blockIdx.x * blockDim.x;
   int iy = threadIdx.y + blockIdx.y * blockDim.y;
   int idx = iy * nx + ix;
   int idy = ix * ny + iy;
   if (ix < nx && iy < ny)
       matB[idy] = matA[idx];
}
__global__ void transCol(float * matA, float *matB, int nx, int ny) {
   int ix = threadIdx.x + blockIdx.x * blockDim.x;
   int iy = threadIdx.y + blockIdx.y * blockDim.y;
   int idx = iy * nx + ix;
   int idy = ix * ny + iy;
   if (ix < nx && iy < ny)
       matB[idx] = matA[idy];
}
int main(void) {
   // set num of input , offset of src and dst, and loop_num and kernelType
   int nx = 1 << 10;
   int ny = 1 << 10;
   size_t N = 1  << 20;
   int dev = 0;
   cudaSetDevice(dev);
   // malloc for host
   float *src_cpu = (float *)malloc(N * sizeof(float));
   float *dst_cpu = (float *)malloc(N * sizeof(float));
   float *dst_dev_cpu = (float *)malloc(N * sizeof(float));
   memset(dst_cpu, 0, N * sizeof(float));
   memset(dst_dev_cpu, 0, N * sizeof(float));
   initialData(src_cpu, N);
   float *src_dev = NULL; 
   float *dst_dev;
   cudaMalloc((float**)&src_dev, N * sizeof(float));
   cudaMalloc((float**)&dst_dev, N * sizeof(float));
   // memcpy from host to device
   cudaMemcpy(src_dev, src_cpu, N * sizeof(float), cudaMemcpyHostToDevice);
   // update N;
   dim3 block(32, 32);
   dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
   // gpu compute
   transRow<<<grid, block>>>(src_dev, dst_dev, nx, ny);
   transCol<<<grid, block>>>(src_dev, dst_dev, nx, ny);
   transformNaiveRowDiagonal<<<grid, block>>>(src_dev, dst_dev, nx, ny);
   transformNaiveColDiagonal<<<grid, block>>>(src_dev, dst_dev, nx, ny);
   cudaDeviceSynchronize();
   cudaMemcpy(dst_dev_cpu, dst_dev, N * sizeof(float), cudaMemcpyDeviceToHost);
   // cpu compute
   transformMatrix2D_CPU(src_cpu,  dst_cpu, nx, ny);
   checkResult(dst_cpu, dst_dev_cpu, N);
   // free device
   cudaFree(src_dev);
   cudaFree(dst_dev);

   // free cpu
   free(src_cpu);
   free(dst_cpu);
   free(dst_dev_cpu);
   return 0;
}
