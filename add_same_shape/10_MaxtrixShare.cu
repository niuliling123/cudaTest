#include <iostream>
#include <math.h>
#include <addTest.h>
#define BDIMX 8
#define BDIMY 8
#define IPAD 2
__global__ void transformShareMem(float *matA, float* matB, int nx, int ny) {
   __shared__ float tile[BDIMY][BDIMX];
   unsigned int ix, iy, transform_in_idx, transform_out_idx;

   // 1
   ix = threadIdx.x + blockIdx.x * blockDim.x;
   iy = threadIdx.y + blockIdx.y * blockDim.y;

   transform_in_idx = ix + iy * nx;
   // 2
   unsigned int bidx,irow, icol;
   bidx = threadIdx.y * blockDim.x + threadIdx.x;
   irow = bidx / blockDim.y;
   icol = bidx % blockDim.y;
   // 3
   ix = blockIdx.y * blockDim.y + icol;
   iy = blockIdx.x * blockDim.x + irow;
   // 4
   transform_out_idx = iy * ny + ix;
   if (ix < nx && iy < ny) {
      tile[threadIdx.y][threadIdx.x] = matA[transform_in_idx];
      __syncthreads();
      matB[transform_out_idx] = tile[icol][irow];
   }
}
__global__ void transformShareMemUnroll(float *matA, float* matB, int nx, int ny) {
   __shared__ float tile[BDIMY * BDIMX * 2];
   // 1
   unsigned int ix, iy,ix2,iy2, transform_in_idx, transform_out_idx;
   ix = threadIdx.x + blockIdx.x * blockDim.x * 2;
   iy = threadIdx.y + blockIdx.y * blockDim.y;
   transform_in_idx = ix + iy * nx;
   // 2
   unsigned int bidx,irow, icol;
   bidx = threadIdx.y * blockDim.x + threadIdx.x;
   irow = bidx / blockDim.y;
   icol = bidx % blockDim.y;
   // 3
   ix2 = blockIdx.y * blockDim.y + icol;
   iy2 = blockIdx.x * blockDim.x * 2 + irow;
   // 4
   transform_out_idx = iy2 * ny + ix2;

	if(ix+blockDim.x<nx&& iy<ny)
	{
		unsigned int row_idx=threadIdx.y*(blockDim.x*2)+threadIdx.x;
		tile[row_idx]=matA[transform_in_idx];
		tile[row_idx+BDIMX]=matA[transform_in_idx+BDIMX];
		__syncthreads();
		unsigned int col_idx=icol*(blockDim.x*2)+irow;
        matB[transform_out_idx]=tile[col_idx];
		matB[transform_out_idx+ny*BDIMX]=tile[col_idx+BDIMX];

	}
}
__global__ void transformShareMemUnroll_t(float * in,float* out,int nx,int ny)
{
	__shared__ float tile[BDIMY*(BDIMX*2+IPAD)];


	unsigned int ix,iy,transform_in_idx,transform_out_idx;
	ix=threadIdx.x+blockDim.x*blockIdx.x*2;
    iy=threadIdx.y+blockDim.y*blockIdx.y;
	transform_in_idx=iy*nx+ix;

	unsigned int bidx,irow,icol;
	bidx=threadIdx.y*blockDim.x+threadIdx.x;
	irow=bidx/blockDim.y;
	icol=bidx%blockDim.y;


	unsigned int ix2=blockIdx.y*blockDim.y+icol;
	unsigned int iy2=blockIdx.x*blockDim.x*2+irow;


	transform_out_idx=iy2*ny+ix2;

	if(ix+blockDim.x<nx&& iy<ny)
	{
		unsigned int row_idx=threadIdx.y*(blockDim.x*2+IPAD)+threadIdx.x;
		tile[row_idx]=in[transform_in_idx];
		tile[row_idx+BDIMX]=in[transform_in_idx+BDIMX];
		__syncthreads();
		unsigned int col_idx=icol*(blockDim.x*2+IPAD)+irow;
        out[transform_out_idx]=tile[col_idx];
		out[transform_out_idx+ny*BDIMX]=tile[col_idx+BDIMX];

	}

}
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
  int dimx=BDIMX;
  int dimy=BDIMY;
  dim3 block(dimx,dimy);
  dim3 grid((nx-1)/block.x+1,(ny-1)/block.y+1);
  dim3 block_1(dimx,dimy);
  dim3 grid_1((nx-1)/(block_1.x*2)+1,(ny-1)/block_1.y+1);
   // gpu compute
   transRow<<<grid, block>>>(src_dev, dst_dev, nx, ny);
   cudaDeviceSynchronize();
   transCol<<<grid, block>>>(src_dev, dst_dev, nx, ny);
   cudaDeviceSynchronize();
   transformNaiveRowDiagonal<<<grid, block>>>(src_dev, dst_dev, nx, ny);
   cudaDeviceSynchronize();
   transformNaiveColDiagonal<<<grid, block>>>(src_dev, dst_dev, nx, ny);
   cudaDeviceSynchronize();
   transformShareMem<<<grid, block>>>(src_dev, dst_dev, nx, ny);
   cudaDeviceSynchronize();
   transformShareMemUnroll<<<grid_1, block_1>>>(src_dev, dst_dev, nx, ny);
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
