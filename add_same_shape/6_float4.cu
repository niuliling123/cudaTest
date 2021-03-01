#include <iostream>
#include <math.h>
#include <addTest.h>
#include <thrust/device_vector.h>


// gpu
__global__ void addEffectOfFloat4_v2(float* src1, float *src2, float*
dst, size_t num) {
  int idx = 4 * (threadIdx.x + blockIdx.x * blockDim.x);
  if (idx < num) {
    float4* a1 = (float4 *)(src1 + idx);
    float4* b1 = (float4 *)(src2 + idx);
    float4* c1 = (float4 *)(dst + idx);
    c1->x = a1->x + b1->x;
    c1->y = a1->y + b1->y;
    c1->z = a1->z + b1->z;
    c1->w = a1->w + b1->w;
  } 
}
__global__ void addEffectOfFloat2_v2(float* src1, float *src2, float*
dst, size_t num) {
  int idx = 2 * (threadIdx.x + blockIdx.x * blockDim.x);
  if (idx < num) {
    float2* a1 = (float2 *)(src1 + idx);
    float2* b1 = (float2 *)(src2 + idx);
    float2* c1 = (float2 *)(dst + idx);
    c1->x = a1->x + b1->x;
    c1->y = a1->y + b1->y;
  } 
}
// gpu
__global__ void addEffectOfFloat2_v(float* src1, float *src2, float*
dst, size_t num) {
  int idx = 2 * (threadIdx.x + blockIdx.x * blockDim.x);
  if (idx < num) {
    float2 a1;
    float2 b1;
    a1.x = src1[idx];
    b1.x = src2[idx];

    a1.y = src1[idx + 1];
    b1.y = src2[idx + 1];
    
    float2 c1;
    c1.x = a1.x + b1.x;
    c1.y = a1.y + b1.y;

    dst[idx] = c1.x;
    dst[idx + 1] = c1.y;
  } 
}
// GPU
__global__ void addEffectOfFloat2(float2* src1, float2 *src2, float2*
dst, size_t num) {
  int idx = 2 * (threadIdx.x + blockIdx.x * blockDim.x);
  if (idx < num) {
    float2 a1 = src1[idx];
    float2 b1 = src2[idx];

    float2 a2 = src1[idx + 1];
    float2 b2 = src2[idx + 1];
    
    float2 c1;
    c1.x = a1.x + b1.x;
    c1.y = a1.y + b1.y;

    float2 c2;
    c2.x = a2.x + b2.x;
    c2.y = a2.y + b2.y;
    
    dst[idx] = c1;
    dst[idx + 1] = c2;
  } 
}
// f
__global__ void addEffectOfaddr(float* src1, float *src2, float*
dst, size_t num) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;;
  if (idx < num)
    dst[idx] = src1[idx] + src2[idx];
}
// CPU 
void sumArrays(float * a,float * b,float * res, const int size)
{
    for(int i=0 ;i <size;i++) {
        res[i]=a[i]+b[i];
    }
}
// test the effect of first address alignment on kernel performancet 


void launchKernel(float* src1, float * src2, float *dst, size_t N, int type, int grid, int block) {
  
  switch(type) {
    case 0:
      addEffectOfFloat2<<<grid / 2 , block>>>((float2 *)thrust::raw_pointer_cast(src1), (float2 *)thrust::raw_pointer_cast(src2), (float2 *)thrust::raw_pointer_cast(dst), N);
      break;
    case 1:
      addEffectOfaddr<<<grid, block>>>(src1, src2, dst, N);
      break;
    case 2:
      addEffectOfFloat2_v<<<grid / 2 , block>>>(src1,src2, dst, N);
      break;
    case 3:
      addEffectOfFloat2_v2<<<grid / 2 , block>>>(src1,src2, dst, N);
      break;
    case 4:
      addEffectOfFloat4_v2<<<grid / 4 , block>>>(src1,src2, dst, N);
      break;
    default:
      std::cout<<" this is default in launchKernel" << std::endl;
      break;
  }
}
void compute(float *src1, float * src2, float * dst, size_t num, int loop, kernel_t type, int grid, int block) {
  cudaEvent_t start, stop;
  float time;
  float total_time = 0.0f;
  float max_time = 0.0f;
  float min_time = (1 << 20) * 1.0f;
  for (int i = 0; i < loop; i++) {
    time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    launchKernel(src1, src2, dst, num, 1, grid, block);
    launchKernel(src1, src2, dst, num, 0, grid, block);
    launchKernel(src1, src2, dst, num, 2, grid, block);
    launchKernel(src1, src2, dst, num, 3, grid, block);
    launchKernel(src1, src2, dst, num, 4, grid, block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop );
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    max_time = fmax(max_time, time);
    min_time = fmin(min_time, time);
    total_time += time;
  }
  std::cout<<"[kernle_type]: " << type <<std::endl;
  std::cout<<"[AVG]: " << (total_time * 1.0f) / loop <<" [max]: " << max_time <<" [MIN]: " << min_time <<std::endl;
}

int main(void) {
   // set num of input , offset of src and dst, and loop_num and kernelType
   size_t N = 1 << 24;
   int loop_num = 1000;
   int offset_src1 = 0;
   int offset_src2 = 0;
   int offset_dst = 0;
   kernel_t kernel_type = MEM_ADDR;
   // device
   int dev = 0;
   cudaSetDevice(dev);
   // malloc for host
   float *src1_cpu = (float *)malloc(N * sizeof(float));
   float *src2_cpu = (float *)malloc(N * sizeof(float));
   float *dst_cpu = (float *)malloc(N * sizeof(float));
   float *dst_dev_cpu = (float *)malloc(N * sizeof(float));
   memset(dst_cpu, 0, N * sizeof(float));
   memset(dst_dev_cpu, 0, N * sizeof(float));
   // init host
   initialData(src1_cpu, N);
   initialData(src2_cpu, N);
   // malloc for device
   float *src1_dev; 
   float *src2_dev; 
   float *dst_dev;
   cudaMalloc((float**)&src1_dev, N * sizeof(float));
   cudaMalloc((float**)&src2_dev, N * sizeof(float));
   cudaMalloc((float**)&dst_dev, N * sizeof(float));
   // memcpy from host to device
   cudaMemcpy(src1_dev, src1_cpu, N * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(src2_dev, src2_cpu, N * sizeof(float), cudaMemcpyHostToDevice);
   // update N;
   size_t count = N - 128;
   int block = 1024;
   int grid = (count + block - 1) / block;
   // gpu compute
   compute(src1_dev + offset_src1, src2_dev + offset_src2, dst_dev + offset_dst, count, loop_num,  kernel_type, grid, block);
   cudaDeviceSynchronize();
   cudaMemcpy(dst_dev_cpu, dst_dev, N * sizeof(float), cudaMemcpyDeviceToHost);
   // cpu compute
   sumArrays(src1_cpu + offset_src1, src2_cpu + offset_src2, dst_cpu + offset_dst, count);
   checkResult(dst_cpu + offset_dst, dst_dev_cpu + offset_dst, count);
   // free device
   cudaFree(src1_dev);
   cudaFree(src2_dev);
   cudaFree(dst_dev);

   // free cpu
   free(src1_cpu);
   free(src2_cpu);
   free(dst_cpu);
   free(dst_dev_cpu);
   return 0;
}
