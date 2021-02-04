#include <iostream>
#include <math.h>
enum kernel_t {
  MEM_ADDR = 0,
  GRID_BLOCK = 1,
  MEM_SHARE = 2
};
__global__ void addEffectOfaddr(float* src1, float *src2, float*
dst, size_t num) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;;
  if (idx < num)
    dst[idx] = src1[idx] + src2[idx];
}
void checkResult(float * hostRef,float * gpuRef,const int N)
{
  double epsilon=1.0E-8;
  for(int i=0;i<N;i++)
  {
    if(abs(hostRef[i]-gpuRef[i])>epsilon)
    {
      printf("Results don\'t match!\n");
      printf("%f(hostRef[%d] )!= %f(gpuRef[%d])\n",hostRef[i],i,gpuRef[i],i);
      return;
    }
  }
  printf("Check result success!\n");
}
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
      addEffectOfaddr<<<grid, block>>>(src1, src2, dst, N);
      break;
    case 1:
      addEffectOfaddr<<<grid, block>>>(src1, src2, dst, N);
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
    launchKernel(src1, src2, dst, num, type, grid, block);
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
void initialData(float* ip,int size)
{
  time_t t;
  srand((unsigned )time(&t));
  for(int i=0;i<size;i++)
  {
    ip[i]=(float)(rand()&0xffff)/1000.0f;
  }
}

int main(void) {
   // set num of input , offset of src and dst, and loop_num and kernelType
   size_t N = 1 << 24;
   int loop_num = 1000;
   int offset_src1 = 13;
   int offset_src2 = 13;
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
   int block = 512;
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
