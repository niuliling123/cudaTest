#include <iostream>
#include <math.h>
__global__ void addSameShape(float* input1, float *input2, float*
output, int num) {
     for (int idx = 0; idx <  num; idx++) {
        output[idx] = input1[idx] + input2[idx];
     }
}

int main(void) {
  int N = 1 << 20;
  float *x, *y, *out;
  cudaMallocManaged(&x, N * sizeof(float));
  cudaMallocManaged(&y, N * sizeof(float));
  cudaMallocManaged(&out, N * sizeof(float));
  for (int idx = 0; idx < N; idx++) {
    x[idx] = 1.0f;
    y[idx] = 2.0f;
  }
  int blockSize = 256;  // blockSize % 32 == 0, blockSize = 256 much better
  int blockNum  = (N + blockSize - 1) / blockSize;

   addSameShape<<<blockNum, blockSize>>>(x, y, out, N);

   cudaDeviceSynchronize();
   float maxError = 0.0f;
   for (int idx = 0; idx < N; idx++) {
     maxError = fmax(maxError, fabs(out[idx] - 3.0f));
   }
   std::cout<<"maxError is "<< maxError << std::endl;
   cudaFree(x);
   cudaFree(y);
   cudaFree(out);
   return 0;
}
