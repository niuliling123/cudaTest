#!/bin/bash
nvcc add_op_same_shape.cu -o add_op_same_shape
nvcc -O3 -arch=sm_35 -Xptxas -dlcm=cg -I . 2_getGPU_CPU_time.cu -o yes
