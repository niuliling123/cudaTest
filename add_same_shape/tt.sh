#!/bin/bash
#nvcc BlockingReduce.cu -Xptxas -dlcm=cg -o yes
nvcc BlockingReduce.cu -o yes
nvprof --metrics gld_efficiency,gst_efficiency,gld_transactions,gst_transactions ./yes
