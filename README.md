# cudaTest

介绍： 本repo用于个人cuda编程入门练习。

目录介绍：
  add_same_shape 文件夹.
  
     ├── add_op_same_shape
  
     ├── add_op_same_shape.cu
     
     └── build_cuda.sh


功能介绍：实现相同shape的add op的基本功能

运行介绍：
   启动一个GPU版本的docker，在add_sample_shape文件夹中执行build_cuda.sh 脚本会在当前路径下生成可执行文件add_op_same_shape
   直接运行add_op_same_shape即可。

运行结果：
   运行后会打印出：add_op_same_shape程序中最大误差，当误差为0表明程序运行正确，反之则程序运行错误
