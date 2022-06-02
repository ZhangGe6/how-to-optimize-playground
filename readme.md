
 # my_HPC_from_scratch

 This repo records my practices when learning High-Performance Computation (HPC). Currently, the following topics are elaborated:

 - x86
   - float32 GEMM optimization [click for more details](./x86/x86_float_gemm/)

   - double GEMM optimization [click for more details](./x86/x86_double_gemm/)

 - CUDA 
   - float32 GEMM optimization [click for more details](./cuda/cuda_float_gemm/)


The following figures show the effects of optimizations, respectively.
![](./x86/x86_float_gemm/res/cur_all.png)

![](./cuda/cuda_float_gemm/res/cur_all.png)

 

>  For HPC, I am still a (very) freshman. Any issues are welcomed!

# Credits
- [BBuf/how-to-optimize-gemm](https://github.com/BBuf/how-to-optimize-gemm)
- [Cjkkkk/CUDA_gemm](https://github.com/Cjkkkk/CUDA_gemm)
- [Liu-xiandong/How_to_optimize_in_GPU](https://github.com/Liu-xiandong/How_to_optimize_in_GPU/tree/master/sgemm)