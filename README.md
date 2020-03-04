## OpenMP-to-CUDA Transformation with TXL
### Overview

In this project we are going to transform a simplified C OpenMP source code to its CUDA version and vice versa using [TXL]( https://www.txl.ca/) language. We have provided some specifications and scope analysis in the following section.

## Specification

There are multiple research and projects on source-to-source compilation from OpenMP or in general C programs to CUDA source codes. <sup>[1](#myfootnote1)</sup> <sup>[2](#myfootnote2)</sup> <sup>[3](#myfootnote3)</sup> As it can be seen in the first picture, on the left side, an OpenMP example is written to initialize an array of integers, and on the right side, there is its CUDA equivalent. Basically, every CUDA source code consists these four main parts:
1. Data allocation on device (GPU) memory
2. Copy input data from host memory to device memory <sup>[4](#myfootnote4)</sup>
3. GPU computation function, which is called Kernel.
4. Copy output data back to Host memory from GPU memory

The NVIDIA GPUs that have compute capability<sup>[5](#myfootnote5)</sup> higher than 3.5 have a feature called UnifiedMemory<sup>[6](#myfootnote6)</sup> that enables developers to access the same data from host and device without worrying about where the data is located at the moment. To narrow the scope of this project, we will focus on the GPUs that have this feature.  
Some specifications are taken into account:
1. The OpenMP source code has correct syntax and it is parallelizable and does not have any semantic problem due to any race condition.
2. We want to provide the capability to transform these features of OpenMP:

|#|OpenMP directives|CUDA possible choice(s)|
|:-|:---|:---|
|1|`#pragma omp parallel`\n `#pragma omp parallel for`|A CUDA Kernel launch|
|2|`#pragma omp barrier`|`__syncthreads();` or `cudaDeviceSynchronize();`|
|3|`#pragma omp parallel shared(variable)`|Shared memory or Global memory to store the shared variable.|
|4|`#pragma omp parallel private(variable)`|Local memory or Global memory to store the local variable.|
|5|`#pragma omp atomic`|CUDA atomic instructions|
|6|`omp_get_thread_num()`|Calculate thread Id: `threadIdx.x + blockDim.x * blockId.x`|

## Milestones
- Write C grammar extension for OpenMP
- Tacke each of the directives and instructions of OpenMP discribed in the table
- Transform simple With Unified Memory
- Without Unified Memory

## Possiblities
- Transform CUDA code back to OpenMP source code


## References

<a name="myfootnote1">1</a>: Gabriel Noaje, Christophe Jaillet, Michael Krajeck, "Source-to-source code translator: OpenMP C to CUDA", 2011

<a name="myfootnote2">2</a>: kihiro Tabuchi, Masahiro Nakao, and Mitsuhisa Sato, "A Source-to-Source OpenACC Compiler for CUDA", 2014

<a name="myfootnote3">3</a>: Nugteren, C. ; Corporaal, H. "Bones : an automatic skeleton-based C-to-CUDA compiler for GPUs." 2014

<a name="myfootnote4">4</a>: Note that in the provided sample, step 2 and 4 is done automatically by some GPUs at runtime because of UnifiedMemory feature. 

<a name="myfootnote5">5</a>: Device capabilities and their features can be found [here](https://en.wikipedia.org/wiki/CUDA).

<a name="myfootnote6">6</a>: More information can be found [here](https://devblogs.nvidia.com/unified-memory-in-cuda-6/).

