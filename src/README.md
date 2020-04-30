## A simple program in OpenMP

A TXL transformation from OpenMP C sources to CUDA equivalent.

Expectations:
      - Create Kernel function with appropriate header
      - Pass all the referenced variables within the for loop to the kernel
      - Maintain the order of arguments in kernel header and kernel call
      - Remove duplicate uses of the variables in the kernel header and call
      - Do not pass defined variables
      - Manage allocation/deallocations properly
      - Do not transform for loops without preprocessor directives
      - Do not pass local variables defined in the loop body

For more information on TXL, visit: txl.ca
