#*******************************************************************
# sample.cu
# 
# A simple array initialization in CUDA
#*******************************************************************

#include<stdlib.h>
#include<omp.h>
#include<stdio.h>

#define SIZE 10000
#define BLOCK_SIZE 64

__global__ void kernel(int *array, int size){
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < size) {
		array[i] = i;
	}
}

int main(){

	int *array;
	int size = SIZE;

	cudaMallocManaged(&array, size * sizeof(int));
	
	dim3 blockDime(BLOCK_SIZE);
	// Compute ceil(size/blockSize)
	dim3 gridDime((size - 1)/BLOCK_SIZE + 1); 

	kernel<<<gridDime, blockDime>>>(array, size);	
	cudaDeviceSynchronize();
	
	// A simple validity test
	printf("Array[%d] = %d", size - 1, array[size - 1]);

	cudaFree(array);

	return 0;
}

