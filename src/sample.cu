#include<stdlib.h>
#include<omp.h>
#include<stdio.h>

#define SIZE 10000

__global__ void kernel(int *array, int size){
	
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id < size) {
		array[id] = id;
	}
}

int main(){

	int *array;
	int size = SIZE;

	cudaMallocManaged(&array, size * sizeof(int));
	
	dim3 blockDime(blockSize);
	// Compute ceil(size/blockSize)
	dim3 gridDime((size - 1)/blockSize + 1); 

	kernel<<<gridDime, blockDime>>>(array, size);	
	cudaDeviceSynchronize();
	
	// A simple validity test
	printf("Array[%d] = %d", size - 1, array[size - 1]);

	cudaFree(array);

	return 0;
}

