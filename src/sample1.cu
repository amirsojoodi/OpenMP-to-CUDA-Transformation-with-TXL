//*****************************************************************************
// sample1.cu
// 
// Expected output of the TXL transformation from sample1.c
//
// Project Description: A TXL transformation from OpenMP C sources to CUDA 
// equivalent. 
//
// For more information on TXL, visit: txl.ca
// Authors: AmirHossein Sojoodi, Nicolas Merz
// Course: ELEC-875 2020, Tom Dean
// Queen's University
//*****************************************************************************

#include<stdlib.h>
#include<omp.h>
#include<stdio.h>

#define SIZE 10000
#define BLOCK_SIZE 64

__global__ void kernel(int *array, int s){
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < s) {
		array[i] = i;
	}
}

int main(){

	int *array;
	int size = SIZE;

	cudaMallocManaged(&array, size * sizeof(int));
	
	// Compute ceil(size/blockSize)
	kernel<<<(size - 1)/BLOCK_SIZE + 1, BLOCK_SIZE>>>(array, size);	
	cudaDeviceSynchronize();
	
	// A simple validity test
	printf("Array[%d] = %d", size - 1, array[size - 1]);

	cudaFree(array);

	return 0;
}

