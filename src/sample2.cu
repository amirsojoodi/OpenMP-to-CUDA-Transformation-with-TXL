//*****************************************************************************
// sample2.cu
// 
// Expected output of the TXL transformation from sample2.c
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
#define DUMMY 1369
#define BLOCK_SIZE 64

__global__ void kernel(int *array1, int *array2, float *array3, int s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < s) {
        array1[i] = i;
        array2[i] = array1[i] + 1;
        array2[i] = array2[i] + DUMMY;
        array3[i] = array2[i] / (array1[i] + 0.5);
    }
}


int main(){

	int *array1;
    int *array2;
    float *array3;
    float *array4;
    int size = SIZE;
	
    cudaMallocManaged(&array1, size * sizeof (int));
    cudaMallocManaged(&array2, size * sizeof (int));
    cudaMallocManaged(&array3, size * sizeof (int));
    array4 = (float *) malloc (size * sizeof (float));
	
    kernel<<<((size) - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(array1, array2, array3, size);
    cudaDeviceSynchronize();
	
    printf ("Array3[%d] = %f", size - 1, array3[size - 1]);
    
	cudaFree(array1);
    cudaFree(array2);
    cudaFree(array3);
    free(array4);
    
	return 0;
}

