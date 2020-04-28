//*****************************************************************************
// sample4.cu
// 
// Expected output of the TXL transformation from sample4.c
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
#define PI 3.14
#define BLOCK_SIZE 64

__global__ void kernel (int *array1, int *array2, double *array3, float *array4, double foo, int s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < s) {
        array1[i] = i;
        array2[i] = array3[i] + 1;
        int tmp = array2[i] + 10;
        array4[i] = foo * tmp;
    }
}

int main(){

	int *array1;
    int *array2;
    double *array3;
    float *array4;
    int *array5;
    double foo;
    int size = SIZE;
	
    cudaMallocManaged (&array1, size * sizeof(int));
    cudaMallocManaged (&array2, size * sizeof(int));
    cudaMallocManaged (&array3, size * sizeof(double));
    cudaMallocManaged (&array4, size * sizeof(float));
    array5 = (int *) malloc (size * sizeof(int));
    foo = PI * 2;
	
    kernel<<<((size) - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(array1, array2, array3, array4, foo, size);
    cudaDeviceSynchronize ();
    
	for (int i = 0; i < size; i++) {
        array3[i] = 0.1;
        array4[i] = i;
    }
	
    printf ("Array[%d] = %d", size - 1, array1[size - 1]);
    cudaFree(array1);
    cudaFree(array2);
    cudaFree(array3);
    cudaFree(array4);
    free(array5);
    
	return 0;
}

