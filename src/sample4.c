//*****************************************************************************
// sample3.c
//
// A simple array initialization in OpenMP
//
// Expectations:
//		- Create Kernel function with appropriate header
//		- Pass all the referenced variables within the for loop to the kernel
//		- Maintain the order of arguments in kernel header and kernel call
//		- Remove duplicate uses of the variables in the kernel header and call
//		- Do not pass defined variables
//		- Manage allocation/deallocations properly
//		- Do not transform for loops without preprocessor directives
//		- Do not pass local variables defined in the loop body
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

int main(){

	int *array1;
	int *array2;
    double *array3;
	float *array4;
	int *array5;
	double foo;

	int size = SIZE;
	array1 = (int *)malloc(size * sizeof(int));
	array2 = (int *)malloc(size * sizeof(int));
	array3 = (double *)malloc(size * sizeof(double));
	array4 = (float *)malloc(size * sizeof(float));
	array5 = (int *)malloc(size * sizeof(int));	
	foo = PI * 2;
	#pragma omp parallel
	for(int i = 0; i < size; i++){
		array1[i] = i;
        array2[i] = array3[i] + 1;
		int tmp = array2[i] + 10;
		array4[i] = foo * tmp;
	}

	for(int i = 0; i < size; i++){
		array3[i] = 0.1;
		array4[i] = i;
	}

	// A simple validity test
	printf("Array[%d] = %d", size - 1, array1[size - 1]);

	free(array1);
    free(array2);
    free(array3);
	free(array4);
	free(array5);

	return 0;
}

