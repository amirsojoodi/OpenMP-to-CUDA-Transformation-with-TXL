//*****************************************************************************
// sample2.c
//
// A simple array initialization in OpenMP
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

int main(){

	int *array;
	int *array2;
    int *array3;
	float *array4;
	int size = SIZE;
	array = (int *)malloc(size * sizeof(int));
	array2 = (int *)malloc(size * sizeof(int));
	array3 = (int *)malloc(size * sizeof(int));
	array4 = (float *)malloc(size * sizeof(float));

	#pragma omp parallel
	for(int i = 0; i < size; i++){
		array[i] = i;
        array2[i] = i+1;
        array2[i] = i+1;
		array4[i] = array2[i] * array3[i];
	}

	// A simple validity test
	printf("Array[%d] = %d", size - 1, array[size - 1]);

	free(array);
    free(array2);
    free(array3);
	free(array4);

	return 0;
}

