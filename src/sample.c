//*****************************************************************************
// sample.c
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
	int size = SIZE;
	array = (int *)malloc(size * sizeof(int));

	#pragma omp parallel
	for(int i = 0; i < size; i++){
		array[i] = i;
	}

	// A simple validity test
	printf("Array[%d] = %d", size - 1, array[size - 1]);

	free(array);

	return 0;
}

